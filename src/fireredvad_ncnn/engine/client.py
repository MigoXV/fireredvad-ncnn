"""Bounded asyncio bridge to the native engine, without a child process.

The bridge thread owns every native call. asyncio owns Future delivery; a
request keeps its admission lease until both native completion and consumption.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fireredvad_ncnn.configs.engine_config import EngineConfig
from fireredvad_ncnn.configs.runtime_config import RuntimeConfig
from fireredvad_ncnn.configs.stream_vad_config import FireRedStreamVadConfig, resolve_postprocessing_config
from fireredvad_ncnn.engine.errors import (
    EngineError, InvalidArgumentError, ResourceExhaustedError,
    DeadlineExceededError, FailedPreconditionError, UnavailableError,
)
from fireredvad_ncnn.native import NativeEngine, NativeError

logger = logging.getLogger(__name__)
_NATIVE_ERRORS = {1: InvalidArgumentError, 2: ResourceExhaustedError,
                  3: DeadlineExceededError, 4: FailedPreconditionError,
                  5: UnavailableError, 7: FailedPreconditionError}


def _error(exc: Exception) -> Exception:
    if isinstance(exc, NativeError):
        return _NATIVE_ERRORS.get(exc.status, EngineError)(str(exc))
    if isinstance(exc, ValueError):
        return InvalidArgumentError(str(exc))
    return exc


@dataclass(eq=False)
class _Request:
    operation: str
    future: asyncio.Future
    deadline: float
    fields: dict[str, Any]
    payload: bytes = b""
    size: int = 0
    result_bytes: int = 0
    canceled: threading.Event = field(default_factory=threading.Event)
    native_id: int = 0
    cancel_sent: bool = False
    notified: bool = False  # Bridge thread only.
    completed: bool = False  # asyncio loop only.
    consumed: bool = False  # asyncio loop only.


class EngineClient:
    def __init__(self, model_path: str, config: EngineConfig, *,
                 runtime_config: RuntimeConfig | None = None,
                 stream_vad_config: FireRedStreamVadConfig | None = None,
                 max_payload_bytes: int = 100 * 1024 * 1024):
        self.model_path = model_path
        self.config = config
        self.runtime_config = runtime_config or RuntimeConfig()
        self.stream_vad_config = stream_vad_config or FireRedStreamVadConfig()
        self.max_payload_bytes = max_payload_bytes
        self.generation = uuid.uuid4().hex
        self._condition = threading.Condition()
        self._queue: deque[_Request] = deque()
        self._requests: set[_Request] = set()
        self._bytes = 0
        self._session_names: dict[str, str] = {}
        # Cleanup is separate from data admission, bounded by accepted sessions.
        self._cleanup: dict[tuple[str, str], concurrent.futures.Future] = {}
        self._sessions: dict[tuple[str, str], int] = {}  # Bridge thread only.
        self._pending: dict[int, _Request] = {}  # Bridge thread only.
        self._ready: concurrent.futures.Future = concurrent.futures.Future()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._failure: Exception | None = None
        self._snapshot: dict[str, Any] = {}

    @property
    def failure(self) -> Exception | None:
        with self._condition:
            return self._failure

    async def start(self) -> None:
        if self._closed:
            raise UnavailableError("engine is closed")
        if self._thread is None:
            self._loop = asyncio.get_running_loop()
            self._thread = threading.Thread(target=self._run, name="fireredvad-bridge")
            self._thread.start()
        ready = asyncio.wrap_future(self._ready)
        # Observe late startup failures even if the startup deadline has elapsed.
        ready.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        await asyncio.wait_for(asyncio.shield(ready), self.config.startup_timeout_s)

    async def health(self) -> dict[str, Any]:
        with self._condition:
            if self._closed or self._failure or not self._thread or not self._thread.is_alive() or not self._ready.done():
                raise UnavailableError("native engine is not running")
            return {**self._snapshot, "pid": os.getpid(), "generation": self.generation,
                    "adapter_pending": len(self._requests), "adapter_inflight_bytes": self._bytes}

    async def _request(self, operation: str, *, payload: bytes = b"",
                       timeout_s: float | None = None, deadline_s: float | None = None,
                       **fields):
        if len(payload) > self.max_payload_bytes:
            raise ResourceExhaustedError("audio payload exceeds configured maximum")
        if len(payload) % 2:
            raise InvalidArgumentError("LINEAR16 payload must contain an even number of bytes")
        timeout = timeout_s if timeout_s is not None else self.config.queue_timeout_s
        deadline = deadline_s if deadline_s is not None else timeout
        request = _Request(operation, asyncio.get_running_loop().create_future(),
                           time.monotonic() + deadline, fields, payload, len(payload))
        with self._condition:
            if self._closed or self._failure or not self._thread or not self._thread.is_alive():
                raise UnavailableError("engine is not running")
            if len(self._requests) >= self.config.ingress_capacity or self._bytes + request.size > self.config.max_inflight_bytes:
                raise ResourceExhaustedError("engine admission capacity reached")
            if fields.get("session_id"):
                epoch = fields.get("epoch", self._session_names.get(fields["session_id"]))
                if epoch is None or self._session_names.get(fields["session_id"]) != epoch:
                    raise FailedPreconditionError("unknown stream session")
                fields["epoch"] = epoch
            self._requests.add(request)
            self._bytes += request.size
            self._queue.append(request)
            self._condition.notify()
        try:
            return await asyncio.wait_for(asyncio.shield(request.future), timeout)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            request.canceled.set()
            request.future.cancel()
            if fields.get("session_id"):
                self._queue_cleanup(fields["session_id"], fields.get("epoch"))
            with self._condition:
                self._condition.notify()
            raise
        finally:
            if request.future.done() and not request.future.cancelled():
                request.future.exception()  # Also observe completion/cancel races.
            request.consumed = True
            self._release(request)

    def _release(self, request):
        if request.completed and request.consumed:
            with self._condition:
                if request in self._requests:
                    self._requests.remove(request)
                    self._bytes -= request.size + request.result_bytes

    def _settle(self, request, result, error):
        request.completed = True
        # An open/chunk canceled just before delivery must not leave a session.
        if (error or request.canceled.is_set()) and request.fields.get("session_id"):
            self._queue_cleanup(request.fields["session_id"], request.fields.get("epoch"))
        if not request.future.done():
            if error is not None:
                request.future.set_exception(error)
            else:
                request.future.set_result(result)
        self._release(request)

    def _finish(self, request, result=None, error=None):
        if request.notified:
            return
        request.notified = True
        if result is not None:
            # Bound retained Python event objects as well as input bytes.
            size = len(result) * 160 if isinstance(result, list) else 0
            with self._condition:
                if self._bytes + size > self.config.max_inflight_bytes:
                    error = ResourceExhaustedError("engine result capacity reached")
                    result = None
                else:
                    request.result_bytes = size
                    self._bytes += size
        self._loop.call_soon_threadsafe(self._settle, request, result, error)

    async def detect(self, audio: bytes, sample_rate: int, postprocessing=None):
        if not audio:
            raise InvalidArgumentError("detect audio must be non-empty")
        return await self._request("detect", payload=audio, sample_rate=sample_rate,
            postprocessing=postprocessing, deadline_s=self.config.detect_queue_timeout_s,
            timeout_s=max(300., self.config.detect_queue_timeout_s))

    async def open_stream(self, session_id: str, endpoint_only: bool, postprocessing=None):
        with self._condition:
            if self._closed or self._failure or not self._thread or not self._thread.is_alive():
                raise UnavailableError("engine is not running")
            if not session_id or session_id in self._session_names:
                raise FailedPreconditionError("stream session already exists or has an empty ID")
            if len(self._session_names) >= self.config.max_sessions:
                raise ResourceExhaustedError("maximum session capacity reached")
            epoch = uuid.uuid4().hex
            self._session_names[session_id] = epoch
        try:
            await self._request("open", session_id=session_id, endpoint_only=endpoint_only,
                                postprocessing=postprocessing, epoch=epoch)
        except BaseException:
            self._queue_cleanup(session_id, epoch)
            raise

    async def stream_chunk(self, session_id, sequence, audio, sample_rate, is_final):
        with self._condition:
            epoch = self._session_names.get(session_id)
        if epoch is None:
            raise FailedPreconditionError("unknown stream session")
        try:
            return await self._request("chunk", payload=audio, session_id=session_id,
                sequence=sequence, sample_rate=sample_rate, is_final=is_final, epoch=epoch)
        except BaseException:
            self._queue_cleanup(session_id, epoch)
            raise

    def _queue_cleanup(self, session_id, epoch=None):
        with self._condition:
            current = self._session_names.get(session_id)
            if current is None or (epoch is not None and current != epoch):
                return None
            key = (session_id, current)
            future = self._cleanup.get(key)
            if future is None:
                future = concurrent.futures.Future()
                self._cleanup[key] = future
            self._condition.notify()
            return future

    async def close_stream(self, session_id, sequence=0):
        future = self._queue_cleanup(session_id)
        if future is not None:
            await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), self.config.queue_timeout_s)

    async def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        deadline = time.monotonic() + self.config.shutdown_grace_s
        while self._thread and self._thread.is_alive():
            if time.monotonic() >= deadline:
                raise TimeoutError("native engine did not stop within shutdown grace period")
            await asyncio.sleep(.01)
        if self._thread:
            self._thread.join()
        # Flush bridge callbacks before returning from close.
        await asyncio.sleep(0)

    def _drop_session(self, engine, key):
        native = self._sessions.pop(key, None)
        if native is not None:
            for request in self._pending.values():
                if (request.fields.get("session_id"), request.fields.get("epoch")) == key and not request.cancel_sent:
                    engine.cancel(request.native_id)
                    request.cancel_sent = True
            engine.session_close(native)
        with self._condition:
            name, epoch = key
            if self._session_names.get(name) == epoch:
                self._session_names.pop(name)

    def _execute(self, engine, request):
        try:
            if request.canceled.is_set():
                raise FailedPreconditionError("request canceled before submission")
            remaining = request.deadline - time.monotonic()
            if remaining <= 0:
                raise DeadlineExceededError("request expired before submission")
            fields = request.fields
            if request.operation == "open":
                name = fields["session_id"]
                key = (name, fields["epoch"])
                with self._condition:
                    closing = key in self._cleanup or self._session_names.get(name) != fields["epoch"]
                if closing:
                    raise FailedPreconditionError("session is closing")
                post = resolve_postprocessing_config(self.stream_vad_config, fields.get("postprocessing"))
                self._sessions[key] = engine.session_open(post, fields["endpoint_only"])
                self._finish(request)
                return
            native = 0
            if request.operation == "chunk":
                native = self._sessions.get((fields["session_id"], fields["epoch"]), 0)
                if not native:
                    raise FailedPreconditionError("unknown stream session")
            post = None if native else resolve_postprocessing_config(self.stream_vad_config, fields.get("postprocessing"))
            request.native_id = engine.submit(np.frombuffer(request.payload, dtype="<i2"), fields["sample_rate"],
                session=native, sequence=fields.get("sequence", 0),
                is_final=fields.get("is_final", request.operation == "detect"),
                postprocessing_config=post, timeout_ms=min(0xFFFFFFFF, max(1, math.ceil(remaining * 1000))))
            self._pending[request.native_id] = request
        except (EngineError, NativeError, ValueError) as exc:
            if request.fields.get("session_id"):
                self._drop_session(engine, (request.fields["session_id"], request.fields.get("epoch")))
            self._finish(request, error=_error(exc))
        finally:
            request.payload = b""  # frv_submit owns its input copy after return.

    def _run(self):
        engine = None
        failure = None
        try:
            runtime = self.runtime_config
            engine = NativeEngine(self.model_path, cpu_budget=runtime.num_threads,
                worker_count=runtime.worker_count, threads_per_worker=runtime.effective_threads_per_worker,
                chunk_frames=runtime.stream_chunk_frame, engine_config=self.config)
            with self._condition:
                self._snapshot = {"native": engine.build_info, "metrics": engine.metrics()}
            self._ready.set_result(None)
            logger.info("Native engine ready: pid=%s generation=%s workers=%s threads=%s native=%s",
                        os.getpid(), self.generation, runtime.worker_count,
                        runtime.effective_threads_per_worker, engine.build_info)
            last_metrics = time.monotonic()
            while True:
                with self._condition:
                    if self._closed:
                        break
                    cleanup = list(self._cleanup.items())
                    commands = [self._queue.popleft() for _ in range(min(64, len(self._queue)))]
                for name, future in cleanup:
                    self._drop_session(engine, name)
                    with self._condition:
                        self._cleanup.pop(name, None)
                    future.set_result(None)
                for request in commands:
                    self._execute(engine, request)
                for request in self._pending.values():
                    if request.canceled.is_set() and not request.cancel_sent:
                        engine.cancel(request.native_id)
                        request.cancel_sent = True
                for index in range(64):
                    completion = engine.poll(wait_ms=1 if index == 0 and self._pending else 0)
                    if completion is None:
                        break
                    request = self._pending.pop(completion.request_id)
                    error = None
                    if time.monotonic() >= request.deadline:
                        error = DeadlineExceededError("request deadline exceeded")
                    elif completion.status:
                        error = _error(NativeError(completion.status, completion.message))
                    if error and request.fields.get("session_id"):
                        self._drop_session(engine, (request.fields["session_id"], request.fields.get("epoch")))
                    events = completion.events if request.operation == "chunk" else [e[:2] for e in completion.events]
                    self._finish(request, result=events if error is None else None, error=error)
                if time.monotonic() - last_metrics >= self.config.metrics_interval_s:
                    last_metrics = time.monotonic()
                    metrics = engine.metrics()
                    with self._condition:
                        self._snapshot["metrics"] = metrics
                        metrics = {**metrics, "adapter_pending": len(self._requests), "adapter_inflight_bytes": self._bytes}
                    logger.info("engine_metrics=%s", json.dumps(metrics, sort_keys=True))
                # Do not retain consumed event lists through the idle wait.
                commands.clear()
                request = completion = events = None
                with self._condition:
                    if not self._queue and not self._pending and not self._cleanup and not self._closed:
                        self._condition.wait(timeout=self.config.metrics_interval_s)
        except Exception as exc:
            failure = _error(exc)
            with self._condition:
                self._failure = failure
            if not self._ready.done():
                self._ready.set_exception(failure)
            logger.exception("Native engine bridge failed")
        finally:
            try:
                if engine is not None:
                    try:
                        logger.info("engine_metrics_final=%s", json.dumps(engine.metrics(), sort_keys=True))
                    finally:
                        engine.close()
            except Exception as exc:
                failure = failure or exc
                with self._condition:
                    self._failure = failure
                logger.exception("Native engine cleanup failed")
            with self._condition:
                requests = list(self._requests)
                cleanup = list(self._cleanup.values())
                self._queue.clear()
                self._cleanup.clear()
                self._session_names.clear()
                self._sessions.clear()
                self._pending.clear()
            for request in requests:
                self._finish(request, error=UnavailableError(str(failure or "engine closed")))
            for future in cleanup:
                future.set_result(None)
            logger.info("Native engine stopped")
