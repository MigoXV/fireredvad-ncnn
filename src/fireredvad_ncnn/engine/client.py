from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from typing import Any

import zmq
import zmq.asyncio

from fireredvad_ncnn.configs.engine_config import EngineConfig
from fireredvad_ncnn.engine.protocol import (
    PROTOCOL_VERSION,
    EngineError,
    Operation,
    ResourceExhaustedError,
    UnavailableError,
    pack_metadata,
    raise_for_response,
    unpack_metadata,
)

logger = logging.getLogger(__name__)


class EngineClient:
    def __init__(self, endpoint: str, config: EngineConfig) -> None:
        self.endpoint = endpoint
        self.config = config
        self.generation = ""
        self._context: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None
        self._send_queue: asyncio.Queue[tuple[dict[str, Any], bytes] | None] = (
            asyncio.Queue(maxsize=config.ingress_capacity)
        )
        self._pending: dict[str, asyncio.Future] = {}
        self._pending_bytes: dict[str, int] = {}
        self._inflight_bytes = 0
        self._reader_task: asyncio.Task | None = None
        self._writer_task: asyncio.Task | None = None
        self._closed = False

    async def start(self) -> None:
        if self._socket is not None:
            return
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.IDENTITY, uuid.uuid4().hex.encode())
        self._socket.setsockopt(zmq.SNDHWM, self.config.ingress_capacity)
        self._socket.setsockopt(zmq.RCVHWM, self.config.ingress_capacity)
        self._socket.setsockopt(zmq.LINGER, 1000)
        self._socket.connect(self.endpoint)
        self._reader_task = asyncio.create_task(self._reader_loop(), name="engine-reader")
        self._writer_task = asyncio.create_task(self._writer_loop(), name="engine-writer")

    async def wait_ready(
        self,
        timeout_s: float,
        is_worker_alive=None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        health_task = asyncio.create_task(
            self.request(
                Operation.HEALTH,
                timeout_s=timeout_s,
                generation="",
            )
        )
        while True:
            if is_worker_alive is not None and not is_worker_alive():
                health_task.cancel()
                raise UnavailableError("engine worker exited during startup")
            done, _ = await asyncio.wait({health_task}, timeout=0.1)
            if done:
                response = await health_task
                self.generation = str(response["generation"])
                return response
            if time.monotonic() >= deadline:
                health_task.cancel()
                raise UnavailableError("engine worker readiness timed out")

    async def request(
        self,
        operation: Operation,
        *,
        payload: bytes = b"",
        timeout_s: float | None = None,
        deadline_s: float | None = None,
        generation: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        if self._closed or self._socket is None:
            raise UnavailableError("engine client is not running")
        if len(self._pending) >= self.config.ingress_capacity:
            raise ResourceExhaustedError("engine pending request capacity reached")
        if self._inflight_bytes + len(payload) > self.config.max_inflight_bytes:
            raise ResourceExhaustedError("engine inflight byte capacity reached")

        request_id = uuid.uuid4().hex
        effective_timeout = timeout_s or self.config.queue_timeout_s
        effective_deadline = deadline_s or effective_timeout
        metadata = {
            "version": PROTOCOL_VERSION,
            "op": operation.value,
            "generation": self.generation if generation is None else generation,
            "request_id": request_id,
            "session_id": "",
            "sequence": 0,
            "deadline_ns": time.monotonic_ns() + int(effective_deadline * 1e9),
            "payload_length": len(payload),
            **fields,
        }
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[request_id] = future
        self._pending_bytes[request_id] = len(payload)
        self._inflight_bytes += len(payload)
        try:
            self._send_queue.put_nowait((metadata, payload))
        except asyncio.QueueFull as exc:
            self._release_pending(request_id)
            raise ResourceExhaustedError("engine ingress queue is full") from exc

        try:
            response = await asyncio.wait_for(asyncio.shield(future), effective_timeout)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            await self._send_cancel(request_id)
            self._release_pending(request_id)
            raise
        raise_for_response(response)
        return response

    async def health(self) -> dict[str, Any]:
        return await self.request(Operation.HEALTH, timeout_s=1.0)

    async def detect(
        self,
        audio: bytes,
        sample_rate: int,
        postprocessing: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        queue_timeout_s = self.config.detect_queue_timeout_s
        response = await self.request(
            Operation.DETECT,
            payload=audio,
            sample_rate=sample_rate,
            postprocessing=postprocessing,
            timeout_s=max(300.0, queue_timeout_s),
            deadline_s=queue_timeout_s,
        )
        return response.get("segments", [])

    async def open_stream(
        self,
        session_id: str,
        endpoint_only: bool,
        postprocessing: dict[str, Any] | None = None,
    ) -> None:
        await self.request(
            Operation.STREAM_OPEN,
            session_id=session_id,
            sequence=0,
            endpoint_only=endpoint_only,
            postprocessing=postprocessing,
        )

    async def stream_chunk(
        self,
        session_id: str,
        sequence: int,
        audio: bytes,
        sample_rate: int,
        is_final: bool,
    ) -> list[list[float]]:
        response = await self.request(
            Operation.STREAM_CHUNK,
            payload=audio,
            session_id=session_id,
            sequence=sequence,
            sample_rate=sample_rate,
            is_final=is_final,
        )
        return response.get("events", [])

    async def close_stream(self, session_id: str, sequence: int) -> None:
        await self.request(
            Operation.STREAM_CLOSE,
            session_id=session_id,
            sequence=sequence,
        )

    async def shutdown_worker(self) -> None:
        if self._closed:
            return
        with contextlib.suppress(Exception):
            await self.request(
                Operation.SHUTDOWN,
                timeout_s=self.config.shutdown_grace_s,
            )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(asyncio.QueueFull):
            self._send_queue.put_nowait(None)
        tasks = [task for task in (self._writer_task, self._reader_task) if task]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._fail_all(UnavailableError("engine client closed"))
        if self._socket is not None:
            self._socket.close(linger=1000)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

    async def _writer_loop(self) -> None:
        assert self._socket is not None
        try:
            while True:
                item = await self._send_queue.get()
                if item is None:
                    return
                metadata, payload = item
                await self._socket.send_multipart([pack_metadata(metadata), payload])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Engine writer failed")
            self._fail_all(UnavailableError(f"engine writer failed: {exc}"))

    async def _reader_loop(self) -> None:
        assert self._socket is not None
        try:
            while True:
                frames = await self._socket.recv_multipart()
                if len(frames) != 2:
                    logger.error("Dropping malformed engine response: frames=%s", len(frames))
                    continue
                metadata = unpack_metadata(frames[0])
                request_id = metadata.get("request_id")
                future = self._pending.get(request_id)
                if future is None:
                    logger.warning("Dropping late or unknown engine response: request_id=%s", request_id)
                    continue
                self._release_pending(request_id)
                if not future.done():
                    future.set_result(metadata)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Engine reader failed")
            self._fail_all(UnavailableError(f"engine reader failed: {exc}"))

    async def _send_cancel(self, target_request_id: str) -> None:
        if self._closed:
            return
        metadata = {
            "version": PROTOCOL_VERSION,
            "op": Operation.CANCEL.value,
            "generation": self.generation,
            "request_id": uuid.uuid4().hex,
            "session_id": "",
            "sequence": 0,
            "target_request_id": target_request_id,
            "deadline_ns": time.monotonic_ns() + int(1e9),
            "payload_length": 0,
        }
        with contextlib.suppress(asyncio.QueueFull):
            self._send_queue.put_nowait((metadata, b""))

    def _release_pending(self, request_id: str) -> asyncio.Future | None:
        future = self._pending.pop(request_id, None)
        self._inflight_bytes -= self._pending_bytes.pop(request_id, 0)
        return future

    def _fail_all(self, error: Exception) -> None:
        pending = list(self._pending.items())
        self._pending.clear()
        self._pending_bytes.clear()
        self._inflight_bytes = 0
        for _, future in pending:
            if not future.done():
                future.set_exception(error)
