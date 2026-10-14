"""IPC adapter and process supervision for the native FireRedVAD engine."""
from __future__ import annotations

import json
import logging
import os
import signal
import time
import uuid
from collections import deque
from dataclasses import dataclass

import numpy as np
import zmq

from fireredvad_ncnn.configs.engine_config import EngineConfig
from fireredvad_ncnn.configs.runtime_config import RuntimeConfig
from fireredvad_ncnn.configs.stream_vad_config import FireRedStreamVadConfig, resolve_postprocessing_config
from fireredvad_ncnn.engine.protocol import (
    EngineError, InvalidArgumentError, FailedPreconditionError, ResourceExhaustedError,
    DeadlineExceededError, Operation, Status, build_response, pack_metadata,
    unpack_metadata, validate_request,
)
from fireredvad_ncnn.native import NativeEngine, NativeError

logger = logging.getLogger(__name__)
_STATUS = {0: Status.OK, 1: Status.INVALID_ARGUMENT, 2: Status.RESOURCE_EXHAUSTED,
           3: Status.DEADLINE_EXCEEDED, 4: Status.FAILED_PRECONDITION,
           5: Status.UNAVAILABLE, 6: Status.INTERNAL, 7: Status.FAILED_PRECONDITION}


@dataclass
class _Pending:
    identity: bytes
    metadata: dict
    native_session: int
    canceled: bool = False


def _string(metadata, field):
    value = metadata.get(field)
    if not isinstance(value, str) or not value:
        raise InvalidArgumentError(f"{field} must be a non-empty string")
    return value


def _integer(metadata, field):
    value = metadata.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 0xFFFFFFFF:
        raise InvalidArgumentError(f"{field} must be a positive uint32")
    return value


def run_engine_worker(endpoint: str, parent_pid: int, model_path: str,
                      runtime_config: RuntimeConfig, engine_config: EngineConfig,
                      stream_vad_config: FireRedStreamVadConfig,
                      endpoint_only_default: bool, max_payload_bytes: int,
                      log_level: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    logging.basicConfig(level=log_level,
        format="%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", force=True)
    generation = uuid.uuid4().hex
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.SNDHWM, engine_config.ingress_capacity)
    socket.setsockopt(zmq.RCVHWM, engine_config.ingress_capacity)
    socket.setsockopt(zmq.LINGER, 1000)
    engine = None
    sessions: dict[str, int] = {}
    pending: dict[int, _Pending] = {}
    requests: dict[str, int] = {}
    responses = deque()
    response_bytes = 0

    def respond(identity, request_id, status=Status.OK, **fields):
        nonlocal response_bytes
        packed = pack_metadata(build_response(request_id, generation, status=status, **fields))
        if len(responses) >= engine_config.ingress_capacity or response_bytes + len(packed) > engine_config.max_inflight_bytes:
            raise RuntimeError("IPC response capacity exhausted; stopping worker")
        responses.append((identity, packed))
        response_bytes += len(packed)

    def close_session(name):
        native = sessions.pop(name, None)
        if native is not None:
            try:
                engine.session_close(native)
            except NativeError as exc:
                if exc.status != 4:
                    raise

    try:
        socket.bind(endpoint)
        engine = NativeEngine(model_path, cpu_budget=runtime_config.num_threads,
            worker_count=runtime_config.worker_count,
            threads_per_worker=runtime_config.effective_threads_per_worker,
            chunk_frames=runtime_config.stream_chunk_frame, engine_config=engine_config)
        logger.info("Engine worker ready: pid=%s generation=%s native=%s workers=%s threads=%s endpoint=%s",
            os.getpid(), generation, engine.build_info, runtime_config.worker_count,
            runtime_config.effective_threads_per_worker, endpoint)
        last_metrics = time.monotonic()
        while not stopping:
            if os.getppid() != parent_pid:
                logger.error("Engine parent disappeared; worker exiting")
                break
            # Bound each transport turn so native completions cannot starve.
            if socket.poll(1 if pending or responses else 50, zmq.POLLIN):
                for _ in range(64):
                    if not socket.poll(0, zmq.POLLIN):
                        break
                    frames = socket.recv_multipart()
                    if len(frames) != 3:
                        logger.warning("Dropping malformed engine request: frames=%s", len(frames))
                        continue
                    identity, packed, payload = frames
                    request_id = "unknown"
                    try:
                        metadata = unpack_metadata(packed)
                        request_id = str(metadata.get("request_id", "unknown"))
                        op = validate_request(metadata, payload, max_payload_bytes)
                        if op is not Operation.HEALTH and metadata["generation"] != generation:
                            raise FailedPreconditionError("engine generation mismatch")
                        remaining_ns = int(metadata["deadline_ns"]) - time.monotonic_ns()
                        if remaining_ns <= 0:
                            raise DeadlineExceededError("request deadline expired before ingress")
                        if request_id in requests:
                            raise FailedPreconditionError("duplicate pending request ID")
                        if op is Operation.HEALTH:
                            respond(identity, request_id, ready=True, pid=os.getpid(),
                                sessions=len(sessions), native=engine.build_info, metrics=engine.metrics())
                        elif op is Operation.STREAM_OPEN:
                            name = _string(metadata, "session_id")
                            if name in sessions:
                                raise FailedPreconditionError("stream session already exists")
                            post = resolve_postprocessing_config(stream_vad_config, metadata.get("postprocessing"))
                            sessions[name] = engine.session_open(post, bool(metadata.get("endpoint_only", endpoint_only_default)))
                            respond(identity, request_id)
                        elif op is Operation.STREAM_CLOSE:
                            name = _string(metadata, "session_id")
                            if name not in sessions:
                                raise FailedPreconditionError("unknown stream session")
                            if any(p.native_session == sessions[name] for p in pending.values()):
                                raise FailedPreconditionError("stream session still has pending work")
                            close_session(name)
                            respond(identity, request_id)
                        elif op in (Operation.STREAM_CHUNK, Operation.DETECT):
                            if len(pending) + len(responses) >= engine_config.ingress_capacity:
                                raise ResourceExhaustedError("IPC pending capacity reached")
                            rate = _integer(metadata, "sample_rate")
                            if len(payload) % 2 or (op is Operation.DETECT and not payload):
                                raise InvalidArgumentError("LINEAR16 payload must have even bytes; detect must be non-empty")
                            native_session, sequence, post = 0, 0, None
                            if op is Operation.STREAM_CHUNK:
                                name = _string(metadata, "session_id")
                                if name not in sessions:
                                    raise FailedPreconditionError("unknown stream session")
                                native_session = sessions[name]
                                sequence = _integer(metadata, "sequence")
                            else:
                                post = resolve_postprocessing_config(stream_vad_config, metadata.get("postprocessing"))
                            native_id = engine.submit(np.frombuffer(payload, np.int16), rate,
                                session=native_session, sequence=sequence,
                                is_final=op is Operation.DETECT or bool(metadata.get("is_final", False)),
                                postprocessing_config=post,
                                timeout_ms=min(0xFFFFFFFF, max(1, (remaining_ns + 999999) // 1000000)))
                            pending[native_id] = _Pending(identity, metadata, native_session)
                            requests[request_id] = native_id
                        elif op is Operation.CANCEL:
                            target = _string(metadata, "target_request_id")
                            native_id = requests.get(target)
                            if native_id is not None:
                                pending[native_id].canceled = True
                                engine.cancel(native_id)
                                if pending[native_id].native_session:
                                    close_session(pending[native_id].metadata["session_id"])
                        elif op is Operation.SHUTDOWN:
                            respond(identity, request_id)
                            stopping = True
                    except (EngineError, NativeError, ValueError) as exc:
                        status = _STATUS.get(exc.status, Status.INTERNAL) if isinstance(exc, NativeError) else (
                            exc.status if isinstance(exc, EngineError) else Status.INVALID_ARGUMENT)
                        respond(identity, request_id, status, message=str(exc))
                    except Exception:
                        logger.exception("Unhandled engine request failure")
                        respond(identity, request_id, Status.INTERNAL, message="internal engine error")
                    if stopping:
                        break
            for _ in range(64):
                completion = engine.poll()
                if completion is None:
                    break
                p = pending.pop(completion.request_id)
                request_id = p.metadata["request_id"]
                requests.pop(request_id, None)
                expired = time.monotonic_ns() >= p.metadata["deadline_ns"]
                if completion.status or p.canceled or expired:
                    if p.native_session:
                        close_session(p.metadata["session_id"])
                    if not p.canceled:
                        status = Status.DEADLINE_EXCEEDED if expired else _STATUS.get(completion.status, Status.INTERNAL)
                        respond(p.identity, request_id, status, message=completion.message or "request deadline exceeded")
                elif p.native_session:
                    respond(p.identity, request_id, events=completion.events)
                else:
                    respond(p.identity, request_id, segments=[e[:2] for e in completion.events])
            while responses:
                identity, packed = responses[0]
                try:
                    socket.send_multipart([identity, packed, b""], flags=zmq.DONTWAIT)
                except zmq.Again:
                    break
                responses.popleft()
                response_bytes -= len(packed)
            if time.monotonic() - last_metrics >= engine_config.metrics_interval_s:
                last_metrics = time.monotonic()
                logger.info("engine_metrics=%s", json.dumps({**engine.metrics(), "ipc_pending": len(pending),
                    "response_pending": len(responses), "response_bytes": response_bytes}, sort_keys=True))
        # Flush shutdown acknowledgement within the configured grace period.
        until = time.monotonic() + engine_config.shutdown_grace_s
        while responses and time.monotonic() < until:
            if socket.poll(10, zmq.POLLOUT):
                identity, packed = responses[0]
                try:
                    socket.send_multipart([identity, packed, b""], flags=zmq.DONTWAIT)
                    responses.popleft()
                except zmq.Again:
                    pass
    finally:
        if engine is not None:
            logger.info("engine_metrics_final=%s", json.dumps(engine.metrics(), sort_keys=True))
            engine.close()
        socket.close(linger=1000)
        context.term()
