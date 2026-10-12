from __future__ import annotations

import json
import logging
import os
import signal
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import zmq

from fireredvad_ncnn.configs.engine_config import EngineConfig
from fireredvad_ncnn.configs.runtime_config import RuntimeConfig
from fireredvad_ncnn.configs.stream_vad_config import (
    FireRedStreamVadConfig,
    resolve_postprocessing_config,
)
from fireredvad_ncnn.engine.protocol import (
    EngineError,
    FailedPreconditionError,
    InvalidArgumentError,
    Operation,
    PROTOCOL_VERSION,
    ResourceExhaustedError,
    Status,
    build_response,
    pack_metadata,
    unpack_metadata,
    validate_request,
)
from fireredvad_ncnn.inferencers.stream_events import StreamEventPublisher

logger = logging.getLogger(__name__)


@dataclass
class _Request:
    identity: bytes
    metadata: dict[str, Any]
    payload: bytes
    enqueued_at: float = field(default_factory=time.monotonic)


@dataclass
class _Session:
    state: Any
    endpoint_only: bool
    model_cache: object = None
    cache_profile: object = None
    expected_sequence: int = 1
    queue: deque[_Request] = field(default_factory=deque)
    busy: bool = False
    publisher: StreamEventPublisher = field(init=False)

    def __post_init__(self) -> None:
        self.publisher = StreamEventPublisher(endpoint_only=self.endpoint_only)


@dataclass
class _Work:
    request: _Request
    kind: str
    features: deque
    input_duration: float = 0.0
    is_final: bool = False
    session_id: str | None = None
    events: list[list[float]] = field(default_factory=list)
    postprocessing_config: FireRedStreamVadConfig | None = None

    def profile(self, session: _Session | None) -> tuple:
        features = self.features[0]
        cache_profile = None if session is None else session.cache_profile
        if (
            cache_profile is None
            and session is not None
            and session.state.model_caches is not None
        ):
            cache_profile = tuple(
                tuple(cache.shape[1:]) for cache in session.state.model_caches
            )
        return self.kind, tuple(features.shape), cache_profile


class _Metrics:
    def __init__(self) -> None:
        self.counters = Counter()
        self.batch_histogram = Counter()
        self.queue_high_water = 0
        self.last_log = time.monotonic()

    def log_if_due(
        self,
        config: EngineConfig,
        ready: int,
        sessions: int,
        cache_metrics: Callable[[], dict[str, object] | None] | None = None,
    ) -> None:
        now = time.monotonic()
        if now - self.last_log < config.metrics_interval_s:
            return
        self.last_log = now
        cache_arena = None if cache_metrics is None else cache_metrics()
        logger.info(
            "engine_metrics=%s",
            json.dumps(
                {
                    "counters": dict(self.counters),
                    "batch_histogram": dict(self.batch_histogram),
                    "ready": ready,
                    "queue_high_water": self.queue_high_water,
                    "sessions": sessions,
                    "cache_arena": cache_arena,
                },
                sort_keys=True,
            ),
        )


def run_engine_worker(
    endpoint: str,
    parent_pid: int,
    model_path: str,
    runtime_config: RuntimeConfig,
    engine_config: EngineConfig,
    stream_vad_config: FireRedStreamVadConfig,
    endpoint_only_default: bool,
    max_payload_bytes: int,
    log_level: int,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    from fireredvad_ncnn.inferencers.inferencer import Inferencer

    generation = uuid.uuid4().hex
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.SNDHWM, engine_config.ingress_capacity)
    socket.setsockopt(zmq.RCVHWM, engine_config.ingress_capacity)
    socket.setsockopt(zmq.LINGER, 1000)
    socket.bind(endpoint)
    inferencer = None
    try:
        inferencer = Inferencer(
            model_dir=model_path,
            ncpu=runtime_config.num_threads,
            stream_chunk_frame=runtime_config.stream_chunk_frame,
            stream_vad_config=stream_vad_config,
        )
        logger.info(
            "Engine worker ready: pid=%s generation=%s device=%s dtype=%s runner=%s endpoint=%s",
            os.getpid(),
            generation,
            runtime_config.device,
            runtime_config.dtype,
            runtime_config.runner_name,
            endpoint,
        )
        _run_loop(
            socket,
            inferencer,
            generation,
            parent_pid,
            engine_config,
            endpoint_only_default,
            max_payload_bytes,
        )
    finally:
        if inferencer is not None:
            inferencer.close()
        socket.close(linger=1000)
        context.term()


def _run_loop(
    socket,
    inferencer,
    generation: str,
    parent_pid: int,
    config: EngineConfig,
    endpoint_only_default: bool,
    max_payload_bytes: int,
) -> None:
    sessions: dict[str, _Session] = {}
    ready: list[_Work] = []
    canceled: set[str] = set()
    metrics = _Metrics()
    running = True

    try:
        while running:
            if os.getppid() != parent_pid:
                logger.error("Engine parent disappeared; worker exiting")
                break

            timeout_ms = _poll_timeout_ms(ready, sessions, config)
            if socket.poll(timeout_ms, zmq.POLLIN):
                running = _receive_and_dispatch(
                    socket,
                    inferencer,
                    generation,
                    config,
                    endpoint_only_default,
                    max_payload_bytes,
                    sessions,
                    ready,
                    canceled,
                    metrics,
                )
                while running and socket.poll(0, zmq.POLLIN):
                    running = _receive_and_dispatch(
                        socket,
                        inferencer,
                        generation,
                        config,
                        endpoint_only_default,
                        max_payload_bytes,
                        sessions,
                        ready,
                        canceled,
                        metrics,
                    )

            _drop_expired_or_canceled(
                socket,
                inferencer,
                generation,
                sessions,
                ready,
                canceled,
                metrics,
            )
            batch = _select_batch(ready, sessions, config)
            if batch:
                running = _execute_batch(
                    socket,
                    inferencer,
                    generation,
                    sessions,
                    ready,
                    batch,
                    canceled,
                    metrics,
                    config,
                    endpoint_only_default,
                    max_payload_bytes,
                )
            metrics.log_if_due(
                config,
                len(ready),
                len(sessions),
                inferencer.runtime.cache_metrics,
            )
    finally:
        for session_id in list(sessions):
            _remove_session(sessions, session_id, inferencer)


def _receive_and_dispatch(
    socket,
    inferencer,
    generation: str,
    config: EngineConfig,
    endpoint_only_default: bool,
    max_payload_bytes: int,
    sessions: dict[str, _Session],
    ready: list[_Work],
    canceled: set[str],
    metrics: _Metrics,
    inflight_request_ids: set[str] | None = None,
) -> bool:
    frames = socket.recv_multipart()
    if len(frames) != 3:
        logger.error("Dropping malformed engine request: frames=%s", len(frames))
        return True
    identity, packed_metadata, payload = frames
    request_id = "unknown"
    try:
        metadata = unpack_metadata(packed_metadata)
        request_id = str(metadata.get("request_id", "unknown"))
        operation = validate_request(metadata, payload, max_payload_bytes)
        request = _Request(identity, metadata, payload)
        if operation is not Operation.HEALTH and metadata["generation"] != generation:
            raise FailedPreconditionError("engine generation mismatch")
        if time.monotonic_ns() >= int(metadata["deadline_ns"]):
            raise EngineError("request deadline expired before ingress")

        if operation is Operation.HEALTH:
            _send_response(
                socket,
                identity,
                build_response(
                    request_id,
                    generation,
                    ready=True,
                    pid=os.getpid(),
                    sessions=len(sessions),
                ),
            )
        elif operation is Operation.STREAM_OPEN:
            session_id = _required_string(metadata, "session_id")
            if session_id in sessions:
                raise FailedPreconditionError("stream session already exists")
            if len(sessions) >= config.max_sessions:
                raise ResourceExhaustedError("maximum active sessions reached")
            postprocessing_config = _resolve_postprocessing_config(
                inferencer.config,
                metadata.get("postprocessing"),
            )
            try:
                model_cache = inferencer.runtime.allocate_session_cache()
            except RuntimeError as exc:
                raise ResourceExhaustedError(str(exc)) from exc
            try:
                sessions[session_id] = _Session(
                    state=inferencer.new_stream_state(postprocessing_config),
                    endpoint_only=bool(
                        metadata.get("endpoint_only", endpoint_only_default)
                    ),
                    model_cache=model_cache,
                    cache_profile=inferencer.runtime.session_cache_profile(model_cache),
                )
            except Exception:
                inferencer.runtime.release_session_cache(model_cache)
                raise
            _send_response(socket, identity, build_response(request_id, generation))
            metrics.counters["stream_open"] += 1
        elif operation is Operation.STREAM_CHUNK:
            _enqueue_stream_request(
                socket,
                inferencer,
                generation,
                config,
                request,
                sessions,
                ready,
                metrics,
            )
        elif operation is Operation.STREAM_CLOSE:
            session_id = _required_string(metadata, "session_id")
            session = sessions.get(session_id)
            if session is None:
                raise FailedPreconditionError("unknown stream session")
            if session.busy or session.queue:
                raise FailedPreconditionError("stream session still has pending work")
            _remove_session(sessions, session_id, inferencer)
            _send_response(socket, identity, build_response(request_id, generation))
            metrics.counters["stream_close"] += 1
        elif operation is Operation.DETECT:
            if len(ready) >= config.ingress_capacity:
                raise ResourceExhaustedError("engine ready capacity reached")
            sample_rate = _positive_int(metadata, "sample_rate")
            postprocessing_config = _resolve_postprocessing_config(
                inferencer.config,
                metadata.get("postprocessing"),
            )
            if not payload or len(payload) % 2:
                raise InvalidArgumentError("LINEAR16 detect payload must be non-empty and even")
            features = inferencer.prepare_detect(np.frombuffer(payload, np.int16), sample_rate)
            metadata["deadline_ns"] = time.monotonic_ns() + int(
                config.detect_queue_timeout_s * 1e9
            )
            if features is None:
                _send_response(
                    socket,
                    identity,
                    build_response(request_id, generation, segments=[]),
                )
            else:
                ready.append(
                    _Work(
                        request,
                        "detect",
                        deque([features]),
                        postprocessing_config=postprocessing_config,
                    )
                )
                metrics.queue_high_water = max(metrics.queue_high_water, len(ready))
            metrics.counters["detect"] += 1
        elif operation is Operation.CANCEL:
            target = _required_string(metadata, "target_request_id")
            known = target in (inflight_request_ids or set()) or any(
                work.request.metadata["request_id"] == target for work in ready
            ) or any(
                queued.metadata["request_id"] == target
                for session in sessions.values()
                for queued in session.queue
            )
            if known:
                canceled.add(target)
                metrics.counters["cancel"] += 1
            else:
                metrics.counters["late_or_unknown_cancel"] += 1
        elif operation is Operation.SHUTDOWN:
            _send_response(socket, identity, build_response(request_id, generation))
            return False
    except EngineError as exc:
        status = exc.status
        if str(exc) == "request deadline expired before ingress":
            status = Status.DEADLINE_EXCEEDED
        _send_response(
            socket,
            identity,
            build_response(request_id, generation, status=status, message=str(exc)),
        )
        metrics.counters[f"error_{status.value}"] += 1
    except Exception as exc:
        logger.exception("Unhandled engine request failure")
        _send_response(
            socket,
            identity,
            build_response(
                request_id,
                generation,
                status=Status.INTERNAL,
                message="internal engine error",
            ),
        )
        metrics.counters["error_internal"] += 1
    return True


def _enqueue_stream_request(
    socket,
    inferencer,
    generation: str,
    config: EngineConfig,
    request: _Request,
    sessions: dict[str, _Session],
    ready: list[_Work],
    metrics: _Metrics,
) -> None:
    metadata = request.metadata
    session_id = _required_string(metadata, "session_id")
    session = sessions.get(session_id)
    if session is None:
        raise FailedPreconditionError("unknown stream session")
    sequence = _positive_int(metadata, "sequence", allow_zero=False)
    if sequence != session.expected_sequence:
        raise FailedPreconditionError(
            f"stream sequence mismatch: expected={session.expected_sequence} got={sequence}"
        )
    if len(session.queue) >= config.session_queue_capacity:
        raise ResourceExhaustedError("stream session queue capacity reached")
    if len(request.payload) % 2:
        raise InvalidArgumentError("LINEAR16 stream payload must contain even bytes")
    _positive_int(metadata, "sample_rate")
    session.expected_sequence += 1
    session.queue.append(request)
    if not session.busy:
        _prepare_next_stream_work(
            socket,
            inferencer,
            generation,
            session_id,
            session,
            ready,
            metrics,
        )
    metrics.counters["stream_chunk"] += 1


def _prepare_next_stream_work(
    socket,
    inferencer,
    generation: str,
    session_id: str,
    session: _Session,
    ready: list[_Work],
    metrics: _Metrics,
) -> None:
    if session.busy or not session.queue:
        return
    request = session.queue.popleft()
    metadata = request.metadata
    prepared = inferencer.prepare_stream_request(
        session.state,
        np.frombuffer(request.payload, np.int16),
        int(metadata["sample_rate"]),
        bool(metadata.get("is_final", False)),
    )
    if not prepared.features:
        raw_events = inferencer.finish_stream_request(session.state, prepared)
        if prepared.is_final:
            session.model_cache = inferencer.runtime.reset_session_cache(
                session.model_cache
            )
            session.cache_profile = inferencer.runtime.session_cache_profile(
                session.model_cache
            )
        events = _filter_stream_events(session, raw_events)
        _send_response(
            socket,
            request.identity,
            build_response(
                request.metadata["request_id"],
                generation,
                events=events,
            ),
        )
        _prepare_next_stream_work(
            socket,
            inferencer,
            generation,
            session_id,
            session,
            ready,
            metrics,
        )
        return
    session.busy = True
    ready.append(
        _Work(
            request=request,
            kind="stream",
            features=deque(prepared.features),
            input_duration=prepared.input_duration,
            is_final=prepared.is_final,
            session_id=session_id,
        )
    )
    metrics.queue_high_water = max(metrics.queue_high_water, len(ready))


def _select_batch(
    ready: list[_Work],
    sessions: dict[str, _Session],
    config: EngineConfig,
) -> list[_Work]:
    if not ready:
        return []
    ready.sort(key=lambda work: work.request.enqueued_at)
    oldest = ready[0]
    session = sessions.get(oldest.session_id) if oldest.session_id else None
    profile = oldest.profile(session)
    compatible = [
        work
        for work in ready
        if work.profile(sessions.get(work.session_id) if work.session_id else None)
        == profile
    ]
    target = min(config.max_batch_size, max(1, len(sessions)))
    age = time.monotonic() - oldest.request.enqueued_at
    if len(sessions) > 1 and len(compatible) < target and age < config.max_batch_wait_s:
        return []
    return compatible[: config.max_batch_size]


def _execute_batch(
    socket,
    inferencer,
    generation: str,
    sessions: dict[str, _Session],
    ready: list[_Work],
    batch: list[_Work],
    canceled: set[str],
    metrics: _Metrics,
    config: EngineConfig,
    endpoint_only_default: bool,
    max_payload_bytes: int,
) -> bool:
    for work in batch:
        ready.remove(work)
    features = [work.features[0] for work in batch]
    caches = [
        sessions[work.session_id].model_cache if work.session_id else None
        for work in batch
    ]
    started = time.monotonic()
    try:
        if batch[0].kind == "stream":
            probs_batch, caches_batch = inferencer.runtime.forward_session_batch(
                features,
                caches,
            )
        else:
            probs_batch, caches_batch = inferencer.runtime.forward_batch(features, caches)
    except Exception:
        logger.exception("Runtime batch failed: batch=%s", len(batch))
        for work in batch:
            _send_response(
                socket,
                work.request.identity,
                build_response(
                    work.request.metadata["request_id"],
                    generation,
                    status=Status.INTERNAL,
                    message="runtime batch failed",
                ),
            )
            if work.session_id:
                _remove_session(sessions, work.session_id, inferencer)
        metrics.counters["batch_failure"] += 1
        return True

    running = True
    inflight_request_ids = {
        work.request.metadata["request_id"] for work in batch
    }
    while running and socket.poll(0, zmq.POLLIN):
        running = _receive_and_dispatch(
            socket,
            inferencer,
            generation,
            config,
            endpoint_only_default,
            max_payload_bytes,
            sessions,
            ready,
            canceled,
            metrics,
            inflight_request_ids,
        )
    if not running:
        for work, candidate in zip(batch, caches_batch):
            if work.kind == "stream" and work.session_id in sessions:
                inferencer.runtime.discard_session_cache(candidate)
        return False

    metrics.batch_histogram[str(len(batch))] += 1
    metrics.counters["device_batches"] += 1
    metrics.counters["device_items"] += len(batch)
    metrics.counters["device_ms"] += int((time.monotonic() - started) * 1000)

    for work, probs, new_caches in zip(batch, probs_batch, caches_batch):
        request_id = work.request.metadata["request_id"]
        expired = time.monotonic_ns() >= int(work.request.metadata["deadline_ns"])
        was_canceled = request_id in canceled
        session = sessions.get(work.session_id) if work.session_id else None
        if was_canceled or expired or (work.session_id and session is None):
            if work.kind == "stream" and session is not None:
                inferencer.runtime.discard_session_cache(new_caches)
                _remove_session(sessions, work.session_id, inferencer)
            if was_canceled:
                canceled.discard(request_id)
                metrics.counters["inflight_discard"] += 1
            elif expired:
                _send_response(
                    socket,
                    work.request.identity,
                    build_response(
                        request_id,
                        generation,
                        status=Status.DEADLINE_EXCEEDED,
                        message="engine execution deadline exceeded",
                    ),
                )
                metrics.counters["execution_deadline"] += 1
            continue

        work.features.popleft()
        if work.kind == "detect":
            segments = inferencer.postprocess_detect(
                probs,
                work.postprocessing_config,
            )
            _send_response(
                socket,
                work.request.identity,
                build_response(
                    work.request.metadata["request_id"],
                    generation,
                    segments=[[start, end] for start, end in segments],
                ),
            )
            continue

        assert session is not None
        session.model_cache = inferencer.runtime.commit_session_cache(
            session.model_cache,
            new_caches,
        )
        session.cache_profile = inferencer.runtime.session_cache_profile(
            session.model_cache
        )
        raw_events = inferencer.apply_stream_probabilities(
            session.state,
            probs,
            new_caches,
            commit_model_cache=False,
        )
        work.events.extend(_filter_stream_events(session, raw_events))
        if work.features:
            work.request.enqueued_at = time.monotonic()
            ready.append(work)
            continue

        prepared = type(
            "Prepared",
            (),
            {"is_final": work.is_final, "input_duration": work.input_duration},
        )()
        final_events = inferencer.finish_stream_request(
            session.state,
            prepared,
        )
        if work.is_final:
            session.model_cache = inferencer.runtime.reset_session_cache(
                session.model_cache
            )
            session.cache_profile = inferencer.runtime.session_cache_profile(
                session.model_cache
            )
        work.events.extend(_filter_stream_events(session, final_events))
        _send_response(
            socket,
            work.request.identity,
            build_response(
                work.request.metadata["request_id"],
                generation,
                events=work.events,
            ),
        )
        session.busy = False
        _prepare_next_stream_work(
            socket,
            inferencer,
            generation,
            work.session_id,
            session,
            ready,
            metrics,
        )
    return True


def _filter_stream_events(
    session: _Session,
    raw_events: list[tuple[float, float, float]],
) -> list[list[float]]:
    return [list(event) for event in session.publisher.publish(raw_events)]


def _drop_expired_or_canceled(
    socket,
    inferencer,
    generation: str,
    sessions: dict[str, _Session],
    ready: list[_Work],
    canceled: set[str],
    metrics: _Metrics,
) -> None:
    now_ns = time.monotonic_ns()
    for work in list(ready):
        request_id = work.request.metadata["request_id"]
        expired = now_ns >= int(work.request.metadata["deadline_ns"])
        if request_id not in canceled and not expired:
            continue
        ready.remove(work)
        if request_id not in canceled:
            _send_response(
                socket,
                work.request.identity,
                build_response(
                    request_id,
                    generation,
                    status=Status.DEADLINE_EXCEEDED,
                    message="engine queue deadline exceeded",
                ),
            )
            metrics.counters["queue_deadline"] += 1
        else:
            canceled.discard(request_id)
            metrics.counters["hard_cancel"] += 1
        if work.session_id:
            _remove_session(sessions, work.session_id, inferencer)


def _remove_session(
    sessions: dict[str, _Session],
    session_id: str,
    inferencer,
) -> _Session | None:
    session = sessions.pop(session_id, None)
    if session is not None:
        inferencer.runtime.release_session_cache(session.model_cache)
        session.model_cache = None
        session.cache_profile = None
    return session


def _poll_timeout_ms(
    ready: list[_Work],
    sessions: dict[str, _Session],
    config: EngineConfig,
) -> int:
    if not ready:
        return 100
    if len(sessions) <= 1:
        return 0
    oldest = min(work.request.enqueued_at for work in ready)
    remaining = max(0.0, config.max_batch_wait_s - (time.monotonic() - oldest))
    return max(0, min(100, int(remaining * 1000)))


def _send_response(socket, identity: bytes, metadata: dict[str, Any]) -> None:
    socket.send_multipart([identity, pack_metadata(metadata), b""])


def _required_string(metadata: dict[str, Any], field: str) -> str:
    value = metadata.get(field)
    if not isinstance(value, str) or not value:
        raise InvalidArgumentError(f"{field} must be a non-empty string")
    return value


def _positive_int(
    metadata: dict[str, Any],
    field: str,
    *,
    allow_zero: bool = False,
) -> int:
    value = metadata.get(field)
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or value < minimum:
        comparator = ">=" if allow_zero else ">"
        raise InvalidArgumentError(f"{field} must be an integer {comparator} 0")
    return value


def _resolve_postprocessing_config(
    default: FireRedStreamVadConfig,
    options: Any,
) -> FireRedStreamVadConfig:
    try:
        return resolve_postprocessing_config(default, options)
    except ValueError as exc:
        raise InvalidArgumentError(str(exc)) from exc
