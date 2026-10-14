from __future__ import annotations

import logging
import time
from collections import deque
from types import SimpleNamespace

import numpy as np

from fireredvad_ncnn.configs.engine_config import EngineConfig
from tests.reference.worker import (
    _Metrics,
    _Request,
    _Session,
    _Work,
    _execute_batch,
    _select_batch,
)


def _work(session_id: str, age_s: float = 0.0) -> _Work:
    request = _Request(
        identity=b"client",
        metadata={"request_id": session_id},
        payload=b"",
        enqueued_at=time.monotonic() - age_s,
    )
    return _Work(
        request=request,
        kind="stream",
        features=deque([np.zeros((10, 80))]),
        session_id=session_id,
    )


def _session() -> _Session:
    return _Session(state=SimpleNamespace(model_caches=None), endpoint_only=False)


def test_scheduler_waits_for_compatible_peer() -> None:
    config = EngineConfig(max_batch_size=32, max_batch_wait_ms=5)
    sessions = {"one": _session(), "two": _session()}

    assert _select_batch([_work("one")], sessions, config) == []


def test_scheduler_dispatches_full_compatible_batch() -> None:
    config = EngineConfig(max_batch_size=32, max_batch_wait_ms=5)
    sessions = {"one": _session(), "two": _session()}
    ready = [_work("one"), _work("two")]

    assert _select_batch(ready, sessions, config) == ready


def test_scheduler_dispatches_oldest_after_max_wait() -> None:
    config = EngineConfig(max_batch_size=32, max_batch_wait_ms=5)
    sessions = {"one": _session(), "two": _session()}
    work = _work("one", age_s=0.010)

    assert _select_batch([work], sessions, config) == [work]


def test_scheduler_does_not_batch_different_cache_shapes() -> None:
    config = EngineConfig(max_batch_size=32, max_batch_wait_ms=5)
    sessions = {"one": _session(), "two": _session()}
    sessions["one"].state.model_caches = [np.zeros((1, 64, 10))]
    sessions["two"].state.model_caches = [np.zeros((1, 64, 19))]
    oldest = _work("one", age_s=0.010)
    newer = _work("two", age_s=0.009)

    assert _select_batch([oldest, newer], sessions, config) == [oldest]


def test_scheduler_does_not_batch_different_arena_cache_phases() -> None:
    config = EngineConfig(max_batch_size=32, max_batch_wait_ms=5)
    sessions = {"one": _session(), "two": _session()}
    sessions["one"].cache_profile = ("arena", 10)
    sessions["two"].cache_profile = ("arena", 19)
    oldest = _work("one", age_s=0.010)
    newer = _work("two", age_s=0.009)

    assert _select_batch([oldest, newer], sessions, config) == [oldest]


def test_inflight_cancel_discards_candidate_and_releases_session() -> None:
    class FakeSocket:
        def __init__(self) -> None:
            self.responses = []

        def poll(self, *_args) -> bool:
            return False

        def send_multipart(self, frames) -> None:
            self.responses.append(frames)

    class FakeRuntime:
        def __init__(self) -> None:
            self.committed = []
            self.discarded = []
            self.released = []

        def forward_session_batch(self, features, caches):
            assert len(features) == len(caches) == 1
            return [np.zeros((10, 1))], ["candidate"]

        def discard_session_cache(self, candidate) -> None:
            self.discarded.append(candidate)

        def release_session_cache(self, cache) -> None:
            self.released.append(cache)

    runtime = FakeRuntime()
    inferencer = SimpleNamespace(runtime=runtime)
    socket = FakeSocket()
    request = _Request(
        identity=b"client",
        metadata={
            "request_id": "request-one",
            "deadline_ns": time.monotonic_ns() + 1_000_000_000,
        },
        payload=b"",
    )
    work = _Work(
        request=request,
        kind="stream",
        features=deque([np.zeros((10, 80))]),
        session_id="one",
    )
    session = _session()
    session.model_cache = "handle"
    session.cache_profile = ("arena", 19)
    session.busy = True
    sessions = {"one": session}
    ready = [work]
    canceled = {"request-one"}

    running = _execute_batch(
        socket,
        inferencer,
        "generation",
        sessions,
        ready,
        [work],
        canceled,
        _Metrics(),
        EngineConfig(),
        False,
        1024,
    )

    assert running
    assert ready == []
    assert sessions == {}
    assert canceled == set()
    assert runtime.discarded == ["candidate"]
    assert runtime.released == ["handle"]
    assert socket.responses == []


def test_periodic_metrics_are_logged_at_info(caplog) -> None:
    metrics = _Metrics()
    metrics.last_log = 0.0

    with caplog.at_level(logging.DEBUG, logger="tests.reference.worker"):
        metrics.log_if_due(EngineConfig(), ready=0, sessions=0)

    record = next(
        record for record in caplog.records if "engine_metrics=" in record.message
    )
    assert record.levelno == logging.INFO
