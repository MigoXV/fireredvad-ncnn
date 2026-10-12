from __future__ import annotations

import asyncio

from fireredvad_ncnn.configs.engine_config import EngineConfig
from fireredvad_ncnn.engine.client import EngineClient
from fireredvad_ncnn.engine.protocol import Operation


def test_detect_uses_dedicated_queue_timeout() -> None:
    client = EngineClient(
        "ipc://unused",
        EngineConfig(queue_timeout_ms=5000, detect_queue_timeout_ms=30000),
    )
    captured = {}

    async def request(operation, **fields):
        captured.update(operation=operation, **fields)
        return {"segments": []}

    client.request = request

    postprocessing = {
        "processor": "firered.stream_vad",
        "version": 1,
        "parameters": {"speech_threshold": 0.7},
    }
    assert asyncio.run(client.detect(b"\x00\x00", 16000, postprocessing)) == []
    assert captured["operation"] is Operation.DETECT
    assert captured["deadline_s"] == 30.0
    assert captured["timeout_s"] == 300.0
    assert captured["postprocessing"] == postprocessing


def test_open_stream_binds_postprocessing_to_session() -> None:
    client = EngineClient("ipc://unused", EngineConfig())
    captured = {}

    async def request(operation, **fields):
        captured.update(operation=operation, **fields)
        return {}

    client.request = request
    postprocessing = {
        "processor": "firered.stream_vad",
        "version": 1,
        "parameters": {"min_silence_frame": 12},
    }

    asyncio.run(client.open_stream("session", True, postprocessing))

    assert captured["operation"] is Operation.STREAM_OPEN
    assert captured["session_id"] == "session"
    assert captured["postprocessing"] == postprocessing
