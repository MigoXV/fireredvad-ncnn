from __future__ import annotations

import asyncio
import logging

import fireredvad_ncnn.proto.ux_vad_pb2 as pb
from fireredvad_ncnn.configs.engine_config import EngineConfig
from fireredvad_ncnn.servicer.servicer import UxVoiceActivityDetectorService


class _Engine:
    config = EngineConfig()

    async def detect(
        self,
        audio: bytes,
        sample_rate: int,
        postprocessing=None,
    ) -> list[list[float]]:
        return []


async def _empty_requests():
    if False:
        yield


def test_inference_rpcs_log_one_simple_info_record(caplog) -> None:
    service = UxVoiceActivityDetectorService(_Engine())
    request = pb.DetectRequest(
        config=pb.DetectionConfig(
            encoding=pb.DetectionConfig.LINEAR16,
            sample_rate_hertz=16000,
        ),
        audio=b"\x00\x00",
    )

    async def invoke() -> None:
        await service.Detect(request, object())
        responses = service.StreamingDetect(_empty_requests(), object())
        assert await anext(responses, None) is None

    with caplog.at_level(logging.INFO, logger="fireredvad_ncnn.servicer.servicer"):
        asyncio.run(invoke())

    assert [
        record.message
        for record in caplog.records
        if record.message.startswith("rpc_request")
    ] == [
        "rpc_request method=Detect",
        "rpc_request method=StreamingDetect",
    ]
