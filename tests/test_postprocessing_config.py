from __future__ import annotations

import pytest
import numpy as np

import fireredvad_ncnn.proto.ux_vad_pb2 as pb
from fireredvad_ncnn.configs.stream_vad_config import (
    FireRedStreamVadConfig,
    resolve_postprocessing_config,
)
from fireredvad_ncnn.configs.engine_config import EngineConfig
from tests.reference.inferencer import Inferencer
from fireredvad_ncnn.servicer.servicer import UxVoiceActivityDetectorService


class _Engine:
    config = EngineConfig()


def test_servicer_converts_postprocessing_struct_to_plain_values() -> None:
    service = UxVoiceActivityDetectorService(_Engine())
    config = pb.DetectionConfig(
        postprocessing=pb.PostprocessingOptions(
            processor="firered.stream_vad",
            version=1,
            parameters={
                "speech_threshold": 0.7,
                "min_silence_frame": 12,
            },
        )
    )

    assert service._parse_postprocessing(config) == {
        "processor": "firered.stream_vad",
        "version": 1,
        "parameters": {
            "speech_threshold": 0.7,
            "min_silence_frame": 12.0,
        },
    }


def test_resolve_postprocessing_config_overrides_request_values() -> None:
    default = FireRedStreamVadConfig()

    resolved = resolve_postprocessing_config(
        default,
        {
            "processor": "firered.stream_vad",
            "version": 1,
            "parameters": {
                "speech_threshold": 0.7,
                "min_silence_frame": 12.0,
            },
        },
    )

    assert resolved is not default
    assert resolved.speech_threshold == 0.7
    assert resolved.min_silence_frame == 12
    assert resolved.max_speech_frame == default.max_speech_frame


def test_offline_postprocessing_uses_request_config() -> None:
    inferencer = Inferencer.__new__(Inferencer)
    inferencer.config = FireRedStreamVadConfig()
    probabilities = np.array([[[0.5, 0.5]]])
    permissive = FireRedStreamVadConfig(
        smooth_window_size=1,
        speech_threshold=0.2,
        pad_start_frame=1,
        min_speech_frame=1,
    )
    strict = FireRedStreamVadConfig(
        smooth_window_size=1,
        speech_threshold=0.8,
        pad_start_frame=1,
        min_speech_frame=1,
    )

    assert inferencer.postprocess_detect(probabilities, permissive) == [(0.0, 0.01)]
    assert inferencer.postprocess_detect(probabilities, strict) == []


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {
                "processor": "unknown",
                "version": 1,
                "parameters": {},
            },
            "postprocessing.processor",
        ),
        (
            {
                "processor": "firered.stream_vad",
                "version": 2,
                "parameters": {},
            },
            "postprocessing.version",
        ),
        (
            {
                "processor": "firered.stream_vad",
                "version": 1,
                "parameters": {"unknown": 1},
            },
            "unknown postprocessing parameter",
        ),
        (
            {
                "processor": "firered.stream_vad",
                "version": 1,
                "parameters": {"min_silence_frame": 1.5},
            },
            "must be a positive integer",
        ),
    ],
)
def test_resolve_postprocessing_config_rejects_invalid_values(
    options,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_postprocessing_config(FireRedStreamVadConfig(), options)
