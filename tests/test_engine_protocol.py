from __future__ import annotations

import time

import pytest

from fireredvad_ncnn.configs.engine_config import EngineConfig
from fireredvad_ncnn.configs.serving_config import ServingConfig
from fireredvad_ncnn.engine.protocol import (
    PROTOCOL_VERSION,
    FailedPreconditionError,
    InvalidArgumentError,
    Operation,
    pack_metadata,
    unpack_metadata,
    validate_request,
)


def _metadata(**overrides):
    metadata = {
        "version": PROTOCOL_VERSION,
        "op": Operation.DETECT.value,
        "generation": "generation",
        "request_id": "request",
        "session_id": "",
        "sequence": 0,
        "deadline_ns": time.monotonic_ns() + 1_000_000_000,
        "payload_length": 4,
    }
    metadata.update(overrides)
    return metadata


def test_metadata_round_trip_and_validation() -> None:
    payload = b"\x00\x00\x01\x00"
    metadata = unpack_metadata(pack_metadata(_metadata()))

    assert validate_request(metadata, payload, 1024) is Operation.DETECT


def test_protocol_rejects_unknown_version() -> None:
    with pytest.raises(FailedPreconditionError):
        validate_request(_metadata(version=PROTOCOL_VERSION + 1), b"\x00" * 4, 1024)


def test_protocol_rejects_mismatched_payload_length() -> None:
    with pytest.raises(InvalidArgumentError):
        validate_request(_metadata(payload_length=8), b"\x00" * 4, 1024)


def test_protocol_requires_fixed_session_fields() -> None:
    metadata = _metadata()
    del metadata["session_id"]

    with pytest.raises(InvalidArgumentError, match="session_id"):
        validate_request(metadata, b"\x00" * 4, 1024)


def test_engine_config_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError, match="max_sessions"):
        EngineConfig(max_sessions=0)


def test_default_capacity_supports_128_sessions() -> None:
    engine_config = EngineConfig()

    assert engine_config.session_queue_capacity == 2
    assert engine_config.max_sessions == 128
    assert engine_config.ingress_capacity == 2048
    assert engine_config.queue_timeout_ms == 5000
    assert engine_config.detect_queue_timeout_ms == 30000
    assert ServingConfig().max_concurrent_rpcs == 256
