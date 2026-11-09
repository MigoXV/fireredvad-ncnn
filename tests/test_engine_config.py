"""Retained public service capacity defaults after IPC removal."""
import pytest
from fireredvad_ncnn.configs.engine_config import EngineConfig
from fireredvad_ncnn.configs.serving_config import ServingConfig


def test_engine_config_rejects_non_positive_capacity():
    with pytest.raises(ValueError,match='max_sessions'): EngineConfig(max_sessions=0)


def test_default_capacity_supports_128_sessions():
    config=EngineConfig()
    assert config.max_sessions==128 and config.session_queue_capacity==2
    assert config.ingress_capacity==2048 and config.queue_timeout_ms==5000
    assert config.detect_queue_timeout_ms==30000
    assert ServingConfig().max_concurrent_rpcs==256
