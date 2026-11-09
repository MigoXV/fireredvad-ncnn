"""Opt-in real INT8 artifact acceptance through the unchanged native and gRPC APIs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
import numpy as np
import psutil
import pytest
import soundfile as sf

from fireredvad_ncnn.inferencers.inferencer import Inferencer
from fireredvad_ncnn.proto import ux_vad_pb2 as pb, ux_vad_pb2_grpc as rpc
from fireredvad_ncnn.servicer.servicer import _events_to_responses


@pytest.fixture
def real_int8_input():
    model = os.environ.get("NCNN_INT8_TEST_MODEL")
    audio_path = os.environ.get("NCNN_INT8_TEST_WAV")
    if not model or not audio_path:
        pytest.skip("Set NCNN_INT8_TEST_MODEL and NCNN_INT8_TEST_WAV to real research artifacts")
    model = Path(model)
    metadata = json.loads((model / "metadata.json").read_text())
    assert metadata["quantization"]["int8_layers"]
    audio, rate = sf.read(audio_path, dtype="int16")
    assert rate == 16000 and audio.ndim == 1
    return model, audio[:rate * 4], rate


def chunks(audio, rate):
    for start in range(0, len(audio), 1600):
        yield audio[start:start + 1600], rate, False
    yield np.empty(0, np.int16), rate, True


def test_real_int8_concurrent_streams_are_isolated_and_cleaned_up(real_int8_input):
    model, audio, rate = real_int8_input
    with Inferencer(model, ncpu=4, worker_count=4, threads_per_worker=1) as inferencer:
        expected = list(inferencer.infer_stream(chunks(audio, rate)))
        assert any(start >= 0 for start, _, _ in expected), "The acceptance audio must contain speech"
        with ThreadPoolExecutor(max_workers=8) as pool:
            actual = list(pool.map(lambda _: list(inferencer.infer_stream(chunks(audio, rate))), range(8)))
        assert all(events == expected for events in actual)
        final = inferencer.engine.metrics()
        assert final["sessions"] == final["pending"] == final["inflight_bytes"] == 0
    assert inferencer.engine.closed


def test_real_int8_grpc_matches_native_and_releases_port(real_int8_input, tmp_path):
    model, audio, rate = real_int8_input
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    address = f"127.0.0.1:{port}"
    log_path = tmp_path / "grpc-int8.log"
    with log_path.open("w") as log:
        process = subprocess.Popen([
            sys.executable, "-m", "fireredvad_ncnn.commands.app", "serve", str(model),
            "--host", "127.0.0.1", "--port", str(port), "--num-threads", "2",
            "--engine-workers", "2", "--threads-per-worker", "1"], stdout=log, stderr=subprocess.STDOUT)
        try:
            with grpc.insecure_channel(address) as channel:
                health = health_pb2_grpc.HealthStub(channel)
                deadline = time.monotonic() + 20
                while True:
                    assert process.poll() is None, log_path.read_text()
                    try:
                        status = health.Check(health_pb2.HealthCheckRequest(), timeout=.2)
                        if status.status == health_pb2.HealthCheckResponse.SERVING:
                            break
                    except grpc.RpcError:
                        pass
                    if time.monotonic() >= deadline:
                        pytest.fail(f"Service did not become ready: {log_path.read_text()}")
                    time.sleep(.05)
                assert psutil.Process(process.pid).children(recursive=True) == []
                config = pb.DetectionConfig(encoding=1, sample_rate_hertz=rate)
                stub = rpc.UxVoiceActivityDetectorStub(channel)
                offline = stub.Detect(pb.DetectRequest(config=config, audio=audio.tobytes()), timeout=20)
                with Inferencer(model, ncpu=1) as native:
                    expected = native.infer(audio, rate)
                    stream_expected = list(_events_to_responses(native.infer_stream(chunks(audio, rate))))
                assert [(r.start_time.ToNanoseconds()/1e9, r.end_time.ToNanoseconds()/1e9)
                        for r in offline.results] == expected

                def requests():
                    yield pb.StreamingDetectRequest(config=config)
                    for chunk, _, _ in chunks(audio, rate):
                        if chunk.size:
                            yield pb.StreamingDetectRequest(audio_content=chunk.tobytes())

                stream_actual = list(stub.StreamingDetect(requests(), timeout=20))
                assert stream_actual == stream_expected
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                pytest.fail("INT8 gRPC service did not shut down gracefully")
    assert process.returncode == 0, log_path.read_text()
    assert "Native engine stopped" in log_path.read_text()
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
