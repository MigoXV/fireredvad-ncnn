"""Real model, wire compatibility and service lifecycle tests."""
from pathlib import Path
import os
import subprocess
import sys

import grpc
import numpy as np
import psutil
import pytest
import soundfile as sf

from fireredvad_ncnn.benchmark import Server, stream
from fireredvad_ncnn.proto import ux_vad_pb2 as pb, ux_vad_pb2_grpc as rpc
from fireredvad_ncnn.inferencers.inferencer import Inferencer
from fireredvad_ncnn.servicer.servicer import _events_to_responses
from fireredvad_ncnn.configs.stream_vad_config import FireRedStreamVadConfig

MODEL = Path(os.environ.get("NCNN_TEST_MODEL", "model-bin/stream-vad"))


def test_native_service_matches_python_api_and_cleans_up(tmp_path):
    if not MODEL.exists():
        pytest.skip("Need converted model")
    speech = Path(os.environ.get("VAD_TEST_WAV", "model-bin/speech.wav"))
    audio, sr = sf.read(speech, dtype="int16") if speech.exists() else (np.zeros(16000,np.int16),16000)
    with Server(MODEL, tmp_path / "native.log") as server:
        for values, rate in [(audio,sr), (audio[::2],sr//2),
                             (audio[:7],sr), (np.empty(0,np.int16),sr)]:
            chunks = [(values[i:i+rate//10],rate,False) for i in range(0,len(values),rate//10)]
            chunks.append((np.empty(0,np.int16),rate,True))
            with Inferencer(MODEL) as engine:
                events = list(engine.infer_stream(chunks))
            expected = [{"watermark_ns": response.processed_time.ToNanoseconds(),
                "segments": [[r.start_time.ToNanoseconds(),r.end_time.ToNanoseconds()] for r in response.results]}
                for response in _events_to_responses(events)]
            actual = stream(server.address, values.tobytes(), sr=rate)
            assert actual["ok"], actual
            assert actual["events"] == expected
        with grpc.insecure_channel(server.address) as channel:
            stub = rpc.UxVoiceActivityDetectorStub(channel)
            for threshold in [0.2,0.8]:
                config = pb.DetectionConfig(encoding=1,sample_rate_hertz=sr)
                config.postprocessing.processor = "firered.stream_vad"
                config.postprocessing.version = 1
                config.postprocessing.parameters.update({"speech_threshold":threshold})
                response = stub.Detect(pb.DetectRequest(config=config,audio=audio.tobytes()),timeout=30)
                with Inferencer(MODEL,stream_vad_config=FireRedStreamVadConfig(speech_threshold=threshold)) as engine:
                    expected = engine.infer(audio,sr)
                assert [(r.start_time.ToNanoseconds()/1e9,r.end_time.ToNanoseconds()/1e9) for r in response.results] == expected


def test_service_uses_one_process(tmp_path):
    if not MODEL.exists():
        pytest.skip('Need converted model')
    server = Server(MODEL, tmp_path/'single-process.log')
    server.command[server.command.index('--log-level') + 1] = 'INFO'
    with server:
        assert psutil.Process(server.process.pid).children(recursive=True) == []
        assert stream(server.address, np.zeros(3200, np.int16).tobytes())['ok']
    assert server.process.returncode == 0
    assert 'Native engine stopped' in (tmp_path/'single-process.log').read_text()



def test_inference_without_torch_imports():
    if not MODEL.exists():
        pytest.skip('Need converted model')
    code='''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','fireredvad','transformers','pnnx','ncnn','scipy','soxr','kaldiio','kaldi_native_fbank','zmq','msgpack'}:
            raise AssertionError('Runtime dependency: '+fullname)
sys.meta_path.insert(0,Block())
import numpy as np
from fireredvad_ncnn.inferencers.inferencer import Inferencer
from fireredvad_ncnn.commands.app import app
engine=Inferencer(sys.argv[1])
assert engine.infer(np.zeros(16000,np.int16),16000)==[]
engine.close()
'''
    subprocess.run([sys.executable,'-c',code,str(MODEL.resolve())],check=True,timeout=20)


def test_cli_exits_nonzero_when_native_shutdown_stalls():
    code = """
import asyncio, logging
from fireredvad_ncnn.commands.app import _close_engine
class StuckEngine:
    async def close(self):
        raise TimeoutError('native call stalled')
asyncio.run(_close_engine(StuckEngine(), logging.getLogger('test')))
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert 'Native shutdown deadline exceeded' in result.stderr
