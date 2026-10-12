"""Real model, wire compatibility and service lifecycle tests."""
from pathlib import Path
import os
import subprocess
import sys
import time

import grpc
import numpy as np
import psutil
import pytest
import soundfile as sf

from fireredvad_ncnn.benchmark import Server, stream
from fireredvad_ncnn.proto import ux_vad_pb2 as pb, ux_vad_pb2_grpc as rpc

MODEL = Path(os.environ.get("NCNN_TEST_MODEL", "model-bin/stream-vad"))
REFERENCE = Path(os.environ.get("VAD_REFERENCE_PROJECT", "/workspace/opusi/fireredvad"))
SOURCE = REFERENCE / "model-bin/MigoXV/firered-vad-stream"


def test_real_service_matches_original_and_cleans_up(tmp_path):
    if not MODEL.exists() or not SOURCE.exists():
        pytest.skip("Local converted and reference models required")
    path = next((REFERENCE/"data-bin/0713").glob("*.wav"))
    audio, sr = sf.read(path, dtype="int16")
    audio = audio[:160000]
    cases = [(audio, sr), (np.zeros(16000, np.int16),16000),
             (audio[:21], sr), (audio[::2],8000), (np.empty(0,np.int16),16000)]
    outputs = {}
    for backend, model in [("torch",SOURCE),("ncnn",MODEL)]:
        with Server(backend,model,REFERENCE,tmp_path/f"{backend}.log") as server:
            results=[]
            for values, rate in cases:
                result=stream(server.address,values.astype('<i2').tobytes(),sr=rate)
                assert result['ok'], result
                results.append(result['events'])
            with grpc.insecure_channel(server.address) as channel:
                stub=rpc.UxVoiceActivityDetectorStub(channel)
                for threshold in [None,0.7]:
                    config=pb.DetectionConfig(encoding=1,sample_rate_hertz=sr)
                    if threshold is not None:
                        config.postprocessing.processor='firered.stream_vad'
                        config.postprocessing.version=1
                        config.postprocessing.parameters.update({'speech_threshold':threshold})
                    detected=stub.Detect(pb.DetectRequest(config=config,audio=audio.tobytes()),timeout=30)
                    results.append(detected.SerializeToString())
                with pytest.raises(grpc.RpcError) as error:
                    stub.Detect(pb.DetectRequest(config=pb.DetectionConfig(encoding=1,sample_rate_hertz=0),audio=b'\0\0'),timeout=3)
                assert error.value.code()==grpc.StatusCode.INVALID_ARGUMENT
            outputs[backend]=results
    assert outputs['torch']==outputs['ncnn']


def test_worker_failure_exits_service(tmp_path):
    if not MODEL.exists():
        pytest.skip('Need converted model')
    with Server('ncnn',MODEL,REFERENCE,tmp_path/'failure.log') as server:
        workers=[p for p in psutil.Process(server.process.pid).children(recursive=True)
                 if 'spawn_main' in ' '.join(p.cmdline())]
        assert len(workers)==1
        workers[0].kill()
        deadline=time.monotonic()+15
        while server.process.poll() is None and time.monotonic()<deadline:
            time.sleep(0.1)
        assert server.process.poll() is not None


def test_inference_without_torch_imports():
    if not MODEL.exists():
        pytest.skip('Need converted model')
    code='''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','fireredvad','transformers','pnnx'}:
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
