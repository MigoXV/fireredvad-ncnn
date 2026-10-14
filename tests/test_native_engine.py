"""Native migration correctness, ownership, concurrency and capacity tests."""
from __future__ import annotations

import ctypes as ct
import json
import os
import subprocess
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from fireredvad_ncnn.configs.engine_config import EngineConfig
from fireredvad_ncnn.configs.stream_vad_config import FireRedStreamVadConfig
from fireredvad_ncnn.inferencers.inferencer import Inferencer
from fireredvad_ncnn.native import NativeEngine, NativeError, Result
from tests.reference.inferencer import Inferencer as Reference
from tests.reference.audio_feat import AudioFeat
from tests.streaming_test_utils import assert_confirmed_prefixes

MODEL = Path(os.environ.get("NCNN_TEST_MODEL", "model-bin/stream-vad"))
PROBE = Path(os.environ.get("NCNN_NUMERICAL_PROBE", "build/x86_64/frv_numerical_probe"))
SPEECH = Path(os.environ.get("VAD_TEST_WAV", "model-bin/speech.wav"))
pytestmark = pytest.mark.skipif(not (MODEL / "metadata.json").exists(), reason="converted model required")


@pytest.fixture
def native():
    with NativeEngine(MODEL) as engine:
        yield engine


def chunks(audio, rate, size=701, final_in_audio=False):
    result = [(audio[i:i+size], rate, False) for i in range(0, len(audio), size)]
    if final_in_audio and result:
        result[-1] = (*result[-1][:2], True)
    else:
        result.append((np.empty(0, np.int16), rate, True))
    return result


def _probe(tmp_path, requests):
    if not PROBE.exists():
        pytest.skip("build native numerical probe")
    path = tmp_path / "features.json"
    path.write_text(json.dumps(requests))
    result = subprocess.run(shlex.split(os.environ.get("NCNN_PROBE_EMULATOR", "")) +
                            [str(PROBE.resolve()), str(MODEL.resolve()), str(path)],
                            capture_output=True, text=True, check=True, timeout=60)
    return [json.loads(line) for line in result.stdout.splitlines()]


def test_probabilities_and_short_block_cache_continuation(tmp_path):
    torch = pytest.importorskip("torch")
    from fireredvad_ncnn.exporting.convert import load_reference
    source = Path(os.environ.get("VAD_REFERENCE_MODEL", "model-bin/source"))
    if not source.exists():
        pytest.skip("source model required")
    model, _ = load_reference(source)
    torch.set_num_threads(4)
    rng = np.random.default_rng(47)
    features = [rng.normal(size=(n, 80)).astype(np.float32) for n in (10,10,3,1,9,25,10,100)]
    actual = _probe(tmp_path, [x.ravel().tolist() for x in features])
    states = [torch.zeros(1, 128, 19) for _ in range(8)]
    with torch.inference_mode():
        for x, got in zip(features, actual):
            probs, states = model(torch.from_numpy(x)[None], states)
            np.testing.assert_allclose(got["probs"], probs.numpy().ravel(), atol=1e-4, rtol=1e-4)
            for a, b in zip(got["caches"], states):
                np.testing.assert_allclose(a, b.numpy().ravel(), atol=1e-4, rtol=1e-4)


def test_fbank_cmvn_numeric_alignment(tmp_path):
    audio = np.random.default_rng(8).integers(-30000,30000,16007,dtype=np.int16)
    actual = _probe(tmp_path, [{"audio": audio.tolist()}])[0]
    features, _ = AudioFeat(str(MODEL / "cmvn.ark")).extract((audio, 16000))
    np.testing.assert_allclose(np.array(actual["features"]).reshape(-1,80), features, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("rate", [8000,16000,44100,48000])
@pytest.mark.parametrize("samples", [0,7,1840,32013])
@pytest.mark.parametrize("final", [False,True])
def test_stream_events_and_watermarks_match_reference(rate, samples, final):
    audio = np.random.default_rng(9).integers(-8000,8000,samples,dtype=np.int16)
    stream = chunks(audio,rate,final_in_audio=final)
    reference = Reference(MODEL)
    try:
        expected = list(reference.infer_stream(stream))
    finally:
        reference.close()
    with Inferencer(MODEL) as engine:
        actual = list(engine.infer_stream(stream))
    assert actual == expected
    assert_confirmed_prefixes(actual, (samples*1_000_000_000//rate)/1e9)


def test_real_speech_offline_and_stream():
    if not SPEECH.exists():
        pytest.skip("set VAD_TEST_WAV")
    import soundfile as sf
    audio, rate = sf.read(SPEECH, dtype="int16")
    reference = Reference(MODEL)
    try:
        offline = reference.infer(audio, rate)
        stream = list(reference.infer_stream(chunks(audio,rate,size=1600)))
    finally:
        reference.close()
    with Inferencer(MODEL) as engine:
        assert engine.infer(audio,rate) == offline
        actual = list(engine.infer_stream(chunks(audio,rate,size=1600)))
    assert actual == stream
    assert any(start >= 0 for start, _, _ in actual)


def test_real_speech_probabilities_and_caches(tmp_path):
    torch = pytest.importorskip("torch")
    from fireredvad_ncnn.exporting.convert import load_reference
    source = Path(os.environ.get("VAD_REFERENCE_MODEL","model-bin/source"))
    if not SPEECH.exists() or not source.exists():
        pytest.skip("real speech and source model required")
    import soundfile as sf
    audio,rate = sf.read(SPEECH,dtype="int16")
    actual = _probe(tmp_path,[{"audio":audio.tolist(),"rate":rate}])[0]
    features,_ = AudioFeat(str(MODEL / "cmvn.ark")).extract((audio,rate))
    model,_ = load_reference(source)
    torch.set_num_threads(4)
    with torch.inference_mode():
        expected,caches = model(torch.from_numpy(features)[None],[torch.zeros(1,128,19) for _ in range(8)])
    np.testing.assert_allclose(actual["probs"],expected.numpy().ravel(),atol=1e-4,rtol=1e-4)
    for a,b in zip(actual["caches"],caches):
        np.testing.assert_allclose(a,b.numpy().ravel(),atol=1e-4,rtol=1e-4)


@pytest.mark.parametrize("rate",[8000,44100,48000])
def test_offline_uses_unified_soxr_hq(rate):
    import soxr
    audio = np.random.default_rng(41).integers(-8000,8000,rate*2+7,dtype=np.int16)
    converted = np.clip(np.rint(soxr.resample(audio.astype(np.float32),rate,16000,quality="HQ")),-32768,32767).astype(np.int16)
    reference = Reference(MODEL)
    try:
        expected = reference.infer(converted,16000)
    finally:
        reference.close()
    with Inferencer(MODEL) as engine:
        assert engine.infer(audio,rate) == expected


def test_endpoint_only_native_events(native):
    from tests.reference.stream_events import StreamEventPublisher
    if not SPEECH.exists():
        pytest.skip("set VAD_TEST_WAV")
    import soundfile as sf
    audio,rate = sf.read(SPEECH,dtype="int16")
    stream = chunks(audio,rate,size=1600)
    reference = Reference(MODEL)
    try:
        expected = list(StreamEventPublisher(True).publish(reference.infer_stream(stream)))
    finally:
        reference.close()
    session = native.session_open(endpoint_only=True)
    actual = []
    for sequence,(a,sr,final) in enumerate(stream,1):
        actual.extend(native.wait(native.submit(a,sr,session=session,sequence=sequence,is_final=final)))
    native.session_close(session)
    assert actual == expected


def test_multisession_isolation_and_native_parallel_execution():
    audio = np.random.default_rng(87).integers(-6000,6000,32013,dtype=np.int16)
    with Inferencer(MODEL,ncpu=4,worker_count=4,threads_per_worker=1) as engine:
        expected = list(engine.infer_stream(chunks(audio,16000)))
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: list(engine.infer_stream(chunks(audio,16000))), range(8)))
        assert all(r == expected for r in results)
        assert engine.engine.metrics()["sessions"] == 0
        assert engine.engine.metrics()["pending"] == 0


def test_invalid_sequence_rate_and_finalization(native):
    session = native.session_open()
    with pytest.raises(NativeError, match="sequence"):
        native.submit(np.zeros(0,np.int16),16000,session=session,sequence=2)
    first = native.submit(np.zeros(1600,np.int16),16000,session=session,sequence=1)
    native.wait(first)
    bad = native.submit(np.zeros(0,np.int16),8000,session=session,sequence=2)
    with pytest.raises(NativeError, match="rate"):
        native.wait(bad)
    with pytest.raises(NativeError):
        native.submit(np.zeros(0,np.int16),16000,session=session,sequence=3)
    session = native.session_open()
    last = native.submit(np.zeros(0,np.int16),16000,session=session,sequence=1,is_final=True)
    assert native.wait(last) == [(-1.0,-1.0,0.0)]
    with pytest.raises(NativeError, match="finalized"):
        native.submit(np.zeros(0,np.int16),16000,session=session,sequence=2)
    native.session_close(session)


@pytest.mark.parametrize("cancel", [False,True])
def test_execution_timeout_cancel_invalidates_session(native,cancel):
    session = native.session_open()
    request = native.submit(np.zeros(16000*120,np.int16),16000,session=session,sequence=1,
                            timeout_ms=0 if cancel else 1)
    if cancel:
        native.cancel(request)
    with pytest.raises(NativeError) as error:
        native.wait(request)
    assert error.value.status == (7 if cancel else 3)
    with pytest.raises(NativeError):
        native.submit(np.zeros(0,np.int16),16000,session=session,sequence=2)
    assert native.metrics()["pending"] == 0
    assert native.metrics()["sessions"] == 0


def test_completed_results_remain_charged_until_freed():
    with NativeEngine(MODEL,engine_config=EngineConfig(ingress_capacity=1)) as native:
        request = native.submit(np.zeros(1600,np.int16),16000)
        result = ct.POINTER(Result)()
        native._check(native._lib.frv_poll(native._handle,request,10000,ct.byref(result)))
        try:
            with pytest.raises(NativeError) as error:
                native.submit(np.zeros(1600,np.int16),16000)
            assert error.value.status == 2
            assert native.metrics()["pending"] == 1
        finally:
            native._lib.frv_result_free(result)
        assert native.metrics()["pending"] == 0
        native.wait(native.submit(np.zeros(0,np.int16),16000))


def test_input_is_copied_and_errors_do_not_consume_sequence(native):
    session = native.session_open()
    with pytest.raises(NativeError):
        native.submit(np.array([np.nan],np.float32),16000,session=session,sequence=1)
    if SPEECH.exists():
        import soundfile as sf
        audio,rate = sf.read(SPEECH,dtype="int16")
    else:
        audio,rate = np.zeros(16000,np.int16),16000
    isolated = native.session_open()
    original = audio.copy()
    reference_events = native.wait(native.submit(original,rate,session=isolated,sequence=1,is_final=True))
    native.session_close(isolated)
    request = native.submit(audio,rate,session=session,sequence=1,is_final=True)
    audio[:] = 0
    assert native.wait(request) == reference_events
    native.session_close(session)
    assert native.metrics()["pending"] == 0


def test_cpu_budget_and_close():
    with pytest.raises(NativeError, match="CPU"):
        NativeEngine(MODEL,cpu_budget=4,worker_count=4,threads_per_worker=4)
    engine = NativeEngine(MODEL)
    engine.close();engine.close()
    with pytest.raises(RuntimeError,match="closed"):
        engine.submit(np.zeros(0,np.int16),16000)


def test_native_queue_capacity_and_retry_sequence():
    with NativeEngine(MODEL,engine_config=EngineConfig(session_queue_capacity=1,max_sessions=1)) as native:
        session = native.session_open()
        with pytest.raises(NativeError) as error:
            native.session_open()
        assert error.value.status == 2
        first = native.submit(np.zeros(16000*60,np.int16),16000,session=session,sequence=1)
        second = native.submit(np.zeros(0,np.int16),16000,session=session,sequence=2)
        with pytest.raises(NativeError) as error:
            native.submit(np.zeros(0,np.int16),16000,session=session,sequence=3)
        assert error.value.status == 2
        native.cancel(first)
        for request in [first,second]:
            with pytest.raises(NativeError):
                native.wait(request)
        assert native.metrics()["sessions"] == 0


def test_fairness_long_offline_request_yields_to_short_request(native):
    long = native.submit(np.zeros(16000*60,np.int16),16000)
    short = native.submit(np.zeros(1600,np.int16),16000)
    completion = native.poll(wait_ms=10000)
    assert completion.request_id == short
    native.cancel(long)
    with pytest.raises(NativeError):
        native.wait(long)


@pytest.mark.parametrize("dtype",[np.int16,np.float32,np.float64])
def test_float_pcm_and_noncontiguous_input(native,dtype):
    audio = np.zeros(16000,dtype=dtype)[::2]
    request = native.submit(audio,16000)
    assert native.wait(request) == []
    with pytest.raises(ValueError,match="mono"):
        native.submit(np.zeros((3,2)),16000)


def test_offline_chunk_reset_and_request_postprocessing():
    if not SPEECH.exists():
        pytest.skip("set VAD_TEST_WAV")
    import soundfile as sf
    audio, rate = sf.read(SPEECH,dtype="int16")
    for threshold in [0.2,0.8]:
        config = FireRedStreamVadConfig(speech_threshold=threshold,chunk_max_frame=17)
        reference = Reference(MODEL,stream_vad_config=config)
        try:
            expected = reference.infer(audio,rate)
        finally:
            reference.close()
        with Inferencer(MODEL,stream_vad_config=config) as engine:
            assert engine.infer(audio,rate) == expected


@pytest.mark.skipif(os.environ.get("NCNN_TEST_ALL_CHUNKS") != "1", reason="opt-in 1..19 export matrix")
@pytest.mark.parametrize("chunk_frames",range(1,20))
def test_every_exported_chunk_size(tmp_path,chunk_frames):
    from fireredvad_ncnn.exporting.convert import export_model
    source = Path(os.environ.get("VAD_REFERENCE_MODEL","model-bin/source"))
    model_dir = tmp_path / "model"
    export_model(source,model_dir,chunk_frames)
    rng = np.random.default_rng(471)
    features = [rng.normal(size=(n,80)).astype(np.float32) for n in
                [chunk_frames,max(1,chunk_frames-1),1,chunk_frames+3]]
    path = tmp_path / "features.json"
    path.write_text(json.dumps([x.ravel().tolist() for x in features]))
    output = subprocess.check_output(shlex.split(os.environ.get("NCNN_PROBE_EMULATOR", "")) +
        [str(PROBE.resolve()),str(model_dir),str(path)],text=True,timeout=60)
    from tests.reference.model_runtime import VadModelRuntime
    reference = VadModelRuntime(model_dir)
    cache = None
    try:
        for feature,line in zip(features,output.splitlines()):
            expected,cache = reference.forward_chunk(feature,cache)
            actual = json.loads(line)
            np.testing.assert_allclose(actual["probs"],expected.ravel(),atol=1e-4,rtol=1e-4)
            for a,b in zip(actual["caches"],cache):
                np.testing.assert_allclose(a,b.ravel(),atol=1e-4,rtol=1e-4)
    finally:
        reference.close()
    audio = rng.integers(-5000,5000,3213,dtype=np.int16)
    reference = Reference(model_dir,stream_chunk_frame=chunk_frames)
    try:
        expected = list(reference.infer_stream(chunks(audio,16000)))
    finally:
        reference.close()
    with Inferencer(model_dir,stream_chunk_frame=chunk_frames) as engine:
        assert list(engine.infer_stream(chunks(audio,16000))) == expected


def test_model_checksum_and_warmup_failure_cleanup(tmp_path):
    import hashlib
    import shutil
    root = tmp_path / "model"
    shutil.copytree(MODEL,root)
    (root / "model.ncnn.param").write_text("not a graph\n")
    with pytest.raises(NativeError,match="checksum"):
        NativeEngine(root)
    metadata = json.loads((root / "metadata.json").read_text())
    metadata["files"]["model.ncnn.param"] = hashlib.sha256((root / "model.ncnn.param").read_bytes()).hexdigest()
    (root / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(NativeError,match="load"):
        NativeEngine(root)


def test_poll_timeout_does_not_cancel_and_result_survives_engine_close():
    native = NativeEngine(MODEL)
    request = native.submit(np.zeros(16000*60,np.int16),16000)
    assert native.poll(request,wait_ms=1) is None
    native.cancel(request)
    with pytest.raises(NativeError):
        native.wait(request)
    request = native.submit(np.zeros(0,np.int16),16000)
    result = ct.POINTER(Result)()
    native._check(native._lib.frv_poll(native._handle,request,10000,ct.byref(result)))
    native.close()
    try:
        assert result.contents.status == 0
        assert result.contents.event_count == 0
    finally:
        native._lib.frv_result_free(result)


def test_native_cli_raw_eof_and_signal(tmp_path):
    binary = Path("build/x86_64/fireredvad")
    if not binary.exists():
        pytest.skip("native CLI required")
    completed = subprocess.run([str(binary),"stream",str(MODEL),"-","--raw-rate","16000"],
                               input=b"\0\0"*7,capture_output=True,check=True,timeout=20)
    assert json.loads(completed.stdout.splitlines()[-1])["events"][-1] == [-1,-1,7/16000]
    process = subprocess.Popen([str(binary),"stream",str(MODEL),"-","--raw-rate","16000"],
                               stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        time.sleep(0.2)
        process.terminate()
        assert process.wait(timeout=5) == 130
    finally:
        if process.poll() is None:
            process.kill();process.wait(timeout=5)
        process.stdin.close();process.stdout.close();process.stderr.close()


def test_handles_do_not_alias_another_engine_or_recreated_engine():
    with NativeEngine(MODEL) as first:
        session = first.session_open()
        request = first.submit(np.zeros(0,np.int16),16000)
        first.wait(request)
    with NativeEngine(MODEL) as second:
        other = second.session_open()
        assert other != session
        with pytest.raises(NativeError) as error:
            second.session_close(session)
        assert error.value.status == 4
        with pytest.raises(NativeError):
            second.cancel(request)
        second.session_close(other)
