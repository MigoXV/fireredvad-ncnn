from pathlib import Path
import os

import numpy as np
import pytest

from tests.reference.model_runtime import VadModelRuntime

MODEL = Path(os.environ.get("NCNN_TEST_MODEL", "model-bin/stream-vad"))
SOURCE = Path(os.environ.get("VAD_REFERENCE_MODEL", "/workspace/opusi/fireredvad/model-bin/MigoXV/firered-vad-stream")).resolve()


@pytest.fixture
def runtime():
    if not MODEL.exists():
        pytest.skip("Export a model or set NCNN_TEST_MODEL")
    rt = VadModelRuntime(MODEL)
    yield rt
    rt.close()


def test_stream_probabilities_and_caches(runtime):
    torch = pytest.importorskip("torch")
    from fireredvad_ncnn.exporting.convert import load_reference
    if not SOURCE.exists():
        pytest.skip("Set VAD_REFERENCE_MODEL")
    torch.set_num_threads(4)
    model, _ = load_reference(SOURCE)
    rng = np.random.default_rng(42)
    reference = [torch.zeros(1, 128, 19) for _ in range(8)]
    actual = None
    with torch.inference_mode():
        for length in [10, 10, 3, 1, 9, 25, 10, 100]:
            features = rng.normal(size=(length, 80)).astype(np.float32)
            expected, reference = model(torch.from_numpy(features)[None], reference)
            output, actual = runtime.forward_chunk(features, actual)
            np.testing.assert_allclose(output, expected.numpy(), atol=1e-4, rtol=1e-4)
            for got, wanted in zip(actual, reference):
                np.testing.assert_allclose(got, wanted.numpy(), atol=1e-4, rtol=1e-4)


def test_sessions_are_isolated_and_candidate_does_not_mutate(runtime):
    a = runtime.allocate_session_cache()
    before = [x.copy() for x in a]
    x = np.random.default_rng(4).normal(size=(10,80)).astype(np.float32)
    expected, candidate = runtime.forward_chunk(x, a)
    runtime.forward_chunk(x * 2, runtime.allocate_session_cache())
    repeated, _ = runtime.forward_chunk(x, a)
    np.testing.assert_array_equal(expected, repeated)
    for old, saved, new in zip(a, before, candidate):
        np.testing.assert_array_equal(old, saved)
        assert not np.shares_memory(old, new)


def test_empty_and_closed(runtime):
    probs, caches = runtime.forward_chunk(np.empty((0, 80), np.float32))
    assert probs.shape == (1, 0, 1)
    assert len(caches) == 8
    runtime.close()
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.forward_chunk(np.zeros((10,80), np.float32))


def test_real_audio_probabilities_and_cache_alignment(runtime):
    torch = pytest.importorskip("torch")
    sf = pytest.importorskip("soundfile")
    from fireredvad_ncnn.exporting.convert import load_reference
    from tests.reference.audio_feat import AudioFeat
    audio_dir = SOURCE.parents[2] / "data-bin/0713"
    if not SOURCE.exists() or not audio_dir.exists():
        pytest.skip("Local reference audio required")
    model, _ = load_reference(SOURCE)
    torch.set_num_threads(4)
    audio, sr = sf.read(next(audio_dir.glob('*.wav')), dtype='int16')
    import soxr
    audio = np.rint(soxr.resample(audio[:sr*10].astype(np.float32),sr,16000)).clip(-32768,32767).astype(np.int16)
    feats, _ = AudioFeat(str(SOURCE/'cmvn.ark')).extract((audio,16000))
    reference = [torch.zeros(1,128,19) for _ in range(8)]
    actual = None
    with torch.inference_mode():
        for offset in range(0,len(feats),10):
            x = feats[offset:offset+10]
            wanted, reference = model(torch.from_numpy(x)[None],reference)
            got, actual = runtime.forward_chunk(x,actual)
            np.testing.assert_allclose(got,wanted.numpy(),atol=1e-4,rtol=1e-4)
            for a,b in zip(actual,reference):
                np.testing.assert_allclose(a,b.numpy(),atol=1e-4,rtol=1e-4)
