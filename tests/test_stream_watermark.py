from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import numpy as np

from fireredvad_ncnn.configs.stream_vad_config import FireRedStreamVadConfig
from fireredvad_ncnn.engine.worker import _Session, _filter_stream_events
from fireredvad_ncnn.inferencers.inferencer import Inferencer, PreparedStreamRequest, _StreamState
from fireredvad_ncnn.inferencers.stream_events import StreamEventPublisher
from fireredvad_ncnn.models.audio_feat import KaldifeatFbank
from fireredvad_ncnn.models.stream_vad_postprocessor import StreamVadPostprocessor
from fireredvad_ncnn.servicer.servicer import _events_to_responses
from tests.streaming_test_utils import assert_confirmed_prefixes


def _postprocessor(**overrides):
    options = dict(
        smooth_window_size=1, speech_threshold=0.5, pad_start_frame=3,
        min_speech_frame=3, max_speech_frame=20, min_silence_frame=4,
    )
    options.update(overrides)
    return StreamVadPostprocessor(**options)


def _probability_events(probs, chunk_size=1, **options):
    inferencer = Inferencer.__new__(Inferencer)
    state = _StreamState(postprocessor=_postprocessor(**options))
    state.stream_fbank = SimpleNamespace(reset=lambda: None)
    events = []
    for offset in range(0, len(probs), chunk_size):
        chunk = probs[offset:offset + chunk_size]
        events.extend(inferencer.apply_stream_probabilities(state, np.array(chunk), None))
        duration = ((offset + len(chunk)) * 10 + 15) / 1000
        events.extend(inferencer.finish_stream_request(
            state, PreparedStreamRequest([], duration, False)
        ))
    duration = (len(probs) * 10 + 15) / 1000
    events.extend(inferencer.finish_stream_request(
        state, PreparedStreamRequest([], duration, True)
    ))
    return events, duration


@pytest.mark.parametrize("endpoint_only", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 500])
@pytest.mark.parametrize("smooth,pad", [(1, 3), (5, 1), (3, 8)])
def test_every_response_preserves_final_membership(endpoint_only, chunk_size, smooth, pad):
    rng = random.Random(731)
    probs = [0.0] * 12 + [1.0] * 40 + [0.0] * 12
    probs += [float(rng.randrange(2)) for _ in range(400)]
    probs += [1.0] * 50
    raw, duration = _probability_events(
        probs, chunk_size, smooth_window_size=smooth, pad_start_frame=pad
    )
    session = _Session(state=None, endpoint_only=endpoint_only)
    events = [tuple(event) for event in _filter_stream_events(session, raw)]
    assert events == list(StreamEventPublisher(endpoint_only).publish(raw))
    # 同时验证线上 Duration 转换，水位和边界不能因舍入而交叉。
    wire = []
    for response in _events_to_responses(events):
        watermark = response.processed_time.ToNanoseconds() / 1e9
        for result in response.results:
            wire.append((result.start_time.ToNanoseconds() / 1e9,
                         result.end_time.ToNanoseconds() / 1e9, watermark))
    assert_confirmed_prefixes(wire, duration)


def test_silence_confirmation_stays_inside_effective_audio():
    raw, duration = _probability_events([0.0] * 10 + [1.0] * 5 + [0.0] * 4)
    assert (0.07, -1.0, 0.07) in raw
    # 第 19 帧才关闭，前面的三帧静音仍推进有效音水位。
    assert (-1.0, 0.18, 0.18) in raw
    assert (-1.0, -1.0, 0.18) in raw
    assert_confirmed_prefixes(raw, duration)


def test_failed_candidate_and_silence_interruption():
    raw, duration = _probability_events(
        [0.0] * 8 + [1.0] * 2 + [0.0] * 5 + [1.0] * 4
        + [0.0] * 2 + [1.0] * 2 + [0.0] * 4,
        max_speech_frame=100,
    )
    assert sum(start >= 0 for start, _, _ in raw) == 1
    assert sum(end >= 0 for _, end, _ in raw) == 1
    assert_confirmed_prefixes(raw, duration)


def _fake_model_inferencer():
    """真实重采样、FBank、分块与后处理，仅替换模型概率。"""
    inferencer = Inferencer.__new__(Inferencer)
    inferencer.config = FireRedStreamVadConfig(
        smooth_window_size=1, pad_start_frame=2, min_speech_frame=3,
        min_silence_frame=4, max_speech_frame=30,
    )
    inferencer.chunk_length = 1840
    inferencer.chunk_shift = 1600
    inferencer.audio_feat = SimpleNamespace(cmvn=None, new_stream=KaldifeatFbank)

    def forward(state, feats):
        offset = state.model_caches or 0
        probs = [float((i // 12) % 2) for i in range(offset, offset + len(feats))]
        return np.array(probs), offset + len(feats)

    inferencer._forward_stream = forward
    return inferencer


@pytest.mark.parametrize("sample_rate", [8000, 16000, 44100, 48000])
@pytest.mark.parametrize("samples", [0, 7, 1840, 32013])
@pytest.mark.parametrize("final_in_audio", [False, True])
def test_sync_async_alignment_and_exact_eof(sample_rate, samples, final_in_audio):
    inferencer = _fake_model_inferencer()
    audio = np.zeros(samples, dtype=np.int16)
    chunks = [(audio[i:i + 701], sample_rate, False) for i in range(0, samples, 701)]
    if final_in_audio and chunks:
        chunks[-1] = (*chunks[-1][:2], True)
    else:
        chunks.append((np.empty(0, dtype=np.int16), sample_rate, True))
    expected = list(inferencer.infer_stream(chunks))
    state = inferencer.new_stream_state()
    actual = []
    for audio_chunk, sr, final in chunks:
        prepared = inferencer.prepare_stream_request(state, audio_chunk, sr, final)
        for features in prepared.features:
            probs, cache = inferencer._forward_stream(state, features)
            actual.extend(inferencer.apply_stream_probabilities(state, probs, cache))
        actual.extend(inferencer.finish_stream_request(state, prepared))
    assert actual == expected
    duration = (samples * 1_000_000_000 // sample_rate) / 1e9
    for endpoint_only in (False, True):
        assert_confirmed_prefixes(list(StreamEventPublisher(endpoint_only).publish(actual)), duration)


def test_buffered_audio_does_not_advance_watermark():
    inferencer = _fake_model_inferencer()
    state = inferencer.new_stream_state()
    prepared = inferencer.prepare_stream_request(state, np.zeros(100), 16000, False)
    assert prepared.features == []
    assert prepared.input_duration > 0
    assert inferencer.finish_stream_request(state, prepared) == []
    assert state.processed_time == 0


def test_endpoint_only_holds_watermark_at_unpublished_start():
    raw, duration = _probability_events([0.0] * 10 + [1.0] * 10, max_speech_frame=100)
    published = list(StreamEventPublisher(True).publish(raw))
    segment_index = next(i for i, (start, _, _) in enumerate(published) if start >= 0)
    start, end, _ = published[segment_index]
    assert end == duration
    assert all(watermark <= start for _, _, watermark in published[:segment_index])
    assert published[-1] == (-1.0, -1.0, duration)


def test_failed_inference_does_not_publish_eof():
    inferencer = _fake_model_inferencer()

    def fail(*args):
        raise RuntimeError("model failed")

    inferencer._forward_stream = fail
    stream = inferencer.infer_stream([(np.zeros(3200), 16000, True)])
    with pytest.raises(RuntimeError, match="model failed"):
        next(stream)
