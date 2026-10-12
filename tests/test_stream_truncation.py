from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fireredvad_ncnn.inferencers.inferencer import Inferencer
from fireredvad_ncnn.models.constants import SAMPLE_RATE
from fireredvad_ncnn.utils.audio_stream import AudioStreamResampler


class _Resettable:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


class _Postprocessor:
    def __init__(self, last_speech_start_frame: int = -1, frame_cnt: int = 0) -> None:
        self.last_speech_start_frame = last_speech_start_frame
        self.frame_cnt = frame_cnt

    def confirmed_frame_count(self):
        return 0


def _make_inferencer() -> tuple[Inferencer, list[tuple[int, bool]], list[object]]:
    inferencer = Inferencer.__new__(Inferencer)
    inferencer.chunk_length = 10
    inferencer.chunk_shift = 10
    calls: list[tuple[int, bool]] = []
    states: list[object] = []

    def new_stream_state():
        state = SimpleNamespace(
            postprocessor=_Postprocessor(),
            model_caches=None,
            audio_buffer=np.empty(0, dtype=np.int16),
            stream_fbank=_Resettable(),
            stream_started=False,
            resampler=AudioStreamResampler(),
            input_samples=0,
            processed_time=0.0,
        )
        states.append(state)
        return state

    def infer_chunk(state, audio_chunk, sample_rate, is_final=False):
        calls.append((len(audio_chunk), is_final))
        return []

    inferencer._new_stream_state = new_stream_state
    inferencer._infer_chunk = infer_chunk
    return inferencer, calls, states


def test_infer_stream_does_not_reset_after_single_oversized_chunk() -> None:
    inferencer, calls, states = _make_inferencer()
    chunk_samples = int(60 * SAMPLE_RATE / 1000)
    extra_samples = 15
    expected_processed_time = (chunk_samples + extra_samples) / SAMPLE_RATE
    stream = [
        (np.zeros(chunk_samples + extra_samples, dtype=np.int16), SAMPLE_RATE, True)
    ]

    events = list(inferencer.infer_stream(stream))

    assert len(states) == 1
    assert states[0].stream_fbank.reset_count == 1
    assert events[-1] == (-1.0, -1.0, expected_processed_time)
    assert sum(length for length, _ in calls) == chunk_samples + extra_samples


def test_infer_stream_does_not_reset_after_accumulated_limit() -> None:
    inferencer, calls, states = _make_inferencer()
    chunk_samples = int(60 * SAMPLE_RATE / 1000)
    extra_samples = 15
    expected_processed_time = (chunk_samples + extra_samples) / SAMPLE_RATE
    stream = [
        (np.zeros(chunk_samples // 2, dtype=np.int16), SAMPLE_RATE, False),
        (np.zeros(chunk_samples // 2, dtype=np.int16), SAMPLE_RATE, False),
        (np.zeros(extra_samples, dtype=np.int16), SAMPLE_RATE, True),
    ]

    events = list(inferencer.infer_stream(stream))

    assert len(states) == 1
    assert states[0].stream_fbank.reset_count == 1
    assert events[-1] == (-1.0, -1.0, expected_processed_time)
    assert sum(length for length, _ in calls) == chunk_samples + extra_samples


def test_infer_stream_keeps_event_timestamps_continuous_for_long_input() -> None:
    inferencer, _, states = _make_inferencer()
    chunk_samples = int(60 * SAMPLE_RATE / 1000)
    processed_time = (chunk_samples + 10) / SAMPLE_RATE
    calls = 0

    def infer_chunk(state, audio_chunk, sample_rate, is_final=False):
        nonlocal calls
        calls += 1
        if calls == 1:
            state.postprocessor.last_speech_start_frame = 2
            state.postprocessor.frame_cnt = 2
            return [
                SimpleNamespace(
                    is_speech_start=True,
                    is_speech_end=False,
                    speech_start_frame=2,
                )
            ]
        if calls == 8:
            state.postprocessor.last_speech_start_frame = -1
            state.postprocessor.frame_cnt = 8
            return [
                SimpleNamespace(
                    is_speech_start=False,
                    is_speech_end=True,
                    speech_end_frame=8,
                )
            ]

        return []

    inferencer._infer_chunk = infer_chunk
    stream = [
        (np.zeros(chunk_samples + 10, dtype=np.int16), SAMPLE_RATE, True)
    ]

    events = list(inferencer.infer_stream(stream))

    assert len(states) == 1
    assert (0.01, -1.0, 0.0) in events
    assert (-1.0, 0.07, 0.0) in events
    assert events[-1] == (-1.0, -1.0, processed_time)


def test_infer_stream_final_closes_open_segment() -> None:
    inferencer, _, _ = _make_inferencer()

    def infer_chunk(state, audio_chunk, sample_rate, is_final=False):
        state.postprocessor.last_speech_start_frame = 2
        state.postprocessor.frame_cnt = 12
        return []

    inferencer._infer_chunk = infer_chunk
    stream = [(np.zeros(12, dtype=np.int16), SAMPLE_RATE, True)]

    events = list(inferencer.infer_stream(stream))

    assert (-1.0, 12 / SAMPLE_RATE, 0.0) in events
    assert events[-1] == (-1.0, -1.0, 12 / SAMPLE_RATE)
