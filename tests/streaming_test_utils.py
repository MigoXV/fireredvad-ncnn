from __future__ import annotations

from collections.abc import Generator, Iterable
from pathlib import Path

import numpy as np
import soundfile as sf

from fireredvad_ncnn.models.constants import FRAME_SHIFT_SAMPLE, SAMPLE_RATE

DEFAULT_STREAM_CHUNK_FRAME = 10
DEFAULT_INPUT_CHUNK_MS = DEFAULT_STREAM_CHUNK_FRAME * FRAME_SHIFT_SAMPLE * 1000 // SAMPLE_RATE


def load_int16_audio(wav_path: str | Path):
    wav_np, sample_rate = sf.read(wav_path, dtype="int16")
    if wav_np.ndim > 1:
        wav_np = wav_np[:, 0]
    return wav_np, sample_rate


def build_stream_input(
    wav_np: np.ndarray,
    sample_rate: int,
    input_chunk_ms: int = DEFAULT_INPUT_CHUNK_MS,
) -> Generator[tuple[np.ndarray, int, bool], None, None]:
    chunk_size = int(input_chunk_ms * sample_rate / 1000)
    if chunk_size <= 0:
        raise ValueError("input_chunk_ms must be positive")

    total = len(wav_np)
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        yield wav_np[start:end], sample_rate, end >= total


def collect_stream_timestamps(
    events: Iterable[tuple[float, float, float]],
    fallback_end_time: float | None = None,
):
    timestamps = []
    current_start = None
    for start_sec, end_sec, _ in events:
        if start_sec >= 0.0:
            current_start = start_sec
        elif end_sec >= 0.0 and current_start is not None:
            timestamps.append((current_start, end_sec))
            current_start = None

    if current_start is not None and fallback_end_time is not None:
        timestamps.append((current_start, fallback_end_time))
    return timestamps



def assert_confirmed_prefixes(events, duration):
    """每条响应的已确认区间必须与最终分段一致，包括尚未闭合的段。"""
    closed = []
    opened = None
    snapshots = []
    previous = 0.0
    for start, end, watermark in events:
        assert previous <= watermark <= duration
        if start >= 0:
            assert start >= previous
            assert opened is None
            opened = start
        if end >= 0:
            assert end >= previous
            assert opened is not None
            closed.append((opened, end))
            opened = None
        intervals = closed + ([(opened, duration)] if opened is not None else [])
        snapshots.append((watermark, _prefix(intervals, watermark)))
        previous = watermark
    assert opened is None
    assert previous == duration
    for watermark, prefix in snapshots:
        assert prefix == _prefix(closed, watermark)


def _prefix(intervals, watermark):
    return [(start, min(end, watermark)) for start, end in intervals
            if start < min(end, watermark)]
