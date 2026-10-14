"""Pre-migration streaming resampling oracle."""
from __future__ import annotations

from collections.abc import Generator, Iterable

import numpy as np
import soxr

from fireredvad_ncnn.models.constants import SAMPLE_RATE


class AudioStreamResampler:
    """Stateful single-channel streaming resampler owned by one VAD session."""

    def __init__(self, target_sample_rate: int = SAMPLE_RATE) -> None:
        self.target_sample_rate = target_sample_rate
        self.source_sample_rate: int | None = None
        self._resampler = None

    def process(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        is_final: bool,
    ) -> tuple[np.ndarray, int]:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")

        if self.source_sample_rate is None:
            self.source_sample_rate = sample_rate
            if sample_rate != self.target_sample_rate:
                self._resampler = soxr.ResampleStream(
                    in_rate=sample_rate,
                    out_rate=self.target_sample_rate,
                    num_channels=1,
                    dtype="float32",
                    quality="HQ",
                )
        elif sample_rate != self.source_sample_rate:
            raise ValueError("sample rate changed within stream")

        audio_chunk = np.asarray(audio_chunk)
        if self._resampler is None:
            return _to_int16(audio_chunk), sample_rate

        audio_f32 = audio_chunk.astype(np.float32, copy=False)
        resampled = self._resampler.resample_chunk(audio_f32, last=is_final)
        if resampled is None:
            resampled = np.empty(0, dtype=np.float32)
        return _to_int16(resampled), self.target_sample_rate


def resample_audio_stream(
    data_itr: Iterable[tuple[np.ndarray, int, bool]],
    target_sample_rate: int = SAMPLE_RATE,
) -> Generator[tuple[np.ndarray, int, bool], None, None]:
    resampler = AudioStreamResampler(target_sample_rate)

    for audio_chunk, sample_rate, is_final in data_itr:
        output, output_sample_rate = resampler.process(
            audio_chunk,
            sample_rate,
            is_final,
        )
        yield output, output_sample_rate, is_final


def _to_int16(audio_chunk: np.ndarray) -> np.ndarray:
    if audio_chunk.dtype == np.int16:
        return audio_chunk

    clipped = np.clip(
        np.rint(audio_chunk),
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max,
    )
    return clipped.astype(np.int16)
