from __future__ import annotations

from importlib import metadata

import kaldi_native_fbank as knf
import numpy as np
import pytest

from fireredvad_ncnn.models.audio_feat import KaldifeatFbank


SAMPLE_RATE = 16000


def _waveform(num_samples: int = SAMPLE_RATE * 2) -> np.ndarray:
    indices = np.arange(num_samples, dtype=np.int64)
    return ((indices * 997 + 13) % 60001 - 30000).astype(np.int16)


def _legacy_features(waveform: np.ndarray) -> np.ndarray:
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = SAMPLE_RATE
    opts.frame_opts.frame_length_ms = 25
    opts.frame_opts.frame_shift_ms = 10
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = 80
    opts.mel_opts.debug_mel = False
    stream = knf.OnlineFbank(opts)
    stream.accept_waveform(SAMPLE_RATE, waveform.tolist())
    stream.input_finished()
    return np.vstack(
        [stream.get_frame(index) for index in range(stream.num_frames_ready)]
    )


def test_buffer_distribution_and_api_are_installed() -> None:
    assert metadata.version("kaldi-native-fbank-buffer") == knf.__version__
    with pytest.raises(metadata.PackageNotFoundError):
        metadata.version("kaldi-native-fbank")
    assert hasattr(knf.OnlineFbank, "accept_waveform_and_get_frames")


def test_buffer_fbank_matches_legacy_one_shot() -> None:
    waveform = _waveform()

    actual = KaldifeatFbank()((SAMPLE_RATE, waveform), is_final=True)
    expected = _legacy_features(waveform)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous


def test_buffer_fbank_matches_legacy_when_chunked() -> None:
    waveform = _waveform(SAMPLE_RATE * 3 + 137)
    stream = KaldifeatFbank()
    parts = []
    chunk_samples = 1600
    for offset in range(0, waveform.size, chunk_samples):
        chunk = waveform[offset : offset + chunk_samples]
        parts.append(
            stream(
                (SAMPLE_RATE, chunk),
                is_final=offset + chunk_samples >= waveform.size,
            )
        )
    actual = np.concatenate(parts, axis=0)

    np.testing.assert_array_equal(actual, _legacy_features(waveform))


def test_buffer_fbank_normalizes_dtype_and_noncontiguous_input() -> None:
    waveform = _waveform().astype(np.float64)[::2]
    expected_input = waveform.astype(np.float32)

    actual = KaldifeatFbank()((SAMPLE_RATE, waveform), is_final=True)

    np.testing.assert_array_equal(actual, _legacy_features(expected_input))


def test_buffer_fbank_reset_allows_reuse_after_final() -> None:
    waveform = _waveform(SAMPLE_RATE)
    stream = KaldifeatFbank()
    first = stream((SAMPLE_RATE, waveform), is_final=True)

    stream.reset()
    second = stream((SAMPLE_RATE, waveform), is_final=True)

    np.testing.assert_array_equal(first, second)
