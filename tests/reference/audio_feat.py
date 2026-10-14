"""Pre-migration Fbank/CMVN oracle."""
# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Wenpeng Li, Kai Huang, Kun Liu)

import math
import os

import soundfile as sf
import kaldiio
import kaldi_native_fbank as knf
import numpy as np


if not hasattr(knf.OnlineFbank, "accept_waveform_and_get_frames"):
    raise ImportError(
        "kaldi-native-fbank-buffer with "
        "OnlineFbank.accept_waveform_and_get_frames is required"
    )


class AudioFeat:
    def __init__(self, kaldi_cmvn_file):
        self.cmvn = CMVN(kaldi_cmvn_file) if kaldi_cmvn_file != "" else None
        self.fbank_kwargs = dict(
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0,
        )

    def reset(self):
        pass

    def extract(self, audio):
        wav_np, sample_rate = self._load_audio(audio)
        dur = wav_np.shape[0] / sample_rate
        fbank = self.new_stream()
        feat = fbank((sample_rate, wav_np), is_final=True)
        if self.cmvn is not None:
            feat = self.cmvn(feat)
        feat = np.asarray(feat, dtype=np.float32)
        return feat, dur

    def new_stream(self):
        return KaldifeatFbank(**self.fbank_kwargs)

    @staticmethod
    def _load_audio(audio):
        if isinstance(audio, str):
            wav_np, sample_rate = sf.read(audio, dtype="int16")
        elif isinstance(audio, (list, tuple)):
            wav_np, sample_rate = audio
        else:
            wav_np = audio
            sample_rate = 16000
        assert sample_rate == 16000
        return wav_np, sample_rate


class CMVN:
    def __init__(self, kaldi_cmvn_file):
        self.dim, self.means, self.inverse_std_variances =             self.read_kaldi_cmvn(kaldi_cmvn_file)

    def __call__(self, x, is_train=False):
        assert x.shape[-1] == self.dim, "CMVN dim mismatch"
        out = x - self.means
        out = out * self.inverse_std_variances
        return out

    def read_kaldi_cmvn(self, kaldi_cmvn_file):
        assert os.path.exists(kaldi_cmvn_file)
        stats = kaldiio.load_mat(kaldi_cmvn_file)
        assert stats.shape[0] == 2
        dim = stats.shape[-1] - 1
        count = stats[0, dim]
        assert count >= 1
        floor = 1e-20
        means = []
        inverse_std_variances = []
        for d in range(dim):
            mean = stats[0, d] / count
            means.append(mean.item())
            variance = (stats[1, d] / count) - mean * mean
            if variance < floor:
                variance = floor
            istd = 1.0 / math.sqrt(variance)
            inverse_std_variances.append(istd)
        return dim, np.array(means), np.array(inverse_std_variances)


class KaldifeatFbank:
    def __init__(self, num_mel_bins=80, frame_length=25, frame_shift=10,
                 dither=0):
        self.dither = dither
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.frame_length_ms = frame_length
        opts.frame_opts.frame_shift_ms = frame_shift
        opts.frame_opts.dither = dither
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = num_mel_bins
        opts.mel_opts.debug_mel = False
        self.opts = opts
        self.reset()

    def reset(self):
        self.online_fbank = knf.OnlineFbank(self.opts)

    def __call__(self, wav, is_train=False, is_final=False):
        sample_rate, wav_np = self._load_wav(wav)
        wav_np = np.asarray(wav_np)
        assert len(wav_np.shape) == 1

        dither = self.dither if is_train else 0.0
        if self.opts.frame_opts.dither != dither:
            self.opts.frame_opts.dither = dither
            self.reset()

        if wav_np.dtype not in (np.dtype(np.int16), np.dtype(np.float32)):
            wav_np = wav_np.astype(np.float32)
        if not wav_np.flags.c_contiguous:
            wav_np = np.ascontiguousarray(wav_np)

        return self.online_fbank.accept_waveform_and_get_frames(
            sample_rate,
            wav_np,
            is_final=is_final,
        )

    @staticmethod
    def _load_wav(wav):
        if isinstance(wav, str):
            wav_np, sample_rate = sf.read(wav, dtype="int16")
        elif isinstance(wav, (tuple, list)) and len(wav) == 2:
            sample_rate, wav_np = wav
        else:
            raise ValueError("wav must be a path or a (sample_rate, wav_np) pair")
        return sample_rate, wav_np
