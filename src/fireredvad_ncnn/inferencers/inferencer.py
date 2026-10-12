from __future__ import annotations

import logging
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Tuple

from scipy.signal import resample
import numpy as np

from fireredvad_ncnn.configs.stream_vad_config import FireRedStreamVadConfig
from fireredvad_ncnn.models.audio_feat import AudioFeat
from fireredvad_ncnn.models.constants import (
    FRAME_LENGTH_SAMPLE,
    FRAME_PER_SECONDS,
    FRAME_SHIFT_SAMPLE,
    SAMPLE_RATE,
)
from fireredvad_ncnn.models.stream_vad_postprocessor import StreamVadPostprocessor
from fireredvad_ncnn.inferencers.model_runtime import VadModelRuntime
from fireredvad_ncnn.utils.audio_stream import AudioStreamResampler

logger = logging.getLogger(__name__)


@dataclass
class _StreamState:
    postprocessor: StreamVadPostprocessor
    model_caches: object = None
    audio_buffer: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int16)
    )
    stream_fbank: object = None
    stream_started: bool = False
    resampler: AudioStreamResampler = field(default_factory=AudioStreamResampler)
    input_samples: int = 0
    processed_time: float = 0.0


@dataclass
class PreparedStreamRequest:
    features: list[np.ndarray]
    input_duration: float
    is_final: bool


class Inferencer:
    def __init__(self, model_dir: str | Path, ncpu: int = 4,
                 stream_chunk_frame: int = 10,
                 stream_vad_config: FireRedStreamVadConfig | None = None):
        self.config = replace(stream_vad_config or FireRedStreamVadConfig(), use_gpu=False)
        if stream_chunk_frame <= 0:
            raise ValueError("stream_chunk_frame must be positive")
        self.audio_feat = AudioFeat(str(Path(model_dir) / "cmvn.ark"))
        self.stream_chunk_frame = stream_chunk_frame
        self.chunk_length = FRAME_LENGTH_SAMPLE + (stream_chunk_frame - 1) * FRAME_SHIFT_SAMPLE
        self.chunk_shift = stream_chunk_frame * FRAME_SHIFT_SAMPLE
        self.runtime = VadModelRuntime(model_dir, num_threads=ncpu)
        if self.runtime.chunk_frames != stream_chunk_frame:
            self.runtime.close()
            raise ValueError("stream_chunk_frame must match the exported model")

    def reset(self) -> None:
        self.audio_feat.reset()

    def close(self) -> None:
        self.runtime.close()

    def _new_postprocessor(
        self,
        config: FireRedStreamVadConfig | None = None,
    ) -> StreamVadPostprocessor:
        config = config or self.config
        return StreamVadPostprocessor(
            config.smooth_window_size,
            config.speech_threshold,
            config.pad_start_frame,
            config.min_speech_frame,
            config.max_speech_frame,
            config.min_silence_frame,
        )

    def _new_stream_state(
        self,
        postprocessing_config: FireRedStreamVadConfig | None = None,
    ) -> _StreamState:
        return _StreamState(
            postprocessor=self._new_postprocessor(postprocessing_config),
            stream_fbank=self.audio_feat.new_stream(),
        )

    def new_stream_state(
        self,
        postprocessing_config: FireRedStreamVadConfig | None = None,
    ) -> _StreamState:
        return self._new_stream_state(postprocessing_config)

    def _extract_feats(self, audio):
        return self.audio_feat.extract(audio)

    def _forward_stream(self, state: _StreamState, feats: np.ndarray):
        return self.runtime.forward_chunk(feats, caches=state.model_caches)

    def _forward_detect_eager(self, feats: np.ndarray):
        outputs = []
        for start in range(0, len(feats), self.config.chunk_max_frame):
            probs, _ = self.runtime.forward_chunk(feats[start:start + self.config.chunk_max_frame])
            outputs.append(probs)
        return np.concatenate(outputs, axis=1)

    @classmethod
    def results_to_timestamps(cls, results):
        results = sorted(results, key=lambda r: r.frame_idx)
        frame_timestamps = []
        start, end = -1, -1
        for result in results:
            if result.is_speech_start:
                if start != -1:
                    logger.warning("start should be -1")
                start = max(0, result.speech_start_frame - 1)
                end = -1
            elif result.is_speech_end:
                assert end == -1
                end = max(0, result.speech_end_frame - 1)
                frame_timestamps.append((start, end))
                start, end = -1, -1
        if start != -1 and results:
            assert end == -1
            end = results[-1].frame_idx - 1
            frame_timestamps.append((start, end))
        return [(s / FRAME_PER_SECONDS, e / FRAME_PER_SECONDS) for s, e in frame_timestamps]

    def infer(self, audio: np.ndarray, sr: int):
        feats = self.prepare_detect(audio, sr)
        if feats is None:
            return []
        return self.postprocess_detect(self._forward_detect_eager(feats))

    def prepare_detect(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        audio_np = np.asarray(audio)
        if audio_np.size == 0:
            return None
        if sr <= 0:
            raise ValueError("sample_rate must be a positive integer")

        if sr != SAMPLE_RATE:
            audio_np = resample(audio_np.astype(np.float32, copy=False),
                         int(np.ceil(len(audio_np) * SAMPLE_RATE / sr)))
            audio_np = self._to_int16(audio_np)
            sr = SAMPLE_RATE
        elif audio_np.dtype != np.int16:
            audio_np = self._to_int16(audio_np)

        feats, _ = self._extract_feats((audio_np, sr))
        if feats.shape[0] == 0:
            return None
        return feats

    def postprocess_detect(
        self,
        probs: np.ndarray,
        postprocessing_config: FireRedStreamVadConfig | None = None,
    ):
        raw_probs = probs.squeeze().tolist()
        if isinstance(raw_probs, float):
            raw_probs = [raw_probs]

        postprocessor = self._new_postprocessor(postprocessing_config)
        frame_results = []
        for raw_prob in raw_probs:
            frame_results.append(postprocessor.process_one_frame(float(raw_prob)))

        return self.results_to_timestamps(frame_results)

    def prepare_stream_request(
        self,
        state: _StreamState,
        audio_chunk: np.ndarray,
        sample_rate: int,
        is_final: bool,
    ) -> PreparedStreamRequest:
        effective_sample_rate = self._accept_stream_audio(
            state, audio_chunk, sample_rate, is_final
        )

        features = []
        while state.audio_buffer.size >= (
            self.chunk_shift if state.stream_started else self.chunk_length
        ):
            consume = self.chunk_shift if state.stream_started else self.chunk_length
            features.append(
                self._prepare_stream_features(
                    state,
                    state.audio_buffer[:consume],
                    effective_sample_rate,
                )
            )
            state.audio_buffer = state.audio_buffer[consume:]
            state.stream_started = True

        if is_final:
            feats = self._prepare_stream_features(
                state,
                state.audio_buffer,
                effective_sample_rate,
                is_final=True,
            )
            if feats.shape[0] > 0:
                features.append(feats)
            state.audio_buffer = np.empty(0, dtype=np.int16)

        return PreparedStreamRequest(
            features=features,
            input_duration=self._input_duration(state),
            is_final=is_final,
        )

    def _accept_stream_audio(
        self,
        state: _StreamState,
        audio_chunk: np.ndarray,
        sample_rate: int,
        is_final: bool,
    ) -> int:
        audio_chunk = np.asarray(audio_chunk)
        resampled, effective_sample_rate = state.resampler.process(
            audio_chunk, sample_rate, is_final
        )
        state.input_samples += audio_chunk.size
        if resampled.size:
            state.audio_buffer = np.concatenate((state.audio_buffer, resampled))
        return effective_sample_rate

    @staticmethod
    def _input_duration(state: _StreamState) -> float:
        sample_rate = state.resampler.source_sample_rate
        if sample_rate is None:
            return 0.0
        # 原始样本数决定 EOF；重采样长度舍入不能延长最终水位。
        return (state.input_samples * 1_000_000_000 // sample_rate) / 1_000_000_000

    def apply_stream_probabilities(
        self,
        state: _StreamState,
        probs: np.ndarray,
        new_caches,
        *,
        commit_model_cache: bool = True,
    ) -> list[tuple[float, float, float]]:
        if commit_model_cache:
            state.model_caches = new_caches
        raw_probs = probs.squeeze().tolist()
        if isinstance(raw_probs, float):
            raw_probs = [raw_probs]
        frame_results = [
            state.postprocessor.process_one_frame(float(raw_prob))
            for raw_prob in raw_probs
        ]
        return list(self._frame_results_to_events(frame_results, state.processed_time))

    def finish_stream_request(
        self,
        state: _StreamState,
        prepared: PreparedStreamRequest,
    ) -> list[tuple[float, float, float]]:
        events = []
        if prepared.is_final:
            if state.postprocessor.last_speech_start_frame >= 0:
                events.append(
                    (
                        -1.0,
                        prepared.input_duration,
                        state.processed_time,
                    )
                )
            state.audio_buffer = np.empty(0, dtype=np.int16)
            state.model_caches = None
            state.stream_fbank.reset()
            state.stream_started = False
            confirmed = prepared.input_duration
        else:
            confirmed = min(
                state.postprocessor.confirmed_frame_count() / FRAME_PER_SECONDS,
                prepared.input_duration,
            )
        if confirmed < state.processed_time:
            raise RuntimeError("VAD confirmation watermark moved backwards")
        if confirmed > state.processed_time or prepared.is_final:
            state.processed_time = confirmed
            events.append((-1.0, -1.0, confirmed))
        return events

    def infer_stream(
        self,
        data_itr: Iterable[Tuple[np.ndarray, int, bool]],
    ) -> Generator[Tuple[float, float, float], None, None]:
        state = self._new_stream_state()
        try:
            for audio_chunk, sample_rate, is_final in data_itr:
                sample_rate = self._accept_stream_audio(
                    state, audio_chunk, sample_rate, is_final
                )
                while state.audio_buffer.size >= (
                    self.chunk_shift if state.stream_started else self.chunk_length
                ):
                    consume = self.chunk_shift if state.stream_started else self.chunk_length
                    frame_results = self._infer_chunk(
                        state, state.audio_buffer[:consume], sample_rate
                    )
                    yield from self._frame_results_to_events(
                        frame_results, state.processed_time
                    )
                    state.audio_buffer = state.audio_buffer[consume:]
                    state.stream_started = True
                if is_final:
                    frame_results = self._infer_chunk(
                        state, state.audio_buffer, sample_rate, is_final=True
                    )
                    yield from self._frame_results_to_events(
                        frame_results, state.processed_time
                    )
                yield from self.finish_stream_request(
                    state,
                    PreparedStreamRequest([], self._input_duration(state), is_final),
                )
        except Exception:
            logger.exception("Error in infer_stream method")
            raise

    def _infer_chunk(
        self,
        state: _StreamState,
        audio_chunk: np.ndarray,
        sample_rate: int,
        is_final: bool = False,
    ):
        if len(audio_chunk) == 0 and not is_final:
            return []

        feats = self._prepare_stream_features(
            state,
            audio_chunk,
            sample_rate,
            is_final=is_final,
        )
        if feats.shape[0] == 0:
            return []
        probs, state.model_caches = self._forward_stream(state, feats)
        raw_probs = probs.squeeze().tolist()
        if isinstance(raw_probs, float):
            raw_probs = [raw_probs]

        frame_results = []
        for raw_prob in raw_probs:
            frame_results.append(state.postprocessor.process_one_frame(float(raw_prob)))
        return frame_results

    def _prepare_stream_features(
        self,
        state: _StreamState,
        audio_chunk: np.ndarray,
        sample_rate: int,
        is_final: bool = False,
    ) -> np.ndarray:
        fbank = state.stream_fbank((sample_rate, audio_chunk), is_final=is_final)
        if self.audio_feat.cmvn is not None:
            fbank = self.audio_feat.cmvn(fbank)
        return np.asarray(fbank, dtype=np.float32)

    @staticmethod
    def _to_int16(audio_chunk: np.ndarray) -> np.ndarray:
        if audio_chunk.dtype == np.int16:
            return audio_chunk

        clipped = np.clip(
            np.rint(audio_chunk),
            np.iinfo(np.int16).min,
            np.iinfo(np.int16).max,
        )
        return clipped.astype(np.int16)

    def _frame_results_to_events(
        self,
        frame_results,
        processed_time: float,
    ):
        for frame_result in frame_results:
            if frame_result.is_speech_start:
                yield (
                    self._frame_to_seconds(frame_result.speech_start_frame),
                    -1.0,
                    processed_time,
                )
            if frame_result.is_speech_end:
                yield (
                    -1.0,
                    self._frame_to_seconds(frame_result.speech_end_frame),
                    processed_time,
                )

    @staticmethod
    def _frame_to_seconds(frame_idx: int) -> float:
        return max(0, frame_idx - 1) / FRAME_PER_SECONDS
