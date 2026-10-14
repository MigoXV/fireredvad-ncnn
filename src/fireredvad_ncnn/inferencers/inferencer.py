"""Public audio-to-events adapter for the native FireRedVAD engine."""
from __future__ import annotations

from dataclasses import replace

from fireredvad_ncnn.configs.stream_vad_config import FireRedStreamVadConfig
from fireredvad_ncnn.configs.runtime_config import RuntimeConfig
from fireredvad_ncnn.native import NativeEngine, NativeError


class Inferencer:
    def __init__(self, model_dir, ncpu=4, stream_chunk_frame=10,
                 stream_vad_config: FireRedStreamVadConfig | None = None, *,
                 worker_count=1, threads_per_worker=None):
        self.config = replace(stream_vad_config or FireRedStreamVadConfig(), use_gpu=False)
        runtime = RuntimeConfig(num_threads=ncpu, stream_chunk_frame=stream_chunk_frame,
                                worker_count=worker_count, threads_per_worker=threads_per_worker)
        self.engine = NativeEngine(model_dir, cpu_budget=ncpu, worker_count=worker_count,
            threads_per_worker=runtime.effective_threads_per_worker,
            chunk_frames=stream_chunk_frame)

    def infer(self, audio, sr):
        request = self.engine.submit(audio, sr, is_final=True, postprocessing_config=self.config)
        return [(start, end) for start, end, _ in self.engine.wait(request)]

    def infer_stream(self, data_itr):
        session = self.engine.session_open(self.config)
        try:
            for sequence, (audio, sr, final) in enumerate(data_itr, 1):
                request = self.engine.submit(audio, sr, session=session, sequence=sequence, is_final=final)
                yield from self.engine.wait(request)
                if final:
                    break
        finally:
            try:
                self.engine.session_close(session)
            except RuntimeError as exc:
                if not self.engine.closed and not (isinstance(exc, NativeError) and exc.status == 4):
                    raise

    def reset(self):
        """Calls own independent sessions; retained for public compatibility."""

    def close(self):
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
