from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    dtype: str = "fp32"
    runner_name: str = "ncnn"
    num_threads: int = 4
    stream_chunk_frame: int = 10
    worker_count: int = 1
    threads_per_worker: int | None = None

    @property
    def effective_threads_per_worker(self) -> int:
        return self.threads_per_worker if self.threads_per_worker is not None else self.num_threads // self.worker_count

    def __post_init__(self):
        if (self.device, self.dtype, self.runner_name) != ("cpu", "fp32", "ncnn"):
            raise ValueError("Only CPU FP32 NCNN is supported")
        if self.num_threads < 1 or self.stream_chunk_frame < 1:
            raise ValueError("num_threads and stream_chunk_frame must be positive")
        if self.worker_count < 1 or self.effective_threads_per_worker < 1:
            raise ValueError("worker_count and threads_per_worker must be positive")
        if self.worker_count * self.effective_threads_per_worker > self.num_threads:
            raise ValueError("worker_count * threads_per_worker must not exceed CPU budget (num_threads)")
