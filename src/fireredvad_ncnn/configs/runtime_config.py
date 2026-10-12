from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    dtype: str = "fp32"
    runner_name: str = "ncnn"
    num_threads: int = 4
    stream_chunk_frame: int = 10

    def __post_init__(self):
        if (self.device, self.dtype, self.runner_name) != ("cpu", "fp32", "ncnn"):
            raise ValueError("Only CPU FP32 NCNN is supported")
        if self.num_threads < 1 or self.stream_chunk_frame < 1:
            raise ValueError("num_threads and stream_chunk_frame must be positive")
