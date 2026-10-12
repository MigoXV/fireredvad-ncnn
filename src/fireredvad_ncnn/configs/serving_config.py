from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ServingConfig:
    host: str = "0.0.0.0"
    port: int = 50051
    max_concurrent_rpcs: int = 256
    max_message_mb: int = 100

    @property
    def max_message_bytes(self) -> int:
        return self.max_message_mb * 1024 * 1024

    def __post_init__(self) -> None:
        if self.port <= 0:
            raise ValueError("port must be positive")
        if self.max_concurrent_rpcs <= 0:
            raise ValueError("max_concurrent_rpcs must be positive")
        if self.max_message_mb <= 0:
            raise ValueError("max_message_mb must be positive")
