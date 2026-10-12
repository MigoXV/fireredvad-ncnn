from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    max_batch_size: int = 32
    max_batch_wait_ms: float = 5.0
    max_sessions: int = 128
    session_queue_capacity: int = 2
    ingress_capacity: int = 2048
    queue_timeout_ms: int = 5000
    detect_queue_timeout_ms: int = 30000
    max_inflight_mb: int = 256
    metrics_interval_s: float = 10.0
    startup_timeout_s: float = 180.0
    shutdown_grace_s: float = 10.0

    @property
    def max_inflight_bytes(self) -> int:
        return self.max_inflight_mb * 1024 * 1024

    @property
    def max_batch_wait_s(self) -> float:
        return self.max_batch_wait_ms / 1000.0

    @property
    def queue_timeout_s(self) -> float:
        return self.queue_timeout_ms / 1000.0

    @property
    def detect_queue_timeout_s(self) -> float:
        return self.detect_queue_timeout_ms / 1000.0

    def __post_init__(self) -> None:
        positive_values = {
            "max_batch_size": self.max_batch_size,
            "max_sessions": self.max_sessions,
            "session_queue_capacity": self.session_queue_capacity,
            "ingress_capacity": self.ingress_capacity,
            "queue_timeout_ms": self.queue_timeout_ms,
            "detect_queue_timeout_ms": self.detect_queue_timeout_ms,
            "max_inflight_mb": self.max_inflight_mb,
            "metrics_interval_s": self.metrics_interval_s,
            "startup_timeout_s": self.startup_timeout_s,
            "shutdown_grace_s": self.shutdown_grace_s,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_batch_wait_ms < 0:
            raise ValueError("max_batch_wait_ms must be non-negative")
