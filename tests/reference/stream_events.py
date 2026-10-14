"""Historical publisher oracle; production publishes from the native session."""
from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass
class StreamEventPublisher:
    """在输出模式过滤之后发布水位，避免确认尚未对外公布的有效音。"""

    endpoint_only: bool = False
    pending_start: float = -1.0
    processed_time: float = 0.0

    def publish(
        self, events: Iterable[tuple[float, float, float]]
    ) -> Iterator[tuple[float, float, float]]:
        for start, end, processed in events:
            if not self.endpoint_only:
                self.processed_time = processed
                yield start, end, processed
                continue
            if start >= 0:
                self.pending_start = start
            if end >= 0 and self.pending_start >= 0:
                yield self.pending_start, end, self.processed_time
                self.pending_start = -1.0
            if self.pending_start >= 0:
                processed = min(processed, self.pending_start)
            if processed > self.processed_time or (start < 0 and end < 0):
                self.processed_time = processed
                yield -1.0, -1.0, processed
