from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections import deque
from collections.abc import AsyncIterator, Iterable
from decimal import Decimal
from typing import Any

import grpc
from google.protobuf import duration_pb2
from google.protobuf.json_format import MessageToDict

import fireredvad_ncnn.proto.ux_vad_pb2 as pb
import fireredvad_ncnn.proto.ux_vad_pb2_grpc as pb_grpc
from fireredvad_ncnn.engine.client import EngineClient
from fireredvad_ncnn.engine.protocol import (
    DeadlineExceededError,
    EngineError,
    FailedPreconditionError,
    InvalidArgumentError,
    ResourceExhaustedError,
    UnavailableError,
)

logger = logging.getLogger(__name__)


class UxVoiceActivityDetectorService(pb_grpc.UxVoiceActivityDetectorServicer):
    def __init__(
        self,
        engine: EngineClient,
        *,
        endpoint_only: bool = True,
        max_message_bytes: int = 100 * 1024 * 1024,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.endpoint_only = endpoint_only
        self.max_message_bytes = max_message_bytes

    async def Detect(
        self,
        request: pb.DetectRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.DetectResponse:
        logger.info("rpc_request method=Detect")
        try:
            sample_rate = self._validate_detection_config(request.config)
            postprocessing = self._parse_postprocessing(request.config)
            self._validate_audio_payload(request.audio, allow_empty=False)
            segments = await self.engine.detect(
                request.audio,
                sample_rate,
                postprocessing,
            )
            return pb.DetectResponse(
                results=list(build_results((float(start), float(end)) for start, end in segments))
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._abort(context, exc)
            raise AssertionError("context.abort must terminate the RPC")

    async def StreamingDetect(
        self,
        request_iterator: AsyncIterator[pb.StreamingDetectRequest],
        context: grpc.aio.ServicerContext,
    ):
        logger.info("rpc_request method=StreamingDetect")
        session_id = uuid.uuid4().hex
        opened = False
        sequence = 0
        try:
            first_request = await anext(request_iterator, None)
            if first_request is None:
                return
            if not first_request.HasField("config"):
                raise InvalidArgumentError(
                    "First StreamingDetectRequest must contain config."
                )
            sample_rate = self._validate_detection_config(first_request.config)
            postprocessing = self._parse_postprocessing(first_request.config)
            await self.engine.open_stream(
                session_id,
                self.endpoint_only,
                postprocessing,
            )
            opened = True

            pending: deque[asyncio.Task] = deque()
            capacity = self.engine.config.session_queue_capacity
            async for request in request_iterator:
                if request.HasField("config"):
                    logger.warning("StreamingDetect received additional config; ignoring")
                    continue
                self._validate_audio_payload(request.audio_content, allow_empty=True)
                sequence += 1
                pending.append(
                    asyncio.create_task(
                        self.engine.stream_chunk(
                            session_id,
                            sequence,
                            request.audio_content,
                            sample_rate,
                            False,
                        )
                    )
                )
                if len(pending) >= capacity:
                    events = await pending.popleft()
                    for response in _events_to_responses(events):
                        yield response

            sequence += 1
            pending.append(
                asyncio.create_task(
                    self.engine.stream_chunk(
                        session_id,
                        sequence,
                        b"",
                        sample_rate,
                        True,
                    )
                )
            )
            while pending:
                events = await pending.popleft()
                for response in _events_to_responses(events):
                    yield response
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._abort(context, exc)
        finally:
            if "pending" in locals():
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
            if opened:
                with contextlib.suppress(Exception):
                    await asyncio.shield(
                        self.engine.close_stream(session_id, sequence + 1)
                    )

    def _validate_detection_config(self, config: pb.DetectionConfig) -> int:
        if config.encoding != pb.DetectionConfig.AudioEncoding.LINEAR16:
            raise InvalidArgumentError("Only LINEAR16 encoding is supported.")
        if config.sample_rate_hertz <= 0:
            raise InvalidArgumentError(
                "sample_rate_hertz must be a positive integer."
            )
        return int(config.sample_rate_hertz)

    def _validate_audio_payload(self, audio: bytes, *, allow_empty: bool) -> None:
        if len(audio) > self.max_message_bytes:
            raise ResourceExhaustedError("audio payload exceeds max message size.")
        if not allow_empty and not audio:
            raise InvalidArgumentError("audio must not be empty.")
        if len(audio) % 2:
            raise InvalidArgumentError(
                "LINEAR16 audio payload must contain an even number of bytes."
            )

    def _parse_postprocessing(
        self,
        config: pb.DetectionConfig,
    ) -> dict[str, Any] | None:
        if not config.HasField("postprocessing"):
            return None
        options = config.postprocessing
        return {
            "processor": options.processor,
            "version": int(options.version),
            "parameters": MessageToDict(
                options.parameters,
                preserving_proto_field_name=True,
            ),
        }

    async def _abort(
        self,
        context: grpc.aio.ServicerContext,
        exc: Exception,
    ) -> None:
        if isinstance(exc, InvalidArgumentError):
            code = grpc.StatusCode.INVALID_ARGUMENT
        elif isinstance(exc, ResourceExhaustedError):
            code = grpc.StatusCode.RESOURCE_EXHAUSTED
        elif isinstance(exc, (DeadlineExceededError, asyncio.TimeoutError)):
            code = grpc.StatusCode.DEADLINE_EXCEEDED
        elif isinstance(exc, FailedPreconditionError):
            code = grpc.StatusCode.FAILED_PRECONDITION
        elif isinstance(exc, UnavailableError):
            code = grpc.StatusCode.UNAVAILABLE
        elif isinstance(exc, (ValueError, EngineError)):
            code = grpc.StatusCode.INTERNAL
        else:
            logger.exception("Unhandled VAD service error", exc_info=exc)
            code = grpc.StatusCode.INTERNAL
            exc = RuntimeError("Internal error")
        await context.abort(code, str(exc))


def _events_to_responses(events: Iterable[list[float]]):
    for start, end, processed_time in events:
        yield pb.StreamingDetectResponse(
            results=list(build_results([(float(start), float(end))])),
            processed_time=_sec(float(processed_time)),
        )


def _sec(sec: float) -> duration_pb2.Duration:
    seconds = int(sec)
    nanos = int((Decimal(str(sec)) - seconds) * 1_000_000_000)
    return duration_pb2.Duration(seconds=seconds, nanos=nanos)


def build_results(
    raw_results: Iterable[tuple[float, float]],
) -> Iterable[pb.DetectionResult]:
    for start_sec, end_sec in raw_results:
        yield pb.DetectionResult(
            start_time=_sec(start_sec),
            end_time=_sec(end_sec),
        )
