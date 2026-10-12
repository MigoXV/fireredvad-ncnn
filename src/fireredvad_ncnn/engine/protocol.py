from __future__ import annotations

from enum import Enum
from typing import Any

import msgpack


PROTOCOL_VERSION = 1


class Operation(str, Enum):
    HEALTH = "health"
    DETECT = "detect"
    STREAM_OPEN = "stream_open"
    STREAM_CHUNK = "stream_chunk"
    STREAM_CLOSE = "stream_close"
    CANCEL = "cancel"
    SHUTDOWN = "shutdown"


class Status(str, Enum):
    OK = "ok"
    INVALID_ARGUMENT = "invalid_argument"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    FAILED_PRECONDITION = "failed_precondition"
    UNAVAILABLE = "unavailable"
    INTERNAL = "internal"


class EngineError(RuntimeError):
    status = Status.INTERNAL


class InvalidArgumentError(EngineError):
    status = Status.INVALID_ARGUMENT


class ResourceExhaustedError(EngineError):
    status = Status.RESOURCE_EXHAUSTED


class DeadlineExceededError(EngineError):
    status = Status.DEADLINE_EXCEEDED


class FailedPreconditionError(EngineError):
    status = Status.FAILED_PRECONDITION


class UnavailableError(EngineError):
    status = Status.UNAVAILABLE


ERROR_BY_STATUS = {
    Status.INVALID_ARGUMENT: InvalidArgumentError,
    Status.RESOURCE_EXHAUSTED: ResourceExhaustedError,
    Status.DEADLINE_EXCEEDED: DeadlineExceededError,
    Status.FAILED_PRECONDITION: FailedPreconditionError,
    Status.UNAVAILABLE: UnavailableError,
    Status.INTERNAL: EngineError,
}


def pack_metadata(metadata: dict[str, Any]) -> bytes:
    return msgpack.packb(metadata, use_bin_type=True)


def unpack_metadata(data: bytes) -> dict[str, Any]:
    try:
        metadata = msgpack.unpackb(data, raw=False, strict_map_key=False)
    except (ValueError, msgpack.UnpackException) as exc:
        raise InvalidArgumentError(f"invalid engine metadata: {exc}") from exc
    if not isinstance(metadata, dict):
        raise InvalidArgumentError("engine metadata must be a map")
    return metadata


def validate_request(
    metadata: dict[str, Any],
    payload: bytes,
    max_payload_bytes: int,
) -> Operation:
    if metadata.get("version") != PROTOCOL_VERSION:
        raise FailedPreconditionError("unsupported engine protocol version")
    try:
        operation = Operation(metadata.get("op"))
    except ValueError as exc:
        raise InvalidArgumentError("unknown engine operation") from exc
    for field in (
        "request_id",
        "session_id",
        "sequence",
        "generation",
        "deadline_ns",
        "payload_length",
    ):
        if field not in metadata:
            raise InvalidArgumentError(f"missing engine field: {field}")
    if not isinstance(metadata["request_id"], str) or not metadata["request_id"]:
        raise InvalidArgumentError("request_id must be a non-empty string")
    if not isinstance(metadata["session_id"], str):
        raise InvalidArgumentError("session_id must be a string")
    if not isinstance(metadata["sequence"], int) or metadata["sequence"] < 0:
        raise InvalidArgumentError("sequence must be a non-negative integer")
    if metadata["payload_length"] != len(payload):
        raise InvalidArgumentError("payload_length does not match payload")
    if len(payload) > max_payload_bytes:
        raise ResourceExhaustedError("engine payload exceeds configured maximum")
    return operation


def build_response(
    request_id: str,
    generation: str,
    *,
    status: Status = Status.OK,
    **fields: Any,
) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "request_id": request_id,
        "generation": generation,
        "status": status.value,
        **fields,
    }


def raise_for_response(metadata: dict[str, Any]) -> None:
    try:
        status = Status(metadata.get("status"))
    except ValueError as exc:
        raise EngineError("engine returned an unknown status") from exc
    if status is Status.OK:
        return
    error_type = ERROR_BY_STATUS.get(status, EngineError)
    raise error_type(str(metadata.get("message", status.value)))
