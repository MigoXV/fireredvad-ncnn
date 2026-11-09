"""Errors exposed by the in-process asynchronous engine adapter."""

class EngineError(RuntimeError):
    pass


class InvalidArgumentError(EngineError):
    pass


class ResourceExhaustedError(EngineError):
    pass


class DeadlineExceededError(EngineError):
    pass


class FailedPreconditionError(EngineError):
    pass


class UnavailableError(EngineError):
    pass


