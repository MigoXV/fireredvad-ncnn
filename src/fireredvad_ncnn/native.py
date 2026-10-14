"""FireRedVAD engine C ABI binding; no NCNN or DSP Python imports."""
from __future__ import annotations

import ctypes as ct
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fireredvad_ncnn.configs.engine_config import EngineConfig
from fireredvad_ncnn.configs.stream_vad_config import FireRedStreamVadConfig


class NativeError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class PostConfig(ct.Structure):
    _fields_ = [(n, ct.c_uint32) for n in (
        "struct_size", "abi_version", "smooth_window_size", "pad_start_frame",
        "min_speech_frame", "max_speech_frame", "min_silence_frame", "chunk_max_frame",
    )] + [("speech_threshold", ct.c_double)]


class NativeConfig(ct.Structure):
    _fields_ = [(n, ct.c_uint32) for n in (
        "struct_size", "abi_version", "cpu_budget", "worker_count", "threads_per_worker",
        "chunk_frames", "max_sessions", "session_queue_capacity", "request_capacity",
    )] + [("max_inflight_bytes", ct.c_uint64)]


class Event(ct.Structure):
    _fields_ = [(n, ct.c_double) for n in ("start", "end", "processed_time")]


class Result(ct.Structure):
    _fields_ = [("request_id", ct.c_uint64), ("status", ct.c_int32),
                ("reserved", ct.c_uint32), ("event_count", ct.c_size_t),
                ("events", ct.POINTER(Event)), ("message", ct.c_char_p)]


class Metrics(ct.Structure):
    _fields_ = [(n, ct.c_uint64) for n in (
        "submitted", "completed", "rejected", "canceled", "expired", "sessions",
        "pending", "inflight_bytes", "queue_high_water", "model_blocks",
        "preprocess_ns", "inference_ns", "postprocess_ns",
    )]


@dataclass(frozen=True)
class Completion:
    request_id: int
    status: int
    events: list[tuple[float, float, float]]
    message: str


def _load_library():
    explicit = os.environ.get("FIREREDVAD_LIBRARY")
    candidates = [Path(explicit)] if explicit else [
        Path(__file__).parent / "lib/libfireredvad.so",
        Path(__file__).resolve().parents[2] / "build/x86_64/libfireredvad.so",
        Path(__file__).resolve().parents[2] / "build/arm64/libfireredvad.so",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise RuntimeError("FireRedVAD native library missing; build with CMake or set FIREREDVAD_LIBRARY")
    lib = ct.CDLL(str(path))  # CDLL releases the GIL during C calls.
    signatures = {
        "frv_abi_version": ([], ct.c_uint32),
        "frv_build_info": ([], ct.c_char_p), "frv_last_error": ([], ct.c_char_p),
        "frv_engine_config_init": ([ct.POINTER(NativeConfig)], None),
        "frv_post_config_init": ([ct.POINTER(PostConfig)], None),
        "frv_engine_create": ([ct.c_char_p, ct.POINTER(NativeConfig), ct.POINTER(ct.c_void_p)], ct.c_int32),
        "frv_engine_destroy": ([ct.c_void_p], None),
        "frv_session_open": ([ct.c_void_p, ct.POINTER(PostConfig), ct.c_int32, ct.POINTER(ct.c_uint64)], ct.c_int32),
        "frv_session_close": ([ct.c_void_p, ct.c_uint64], ct.c_int32),
        "frv_submit": ([ct.c_void_p, ct.c_uint64, ct.c_uint64, ct.c_void_p, ct.c_size_t,
                        ct.c_int32, ct.c_uint32, ct.c_int32, ct.POINTER(PostConfig),
                        ct.c_uint32, ct.POINTER(ct.c_uint64)], ct.c_int32),
        "frv_poll": ([ct.c_void_p, ct.c_uint64, ct.c_uint32, ct.POINTER(ct.POINTER(Result))], ct.c_int32),
        "frv_cancel": ([ct.c_void_p, ct.c_uint64], ct.c_int32),
        "frv_result_free": ([ct.POINTER(Result)], None),
        "frv_get_metrics": ([ct.c_void_p, ct.POINTER(Metrics)], ct.c_int32),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    if lib.frv_abi_version() != 1:
        raise RuntimeError("unsupported FireRedVAD engine ABI")
    return lib


def _uint(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not (0 if zero else 1) <= value <= 0xFFFFFFFF:
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} uint32")
    return int(value)


class NativeEngine:
    def __init__(self, model_dir, *, cpu_budget=4, worker_count=1, threads_per_worker=4,
                 chunk_frames=10, engine_config: EngineConfig | None = None):
        self._lib = _load_library()
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        config = NativeConfig()
        self._lib.frv_engine_config_init(ct.byref(config))
        for name, value in dict(cpu_budget=cpu_budget, worker_count=worker_count,
                                threads_per_worker=threads_per_worker, chunk_frames=chunk_frames).items():
            setattr(config, name, _uint(value, name))
        if engine_config is not None:
            config.max_sessions = engine_config.max_sessions
            config.session_queue_capacity = engine_config.session_queue_capacity
            config.request_capacity = engine_config.ingress_capacity
            config.max_inflight_bytes = engine_config.max_inflight_bytes
        self._check(self._lib.frv_engine_create(os.fsencode(model_dir), ct.byref(config), ct.byref(self._handle)))
        self.build_info = self._lib.frv_build_info().decode()

    def _check(self, status):
        if status:
            raise NativeError(status, self._lib.frv_last_error().decode("utf-8", "replace"))

    def _open(self):
        if not self._handle.value:
            raise RuntimeError("Engine is closed")

    @property
    def closed(self):
        with self._lock:
            return not bool(self._handle.value)

    def _post_config(self, config):
        config = config or FireRedStreamVadConfig()
        native = PostConfig()
        self._lib.frv_post_config_init(ct.byref(native))
        for name, _ in PostConfig._fields_[2:]:
            value = getattr(config, name)
            if name != "speech_threshold":
                value = _uint(value, name, zero=name == "pad_start_frame")
            setattr(native, name, value)
        return native

    def session_open(self, config=None, endpoint_only=False):
        post, session = self._post_config(config), ct.c_uint64()
        with self._lock:
            self._open()
            self._check(self._lib.frv_session_open(self._handle, ct.byref(post), endpoint_only, ct.byref(session)))
        return session.value

    def session_close(self, session):
        with self._lock:
            self._open()
            self._check(self._lib.frv_session_close(self._handle, session))

    def submit(self, audio, sample_rate, *, session=0, sequence=0, is_final=False,
               postprocessing_config=None, timeout_ms=0):
        rate = _uint(sample_rate, "sample_rate")
        timeout = _uint(timeout_ms, "timeout_ms", zero=True)
        audio = np.asarray(audio)
        if audio.ndim != 1:
            raise ValueError("audio must be mono [samples]")
        if audio.dtype == np.int16:
            audio, format_id = np.ascontiguousarray(audio), 1
        else:
            audio, format_id = np.ascontiguousarray(audio, dtype=np.float32), 2
        post = self._post_config(postprocessing_config) if not session else None
        request = ct.c_uint64()
        with self._lock:
            self._open()
            self._check(self._lib.frv_submit(self._handle, session, sequence, audio.ctypes.data,
                audio.size, format_id, rate, is_final, ct.byref(post) if post else None, timeout, ct.byref(request)))
        return request.value

    def poll(self, request_id=0, wait_ms=0):
        wait_ms = _uint(wait_ms, "wait_ms", zero=True)
        result = ct.POINTER(Result)()
        with self._lock:
            self._open()
            status = self._lib.frv_poll(self._handle, request_id, wait_ms, ct.byref(result))
            if status == 8:
                return None
            self._check(status)
            try:
                r = result.contents
                return Completion(r.request_id, r.status,
                    [(e.start, e.end, e.processed_time) for e in r.events[:r.event_count]],
                    r.message.decode("utf-8", "replace"))
            finally:
                self._lib.frv_result_free(result)

    def wait(self, request_id):
        while True:
            result = self.poll(request_id, wait_ms=10)
            if result is not None:
                if result.status:
                    raise NativeError(result.status, result.message)
                return result.events

    def cancel(self, request_id):
        with self._lock:
            self._open()
            self._check(self._lib.frv_cancel(self._handle, request_id))

    def metrics(self):
        metrics = Metrics()
        with self._lock:
            self._open()
            self._check(self._lib.frv_get_metrics(self._handle, ct.byref(metrics)))
        return {name: getattr(metrics, name) for name, _ in Metrics._fields_}

    def close(self):
        with self._lock:
            if self._handle.value:
                self._lib.frv_engine_destroy(self._handle)
                self._handle = ct.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
