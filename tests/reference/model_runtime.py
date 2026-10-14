"""Pre-migration NCNN Python runtime oracle; test-only dependency."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import ncnn
import numpy as np


class VadModelRuntime:
    def __init__(self, model_dir: str | Path, num_threads: int = 4):
        if num_threads < 1:
            raise ValueError("num_threads must be positive")
        root = Path(model_dir)
        meta = json.loads((root / "metadata.json").read_text())
        if meta["format_version"] != 1 or meta["dtype"] != "float32":
            raise ValueError("Unsupported model format or precision")
        if meta["cache_shape"] != [1, 128, 19] or meta["cache_count"] != 8 or meta["feature_dim"] != 80:
            raise ValueError("Unsupported model dimensions")
        self.chunk_frames = int(meta["chunk_frames"])
        if not 1 <= self.chunk_frames <= 19:
            raise ValueError("chunk_frames must be between 1 and 19")
        for name in ("model.ncnn.param", "model.ncnn.bin", "cmvn.ark"):
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != meta["files"][name]:
                raise ValueError(f"Model checksum mismatch: {name}")
        self._lock = threading.Lock()
        self._closed = False
        self.net = ncnn.Net()
        self.net.opt.num_threads = num_threads
        self.net.opt.use_vulkan_compute = False
        self.net.opt.use_fp16_packed = False
        self.net.opt.use_fp16_storage = False
        self.net.opt.use_fp16_arithmetic = False
        self.net.opt.use_bf16_storage = False
        for method, name in [(self.net.load_param, "model.ncnn.param"), (self.net.load_model, "model.ncnn.bin")]:
            if method(str(root / name)) != 0:
                self.close()
                raise RuntimeError(f"Failed to load {name}")
        self.forward_chunk(np.zeros((self.chunk_frames, 80), dtype=np.float32))

    @staticmethod
    def allocate_session_cache():
        return [np.zeros((1, 128, 19), dtype=np.float32) for _ in range(8)]

    @staticmethod
    def release_session_cache(cache):
        pass  # Arrays are owned by the session and reclaimed by Python.

    def reset_session_cache(self, cache):
        return self.allocate_session_cache()

    @staticmethod
    def session_cache_profile(cache):
        return None if cache is None else tuple(tuple(x.shape[1:]) for x in cache)

    @staticmethod
    def commit_session_cache(current, candidate):
        return candidate

    @staticmethod
    def discard_session_cache(candidate):
        pass  # Candidates never alias committed state.

    @staticmethod
    def cache_metrics():
        return None

    def _run(self, features, caches):
        length = len(features)
        padding = self.chunk_frames - length
        padded = np.pad(features, ((0, padding), (0, 0))) if padding else features
        with self.net.create_extractor() as ex:
            arrays = [padded, *(cache[0] for cache in caches)]
            # clone() transfers lifetime ownership away from the NumPy inputs.
            for i, value in enumerate(arrays):
                if ex.input(f"in{i}", ncnn.Mat(np.ascontiguousarray(value)).clone()) != 0:
                    raise RuntimeError(f"NCNN input in{i} failed")
            outputs = []
            for i in range(9):
                status, value = ex.extract(f"out{i}")
                if status != 0:
                    raise RuntimeError(f"NCNN output out{i} failed: {status}")
                outputs.append(np.array(value, dtype=np.float32, copy=True)[None])
        probs, states = outputs[0][:, :length].copy(), outputs[1:]
        if padding:
            # Remove padded time steps without losing the older valid history.
            states = [np.concatenate((old[:, :, :self.chunk_frames], new[:, :, :19-padding]), axis=2)[:, :, -19:].copy()
                      for old, new in zip(caches, states)]
        return probs, states

    def forward_chunk(self, feats: np.ndarray, caches=None):
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[1] != 80 or not np.isfinite(feats).all():
            raise ValueError("features must be finite [T, 80] float32")
        caches = self.allocate_session_cache() if caches is None else caches
        if len(caches) != 8 or any(x.shape != (1, 128, 19) or x.dtype != np.float32 for x in caches):
            raise ValueError("expected eight float32 caches of shape [1,128,19]")
        with self._lock:
            if self._closed:
                raise RuntimeError("Runtime is closed")
            if not len(feats):
                return np.empty((1, 0, 1), np.float32), [x.copy() for x in caches]
            probs = []
            for start in range(0, len(feats), self.chunk_frames):
                result, caches = self._run(feats[start:start+self.chunk_frames], caches)
                probs.append(result)
            return np.concatenate(probs, axis=1), caches

    forward_eager = forward_chunk

    def forward_batch(self, features, caches_batch):
        if len(features) != len(caches_batch):
            raise ValueError("features and caches_batch must have the same length")
        outputs = [self.forward_chunk(feat, cache) for feat, cache in zip(features, caches_batch)]
        return [x[0][0] for x in outputs], [x[1] for x in outputs]

    forward_session_batch = forward_batch

    def close(self):
        with self._lock:
            if not self._closed:
                self.net.clear()
                self._closed = True
