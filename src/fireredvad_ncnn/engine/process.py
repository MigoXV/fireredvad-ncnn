from __future__ import annotations

import logging
import multiprocessing
import os
import tempfile
import uuid
from pathlib import Path

from fireredvad_ncnn.configs.engine_config import EngineConfig
from fireredvad_ncnn.configs.runtime_config import RuntimeConfig
from fireredvad_ncnn.configs.stream_vad_config import FireRedStreamVadConfig
from fireredvad_ncnn.engine.worker import run_engine_worker

logger = logging.getLogger(__name__)


class EngineProcess:
    def __init__(
        self,
        *,
        model_path: str,
        runtime_config: RuntimeConfig,
        engine_config: EngineConfig,
        stream_vad_config: FireRedStreamVadConfig,
        endpoint_only: bool,
        max_payload_bytes: int,
        log_level: int,
    ) -> None:
        socket_name = f"fireredvad-{os.getpid()}-{uuid.uuid4().hex}.sock"
        self.ipc_path = Path(tempfile.gettempdir()) / socket_name
        self.endpoint = f"ipc://{self.ipc_path}"
        self.model_path = model_path
        self.runtime_config = runtime_config
        self.engine_config = engine_config
        self.stream_vad_config = stream_vad_config
        self.endpoint_only = endpoint_only
        self.max_payload_bytes = max_payload_bytes
        self.log_level = log_level
        self._process: multiprocessing.Process | None = None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def exitcode(self) -> int | None:
        return self._process.exitcode if self._process is not None else None

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("engine process already started")
        context = multiprocessing.get_context("spawn")
        self._process = context.Process(
            target=run_engine_worker,
            name="fireredvad-engine",
            kwargs={
                "endpoint": self.endpoint,
                "parent_pid": os.getpid(),
                "model_path": self.model_path,
                "runtime_config": self.runtime_config,
                "engine_config": self.engine_config,
                "stream_vad_config": self.stream_vad_config,
                "endpoint_only_default": self.endpoint_only,
                "max_payload_bytes": self.max_payload_bytes,
                "log_level": self.log_level,
            },
        )
        self._process.start()
        logger.info("Engine worker spawned: pid=%s endpoint=%s", self.pid, self.endpoint)

    def join_or_terminate(self) -> None:
        if self._process is None:
            self._cleanup_ipc()
            return
        self._process.join(timeout=self.engine_config.shutdown_grace_s)
        if self._process.is_alive():
            logger.warning("Engine worker did not stop in grace period; terminating")
            self._process.terminate()
            self._process.join(timeout=2.0)
        if self._process.is_alive():
            logger.error("Engine worker ignored terminate; killing")
            self._process.kill()
            self._process.join(timeout=2.0)
        logger.info("Engine worker stopped: exitcode=%s", self._process.exitcode)
        self._process.close()
        self._process = None
        self._cleanup_ipc()

    def _cleanup_ipc(self) -> None:
        try:
            self.ipc_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to remove engine IPC path: %s", self.ipc_path)
