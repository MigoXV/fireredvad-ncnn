"""Native gRPC service launcher and streaming benchmark helpers (dev dependencies)."""
from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

import grpc
import numpy as np
import psutil
from grpc_health.v1 import health_pb2, health_pb2_grpc

from fireredvad_ncnn.proto import ux_vad_pb2 as pb, ux_vad_pb2_grpc as rpc


class Server:
    def __init__(self, model, log_path, cpus=None):
        self.log_path = Path(log_path)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.address = f"127.0.0.1:{self.port}"
        base = ["poetry", "run", "python", "-m", "fireredvad_ncnn.commands.app", "serve",
                str(Path(model).resolve()), "--host", "127.0.0.1", "--port", str(self.port),
                "--stream-chunk-frame", "10", "--min-silence-frame", "20",
                "--engine-session-queue-capacity", "2", "--log-level", "WARNING"]
        if cpus:
            base = ["taskset", "-c", ",".join(map(str, cpus)), *base]
        self.command = base
        self.process = None
        self.log = None
        self.pids = []

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("w")
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
        # Resolve the project's Poetry environment instead of inheriting an unrelated venv.
        env.pop("VIRTUAL_ENV", None)
        env.pop("CONDA_PREFIX", None)
        env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep)
                                     if Path(p).resolve() != Path(sys.prefix, "bin").resolve())
        self.process = subprocess.Popen(self.command, stdout=self.log, stderr=subprocess.STDOUT,
                                        start_new_session=True, env=env)
        try:
            deadline = time.monotonic() + 180
            with grpc.insecure_channel(self.address) as channel:
                stub = health_pb2_grpc.HealthStub(channel)
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError(f"Service exited: {self.log_path.read_text()[-4000:]}")
                    try:
                        result = stub.Check(health_pb2.HealthCheckRequest(), timeout=0.3)
                        if result.status == health_pb2.HealthCheckResponse.SERVING:
                            self.pids = [p.pid for p in psutil.Process(self.process.pid).children(recursive=True)]
                            return self
                    except grpc.RpcError:
                        pass
                    time.sleep(0.1)
            raise TimeoutError(f"Service startup timed out: {self.log_path}")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def resources(self):
        processes = [psutil.Process(self.process.pid)]
        processes += processes[0].children(recursive=True)
        result = {"rss_bytes": 0, "cpu_seconds": 0.0, "threads": 0}
        for p in processes:
            try:
                result["rss_bytes"] += p.memory_info().rss
                t = p.cpu_times()
                result["cpu_seconds"] += t.user + t.system
                result["threads"] += p.num_threads()
            except psutil.NoSuchProcess:
                pass
        return result

    def __exit__(self, *_):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=5)
            for _ in range(30):
                alive = [pid for pid in self.pids if psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE]
                if not alive:
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(f"Service left child processes alive: {alive}")
        if self.log:
            self.log.close()


def requests(audio, sr=16000, paced=False, send_times=None):
    yield pb.StreamingDetectRequest(config=pb.DetectionConfig(encoding=1, sample_rate_hertz=sr))
    step = sr // 10 * 2
    begin = time.monotonic()
    for offset in range(0, len(audio), step):
        if paced:
            due = begin + min(offset+step, len(audio)) / (sr*2)
            time.sleep(max(0, due-time.monotonic()))
        if send_times is not None:
            send_times.append(time.monotonic())
        yield pb.StreamingDetectRequest(audio_content=audio[offset:offset+step])


def stream(address, audio, barrier=None, paced=False, sr=16000):
    with grpc.insecure_channel(address) as channel:
        grpc.channel_ready_future(channel).result(timeout=10)
        stub = rpc.UxVoiceActivityDetectorStub(channel)
        if barrier:
            barrier.wait(timeout=15)
        start = time.monotonic()
        sends, arrivals, events = [], [], []
        try:
            for response in stub.StreamingDetect(requests(audio, sr, paced, sends), timeout=30):
                arrivals.append(time.monotonic()-start)
                events.append({"watermark_ns": response.processed_time.ToNanoseconds(),
                               "segments": [[r.start_time.ToNanoseconds(), r.end_time.ToNanoseconds()] for r in response.results]})
            elapsed = time.monotonic()-start
            return {"ok": True, "elapsed_s": elapsed, "speed_x": len(audio)/(sr*2)/elapsed,
                    "first_response_s": arrivals[0] if arrivals else None,
                    "response_interarrival_ms": (np.diff(arrivals)*1000).tolist(),
                    "eof_drain_ms": (time.monotonic()-sends[-1])*1000 if sends else None,
                    "events": events}
        except grpc.RpcError as exc:
            return {"ok": False, "elapsed_s": time.monotonic()-start, "error": str(exc)}


def measure(server, audio, concurrency):
    start_box = []
    barrier = threading.Barrier(concurrency+1, action=lambda: start_box.append(time.monotonic()))
    before = server.resources()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(stream, server.address, audio, barrier) for _ in range(concurrency)]
        barrier.wait(timeout=15)
        results = [f.result(timeout=35) for f in futures]
        elapsed = time.monotonic()-start_box[0]
    after = server.resources()
    ok = [r for r in results if r["ok"]]
    durations = [r["elapsed_s"] for r in ok]
    for r in results:
        r.pop("events", None)
        r.pop("response_interarrival_ms", None)
    return {"concurrency": concurrency, "makespan_s": elapsed,
            "aggregate_x": len(ok)*len(audio)/32000/elapsed,
            "min_stream_x": min((r["speed_x"] for r in ok), default=0),
            "completion_p50_p95_p99_s": np.percentile(durations, [50,95,99]).tolist() if ok else [],
            "errors": concurrency-len(ok), "streams": results,
            "server_rss_bytes": after["rss_bytes"], "server_threads": after["threads"],
            "server_cpu_percent": (after["cpu_seconds"]-before["cpu_seconds"])/elapsed*100}
