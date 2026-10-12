"""Short, reproducible end-to-end CPU comparison (development dependencies)."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from importlib.metadata import version

import grpc
import numpy as np
import psutil
import soundfile as sf
import soxr
from grpc_health.v1 import health_pb2, health_pb2_grpc

from fireredvad_ncnn.proto import ux_vad_pb2 as pb, ux_vad_pb2_grpc as rpc


class Server:
    def __init__(self, backend, model, reference_project, log_path, cpus=None):
        self.backend = backend
        self.log_path = Path(log_path)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.address = f"127.0.0.1:{self.port}"
        base = ["poetry"]
        if backend == "torch":
            base += ["-C", str(reference_project)]
        base += ["run", "python", "-m", "fireredvad.commands.app" if backend == "torch" else "fireredvad_ncnn.commands.app"]
        if backend == "ncnn":
            base += ["serve"]
        base += [str(Path(model).resolve()), "--device", "cpu", "--dtype", "fp32",
                 "--runner", "eager" if backend == "torch" else "ncnn",
                 "--host", "127.0.0.1", "--port", str(self.port),
                 "--batch-size", "32", "--batch-wait-ms", "5", "--stream-chunk-frame", "10",
                 "--min-silence-frame", "20", "--engine-session-queue-capacity", "2",
                 "--log-level", "WARNING"]
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
        # Nested Poetry otherwise reuses the caller's active virtualenv even with -C.
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
            # Resource tracker may take a moment to observe the pipe closing.
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


def benchmark(model_path, reference_project, reference_model, wav_dir, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "raw.jsonl").exists():
        raise ValueError("Choose a fresh benchmark output directory")
    samples, paths = [], []
    for path in sorted(Path(wav_dir).glob("*.wav")):
        audio, sr = sf.read(path, dtype="int16")
        if audio.ndim != 1:
            continue
        if sr != 16000:
            audio = np.clip(np.rint(soxr.resample(audio.astype(np.float32), sr, 16000)), -32768, 32767).astype(np.int16)
        samples.append(audio)
        paths.append(str(path))
        if sum(map(len, samples)) >= 160000:
            break
    if not samples:
        raise ValueError("Need mono 16 kHz WAV samples")
    audio = np.concatenate(samples)[:160000].astype("<i2").tobytes()
    if len(audio) < 320000:
        audio = (audio * (320000//len(audio)+1))[:320000]
    available = sorted(os.sched_getaffinity(0))
    # Pick distinct physical cores, not SMT siblings, using Linux topology.
    physical, seen = [], set()
    for cpu in available:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((topology/"physical_package_id").read_text(), (topology/"core_id").read_text())
        if key not in seen:
            physical.append(cpu)
            seen.add(key)
    if len(physical) < 8:
        raise RuntimeError("This fixed benchmark needs four server and four client physical cores")
    server_cpus, client_cpus = physical[:4], physical[4:8]
    environment = {"server_cpus": server_cpus, "client_cpus": client_cpus, "compute_threads": 4, "session_queue_capacity": 2,
                   "audio_paths": paths, "audio_sha256": hashlib.sha256(audio).hexdigest(),
                   "duration_s": 10, "preparation": "resample to 16kHz with soxr before timing", "chunk_ms": 100, "min_silence_frame": 20, "batch_wait_ms": 5,
                   "versions": {k: version(k) for k in ["ncnn", "numpy", "grpcio"]},
                   "cpu": subprocess.check_output(["lscpu"], text=True),
                   "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "reference_revision": subprocess.check_output(["git", "-C", str(reference_project), "rev-parse", "HEAD"], text=True).strip()}
    sf.write(output/"input.wav", np.frombuffer(audio, "<i2"), 16000, subtype="PCM_16")
    (output/"environment.json").write_text(json.dumps(environment, indent=2))
    def record(data):
        with (output/"raw.jsonl").open("a") as f:
            f.write(json.dumps(data)+"\n")
    summaries, stopped = [], {}
    os.sched_setaffinity(0, client_cpus)
    try:
        for level_index, concurrency in enumerate([1,4,8,16,32]):
            order = ["torch", "ncnn"] if level_index % 2 == 0 else ["ncnn", "torch"]
            for backend in order:
                if backend in stopped:
                    record({"backend": backend, "concurrency": concurrency, "event": "skipped", "reason": stopped[backend]})
                    continue
                model = reference_model if backend == "torch" else model_path
                with Server(backend, model, reference_project, output/f"{backend}-{concurrency}.log", server_cpus) as server:
                    record({"event": "command", "backend": backend, "command": server.command})
                    warmup = stream(server.address, audio)
                    if not warmup["ok"]:
                        raise RuntimeError(warmup)
                    if concurrency == 1:
                        for paced_round in range(1, 4):
                            paced = stream(server.address, audio[:64000], paced=True)
                            record({"event": "paced", "backend": backend, "round": paced_round, **paced})
                            if not paced["ok"]:
                                raise RuntimeError(paced)
                    probe = measure(server, audio[:64000], concurrency)
                    record({"event": "probe", "backend": backend, **probe})
                    if probe["errors"] or probe["min_stream_x"] < 2:
                        stopped[backend] = f"probe at {concurrency}: min_stream_x={probe['min_stream_x']:.3f}, errors={probe['errors']}"
                        record({"event": "stopped", "backend": backend, "reason": stopped[backend]})
                        print(stopped[backend], flush=True)
                        continue
                    rounds = []
                    for i in range(3):
                        result = measure(server, audio, concurrency)
                        rounds.append(result)
                        record({"event": "round", "backend": backend, "round": i+1, **result})
                        if result["errors"] or result["min_stream_x"] < 2:
                            stopped[backend] = f"formal round below threshold or failed at {concurrency}"
                            break
                    summary = {"backend": backend, "concurrency": concurrency, "rounds": len(rounds),
                               "aggregate_x_median": statistics.median(r["aggregate_x"] for r in rounds),
                               "min_stream_x": min(r["min_stream_x"] for r in rounds),
                               "errors": sum(r["errors"] for r in rounds)}
                    summaries.append(summary)
                    print(json.dumps(summary), flush=True)
    finally:
        os.sched_setaffinity(0, available)
        (output/"summary.json").write_text(json.dumps({"cases": summaries, "stopped": stopped}, indent=2))
