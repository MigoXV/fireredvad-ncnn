"""Controlled native migration benchmark against the frozen Python service.

Run from the repo: poetry run python scripts/benchmark_native.py --output PATH
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

from fireredvad_ncnn.benchmark import Server, measure, stream
from fireredvad_ncnn.native import _load_library


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("model-bin/stream-vad"))
    parser.add_argument("--wav", type=Path, default=Path("model-bin/speech.wav"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", default="1,4,8,16,32")
    args = parser.parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    args.output.mkdir(parents=True, exist_ok=False)
    cores = []
    seen = set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((topology / "physical_package_id").read_text().strip(),
               (topology / "core_id").read_text().strip())
        if key not in seen:
            seen.add(key)
            cores.append(cpu)
    if len(cores) < 8:
        raise RuntimeError("Need 8 available physical cores: 4 server, 4 clients")
    server_cpus, client_cpus = cores[:4], cores[4:8]
    os.sched_setaffinity(0, client_cpus)
    audio, rate = sf.read(args.wav,dtype="int16")
    if rate != 16000 or audio.ndim != 1:
        raise RuntimeError("Use mono 16kHz WAV for migration benchmark")
    audio = np.tile(audio, int(np.ceil(160000 / len(audio))))[:160000].tobytes()
    env = {"revision": subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
           "worktree": subprocess.check_output(["git","status","--short"],text=True),
           "platform": platform.platform(), "server_cpus": server_cpus,"client_cpus":client_cpus,
           "cpu_model": next((line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),"unknown"),
           "native_build": _load_library().frv_build_info().decode(),
           "model_metadata": json.loads((args.model/"metadata.json").read_text()),
           "duration_s":10,"rounds":3,"wav_sha256":hashlib.sha256(args.wav.read_bytes()).hexdigest(),
           "process_snapshot": [{"pid":p.pid,"name":p.name()} for p in psutil.process_iter()],
           "acceptance": "default native throughput >= 90% baseline; default native P95 <= 110% baseline",
           "isolation": "unique ports/process groups and disjoint server/client affinity; no host exclusivity guarantee"}
    (args.output/"environment.json").write_text(json.dumps(env,indent=2))
    records = []
    stopped = set()
    with (args.output/"raw.jsonl").open("w") as raw:
        for level_index, concurrency in enumerate(levels):
            order = ["python-reference","native-default","native-parallel"]
            if level_index % 2:
                order.reverse()
            for backend in order:
                if backend in stopped:
                    continue
                server = Server("ncnn", args.model, Path("."),args.output/f"{backend}-{concurrency}.log",server_cpus)
                server.command[server.command.index("--log-level")+1] = "INFO"
                if backend == "python-reference":
                    index = server.command.index("fireredvad_ncnn.commands.app")
                    server.command[index] = "tests.reference.serve"
                elif backend == "native-parallel":
                    server.command += ["--engine-workers","4","--threads-per-worker","1"]
                with server:
                    warmup = stream(server.address,audio[:64000])
                    if not warmup["ok"]:
                        raise RuntimeError(warmup)
                    short = measure(server,audio[:64000],concurrency)
                    record = {"backend":backend,"concurrency":concurrency,"kind":"precheck",**short}
                    records.append(record);raw.write(json.dumps(record)+"\n");raw.flush()
                    if short["errors"] or short["min_stream_x"] < 2:
                        stopped.add(backend)
                        continue
                    for iteration in range(3):
                        record = {"backend":backend,"concurrency":concurrency,"kind":"formal","round":iteration,
                                  **measure(server,audio,concurrency)}
                        records.append(record);raw.write(json.dumps(record)+"\n");raw.flush()
                        if record["errors"] or record["min_stream_x"] < 2:
                            stopped.add(backend)
                            break
                for line in server.log_path.read_text().splitlines():
                    if "engine_metrics_final=" in line:
                        record = {"backend":backend,"concurrency":concurrency,"kind":"native_metrics",
                                  **json.loads(line.split("engine_metrics_final=",1)[1])}
                        raw.write(json.dumps(record)+"\n");raw.flush()
    summary = []
    for concurrency in levels:
        for backend in ["python-reference","native-default","native-parallel"]:
            group = [r for r in records if r["concurrency"]==concurrency and r["backend"]==backend and r["kind"]=="formal"]
            if group:
                summary.append({"backend":backend,"concurrency":concurrency,"rounds":len(group),
                    "median_throughput":float(np.median([r["aggregate_x"] for r in group])),
                    "median_p95_s":float(np.median([r["completion_p50_p95_p99_s"][1] for r in group])),
                    "failures":sum(r["errors"] for r in group)})
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
