"""Reproducible NCNN INT8 experiments; all large artifacts live outside the repo.

Run with the research Poetry environment documented in reports/quantization-20261020.
This script adds no production defaults and never stops system services.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import logging
import math
import os
import platform
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import soundfile as sf
import typer

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger(__name__)
TEN_REVISION = "d68ab71c8d5fbd8f98e97ae7c9986af76b5d72b1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_labels(path: Path) -> list[tuple[float, float, int]]:
    fields = path.read_text().strip().split(",")
    if len(fields) < 4 or (len(fields) - 1) % 3:
        raise ValueError(f"Invalid TEN labels: {path}")
    segments = [(float(fields[i]), float(fields[i + 1]), int(fields[i + 2]))
                for i in range(1, len(fields), 3)]
    previous_end = 0.0
    for start, end, label in segments:
        if label not in (0, 1) or not all(map(math.isfinite, (start, end))):
            raise ValueError("Invalid label or nonfinite timestamp")
        if start < 0 or end <= start or abs(start - previous_end) > 0.002:
            raise ValueError("Labels must cover contiguous ordered intervals from zero")
        previous_end = end
    return segments


def binary_at_times(segments, times):
    result = np.zeros(len(times), dtype=bool)
    valid = np.zeros(len(times), dtype=bool)
    for start, end, label in segments:
        selected = (times >= start) & (times < end)
        valid |= selected
        result[selected] = bool(label)
    return result, valid


def confusion(reference, hypothesis):
    reference, hypothesis = np.asarray(reference, bool), np.asarray(hypothesis, bool)
    if reference.shape != hypothesis.shape:
        raise ValueError("Reference and hypothesis must have identical shape")
    return {"tp": int(np.sum(reference & hypothesis)),
            "tn": int(np.sum(~reference & ~hypothesis)),
            "fp": int(np.sum(~reference & hypothesis)),
            "fn": int(np.sum(reference & ~hypothesis))}


def metrics(counts):
    tp, tn, fp, fn = (counts[n] for n in ("tp", "tn", "fp", "fn"))
    return {**counts, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "miss_rate": fn / max(1, tp + fn), "false_alarm_rate": fp / max(1, fp + tn),
            "frame_error_rate": (fp + fn) / max(1, tp + tn + fp + fn)}


def speech_union_labels(intervals, duration):
    """Convert RTTM speakers, including overlaps, to one complete VAD timeline."""
    merged = []
    for start, end in sorted(intervals):
        start, end = max(0., start), min(duration, end)
        if not math.isfinite(start + end) or end <= start:
            raise ValueError("Invalid RTTM speech interval")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    labels, cursor = [], 0.
    for start, end in merged:
        if start > cursor:
            labels.append((cursor, start, 0))
        labels.append((start, end, 1))
        cursor = end
    if cursor < duration:
        labels.append((cursor, duration, 0))
    return labels


def dataset_manifest(root: Path):
    path = root / "provenance.json"
    if not path.is_file():
        path = root / "manifest.json"
    return json.loads(path.read_text()), path


def dataset_files(root: Path, split: str):
    manifest, _ = dataset_manifest(root)
    if "records" in manifest:
        for record in manifest["records"]:
            if record["split"] != split:
                continue
            audio, rttm = Path(record["audio"]), Path(record["rttm"])
            if sha256(audio) != record["audio_sha256"] or sha256(rttm) != record["rttm_sha256"]:
                raise ValueError("VoxConverse dataset SHA256 mismatch")
            intervals = []
            for line in rttm.read_text().splitlines():
                fields = line.split()
                if fields[0] != "SPEAKER":
                    raise ValueError("Unsupported RTTM row")
                start, duration = float(fields[3]), float(fields[4])
                intervals.append((start, start + duration))
            yield audio, speech_union_labels(intervals, sf.info(audio).duration)
        return
    files = [r for r in manifest["files"] if r["split"] == split and r["path"].endswith(".wav")]
    for record in sorted(files, key=lambda r: r["path"]):
        path = root / Path(record["path"]).name
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Dataset SHA256 mismatch: {path}")
        labels = path.with_suffix(".scv")
        label_record = next(r for r in manifest["files"] if Path(r["path"]).name == labels.name)
        if sha256(labels) != label_record["sha256"]:
            raise ValueError(f"Label SHA256 mismatch: {labels}")
        yield path, read_labels(labels)


@app.command()
def fetch(output: Path = typer.Option(..., envvar="VAD_DATASET_DIR")):
    """Fetch the 30 public manually annotated TEN test clips at a pinned revision."""
    output.mkdir(parents=True, exist_ok=True)
    tree_url = f"https://api.github.com/repos/TEN-framework/ten-vad/git/trees/{TEN_REVISION}?recursive=1"
    tree = json.load(urllib.request.urlopen(tree_url, timeout=60))

    def download(item):
        path = output / Path(item["path"]).name
        url = f"https://raw.githubusercontent.com/TEN-framework/ten-vad/{TEN_REVISION}/{item['path']}"
        content = urllib.request.urlopen(url, timeout=60).read()
        git_blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if git_blob != item["sha"]:
            raise ValueError(f"Git blob mismatch: {url}")
        path.write_bytes(content)
        number = int(path.stem.rsplit("-", 1)[1])
        return {"path": str(path), "split": "calibration" if number <= 10 else "test",
                "source_url": url, "git_blob": git_blob, "sha256": sha256(path)}

    selected = [i for i in tree["tree"] if i["path"].startswith("testset/")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(download, selected))
    write_json(output / "provenance.json", {
        "dataset": "TEN-framework/ten-vad testset", "revision": TEN_REVISION,
        "readme_url": f"https://github.com/TEN-framework/ten-vad/blob/{TEN_REVISION}/README.md",
        "split_rule": "01-10 calibration; 11-30 held-out test, fixed before evaluating",
        "files": records})


def rewrite_gemm_to_innerproduct(source_param: str, source_bin: bytes):
    """Rewrite only the exact constant-B Gemm form emitted by FireRedVAD pnnx.

    Gemm B is [N,K], equal to InnerProduct weights. Gemm's optional constant C
    has its own NCNN type tag, whereas InnerProduct bias is raw float32.
    Unsupported layer forms fail instead of guessing a binary layout.
    """
    offset = 0
    result = bytearray()
    lines = source_param.splitlines()
    transformed = []
    weightless = {"Input", "ReLU", "Permute", "Split", "Concat", "Crop", "BinaryOp", "Sigmoid"}

    def block(count, tagged=True):
        nonlocal offset
        size = count * 4 + (4 if tagged else 0)
        data = source_bin[offset:offset + size]
        if len(data) != size:
            raise ValueError("Truncated NCNN model")
        offset += size
        if tagged and struct.unpack_from("<I", data)[0] != 0:
            raise ValueError("Only tagged FP32 NCNN weights are supported")
        return data

    for line_index in range(2, len(lines)):
        words = lines[line_index].split()
        if not words:
            continue
        kind, name = words[:2]
        inputs, outputs = int(words[2]), int(words[3])
        params_at = 4 + inputs + outputs
        params = {int(k): v for k, v in (s.split("=", 1) for s in words[params_at:])}
        if kind == "Gemm":
            expected = {2: "0", 3: "1", 4: "0", 5: "1", 6: "1"}
            if inputs != 1 or outputs != 1 or any(params.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Unsupported Gemm shape: {name}")
            if set(params) - {2, 3, 4, 5, 6, 7, 8, 9, 10}:
                raise ValueError(f"Unsupported Gemm parameters: {name}")
            n, k = int(params[8]), int(params[9])
            broadcast = int(params[10])
            if broadcast not in (-1, 4):
                raise ValueError(f"Unsupported Gemm bias broadcast: {name}")
            result.extend(block(n * k))
            if broadcast == 4:
                result.extend(block(n)[4:])
            prefix = ["InnerProduct", *words[1:params_at]]
            lines[line_index] = " ".join(prefix + [f"0={n}", f"1={int(broadcast == 4)}", f"2={n*k}"])
            transformed.append(name)
        elif kind == "ConvolutionDepthWise1D":
            if params.get(5) != "0":
                raise ValueError("Unsupported depthwise bias")
            result.extend(block(int(params[6])))
        elif kind not in weightless:
            raise ValueError(f"Unsupported NCNN layer: {kind}")
    if offset != len(source_bin):
        raise ValueError(f"Unconsumed NCNN model bytes: {len(source_bin) - offset}")
    if not transformed:
        raise ValueError("No Gemm layers found")
    return "\n".join(lines) + "\n", bytes(result), transformed


def rewrite_gemm_constant_a(source_param: str, source_bin: bytes):
    """Compute (W @ X.T + b[:,None]).T so NCNN quantizes W per output row.

    The original constant-B NCNN INT8 path uses one weight scale for the whole
    matrix. Constant A has per-row scales, so exchanging the operands can retain
    channel detail without changing any FP32 model weights or adding operators.
    """
    # Reuse the strict complete FP32 binary validation before changing metadata.
    _, _, transformed = rewrite_gemm_to_innerproduct(source_param, source_bin)
    lines = source_param.splitlines()
    for line_index in range(2, len(lines)):
        words = lines[line_index].split()
        if not words or words[0] != "Gemm":
            continue
        params_at = 4 + int(words[2]) + int(words[3])
        params = dict(s.split("=", 1) for s in words[params_at:])
        bias_type = "1" if params["10"] == "4" else "-1"
        lines[line_index] = " ".join(words[:params_at] + [
            "2=0", "3=1", "4=1", "5=0", "6=1", f"7={params['8']}",
            f"8={params['7']}", f"9={params['9']}", f"10={bias_type}", "14=1"])
    return "\n".join(lines) + "\n", source_bin, transformed


def copy_metadata(source: Path, target: Path, quantization):
    target.mkdir(parents=True, exist_ok=True)
    (target / "cmvn.ark").write_bytes((source / "cmvn.ark").read_bytes())
    meta = json.loads((source / "metadata.json").read_text())
    meta["quantization"] = quantization
    # dtype describes external floating-point features, cache and outputs.
    meta["files"] = {name: sha256(target / name) for name in ("model.ncnn.param", "model.ncnn.bin", "cmvn.ark")}
    write_json(target / "metadata.json", meta)


@app.command()
def rewrite(source: Path = typer.Option(..., envvar="VAD_FP32_MODEL"),
            output: Path = typer.Option(..., envvar="VAD_INNERPRODUCT_MODEL"),
            strategy: str = typer.Option("innerproduct")):
    """Create an equivalent FP32 graph before evaluating its INT8 conversion."""
    transforms = {"innerproduct": rewrite_gemm_to_innerproduct, "constant-a": rewrite_gemm_constant_a}
    if strategy not in transforms:
        raise ValueError("Strategy must be innerproduct or constant-a")
    output.mkdir(parents=True, exist_ok=False)
    param, binary, names = transforms[strategy]((source / "model.ncnn.param").read_text(),
                                               (source / "model.ncnn.bin").read_bytes())
    (output / "model.ncnn.param").write_text(param)
    (output / "model.ncnn.bin").write_bytes(binary)
    copy_metadata(source, output, {"method": "none", "graph_rewrite": strategy,
                                   "rewritten_layers": names})


def extract_features(audio, rate, cmvn: Path):
    import kaldi_native_fbank as knf
    import kaldiio
    if rate != 16000 or audio.ndim != 1:
        raise ValueError("Research inputs must be mono PCM16 16kHz")
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.frame_length_ms = 25
    opts.frame_opts.frame_shift_ms = 10
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = 80
    opts.mel_opts.debug_mel = False
    fbank = knf.OnlineFbank(opts)
    features = fbank.accept_waveform_and_get_frames(rate, np.ascontiguousarray(audio), is_final=True)
    stats = kaldiio.load_mat(str(cmvn))
    count = stats[0, -1]
    means = np.asarray([float(stats[0, d] / count) for d in range(80)])
    inv = np.asarray([1 / math.sqrt(max(1e-20, float(stats[1, d] / count) - means[d]**2))
                      for d in range(80)])
    return np.asarray((features - means) * inv, dtype=np.float32)


class NcnnModel:
    def __init__(self, root: Path):
        import ncnn
        self.ncnn = ncnn
        self.net = ncnn.Net()
        self.net.opt.num_threads = 1
        self.net.opt.use_vulkan_compute = False
        self.net.opt.use_fp16_packed = False
        self.net.opt.use_fp16_storage = False
        self.net.opt.use_fp16_arithmetic = False
        self.net.opt.use_bf16_storage = False
        if self.net.load_param(str(root / "model.ncnn.param")) or self.net.load_model(str(root / "model.ncnn.bin")):
            raise RuntimeError(f"Cannot load NCNN model {root}")
        self.chunk = json.loads((root / "metadata.json").read_text())["chunk_frames"]

    def forward(self, features, capture=None):
        caches = [np.zeros((128, 19), dtype=np.float32) for _ in range(8)]
        probabilities = []
        for index, start in enumerate(range(0, len(features), self.chunk)):
            value = features[start:start + self.chunk]
            value = np.pad(value, ((0, self.chunk - len(value)), (0, 0)))
            arrays = [value, *caches]
            if capture is not None:
                capture(index, arrays)
            with self.net.create_extractor() as ex:
                for i, array in enumerate(arrays):
                    if ex.input(f"in{i}", self.ncnn.Mat(np.ascontiguousarray(array)).clone()):
                        raise RuntimeError("NCNN input failed")
                out = []
                for i in range(9):
                    status, tensor = ex.extract(f"out{i}")
                    if status:
                        raise RuntimeError("NCNN extract failed")
                    out.append(np.asarray(tensor, dtype=np.float32).copy())
            probabilities.extend(out[0].reshape(-1).tolist())
            caches = out[1:]
        return np.asarray(probabilities[:len(features)], np.float32)


@app.command()
def calibration(model: Path = typer.Option(..., envvar="VAD_FP32_MODEL"),
                dataset: Path = typer.Option(..., envvar="VAD_DATASET_DIR"),
                output: Path = typer.Option(..., envvar="VAD_CALIBRATION_DIR"),
                chunks_per_file: int = typer.Option(16, min=1)):
    """Capture real features and eight recurrent caches using only calibration files."""
    output.mkdir(parents=True, exist_ok=False)
    runner = NcnnModel(model)
    files = [[] for _ in range(9)]
    records = []
    for path, _ in dataset_files(dataset, "calibration"):
        audio, rate = sf.read(path, dtype="int16")
        features = extract_features(audio, rate, model / "cmvn.ark")
        selected = set(np.linspace(0, math.ceil(len(features) / runner.chunk) - 1,
                                  min(chunks_per_file, math.ceil(len(features) / runner.chunk)), dtype=int).tolist())

        def capture(index, arrays):
            if index not in selected:
                return
            record = {"audio": str(path), "chunk": index, "inputs": []}
            for i, array in enumerate(arrays):
                target = output / f"{path.stem}-{index:05d}-in{i}.npy"
                np.save(target, np.ascontiguousarray(array, dtype=np.float32))
                files[i].append(str(target))
                record["inputs"].append({"path": str(target), "sha256": sha256(target),
                                         "min": float(array.min()), "max": float(array.max())})
            records.append(record)
        runner.forward(features, capture)
    for i, paths in enumerate(files):
        (output / f"in{i}.list").write_text("\n".join(paths) + "\n")
    write_json(output / "manifest.json", {"source_model": str(model), "source_param_sha256": sha256(model / "model.ncnn.param"),
               "dataset_provenance": dataset_manifest(dataset)[0], "split": "calibration", "records": records,
               "cache_policy": "actual FP32 previous output; zeros only on the first chunk of each file"})


@app.command()
def quantize(source: Path = typer.Option(..., envvar="VAD_FP32_MODEL"),
             output: Path = typer.Option(..., envvar="VAD_QUANTIZED_MODEL"),
             calibration_dir: Path = typer.Option(..., envvar="VAD_CALIBRATION_DIR"),
             tools_dir: Path = typer.Option(..., envvar="NCNN_QUANTIZATION_TOOLS"),
             method: str = typer.Option("kl")):
    """Invoke the official calibration and INT8 tools, preserving logs and provenance."""
    if method not in {"kl", "aciq", "eq"}:
        raise ValueError("Unsupported calibration method")
    output.mkdir(parents=True, exist_ok=False)
    lists = ",".join(str(calibration_dir / f"in{i}.list") for i in range(9))
    chunk = int(json.loads((source / "metadata.json").read_text())["chunk_frames"])
    shape = f"shape=[80,{chunk}]," + ",".join(["[19,128]"] * 8)
    table = output / "calibration.table"
    commands = [
        [str(tools_dir / "ncnn2table"), str(source / "model.ncnn.param"), str(source / "model.ncnn.bin"),
         lists, str(table), shape, "thread=2", f"method={method}", "type=1"],
        [str(tools_dir / "ncnn2int8"), str(source / "model.ncnn.param"), str(source / "model.ncnn.bin"),
         str(output / "model.ncnn.param"), str(output / "model.ncnn.bin"), str(table)],
    ]
    for command in commands:
        with (output / f"{Path(command[0]).name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    names = [line.split()[1] for line in (output / "model.ncnn.param").read_text().splitlines()[2:]
             if "8=2" in line.split() or "18=2" in line.split()]
    quantization = {"method": f"official-ncnn2table-{method}/ncnn2int8", "commands": commands,
                    "calibration_manifest": str(calibration_dir / "manifest.json"),
                    "calibration_manifest_sha256": sha256(calibration_dir / "manifest.json"),
                    "source_param_sha256": sha256(source / "model.ncnn.param"),
                    "source_bin_sha256": sha256(source / "model.ncnn.bin"),
                    "table_bytes": table.stat().st_size, "int8_layers": names,
                    "note": "Gemm dynamically quantizes activations; InnerProduct uses real-data static activation calibration; depthwise Conv1D remains floating point; official ncnn2int8 stores unquantized weights in FP16 by default",
                    "tools": {name: sha256(tools_dir / name) for name in ("ncnn2table", "ncnn2int8")}}
    copy_metadata(source, output, quantization)
    write_json(output / "conversion.json", quantization)


def source_model(source_repo: Path, weights: Path):
    import torch
    from safetensors.torch import load_file
    module_path = source_repo / "src/fireredvad/models/firered_vad_stream/module_firered_vad_stream.py"
    spec = importlib.util.spec_from_file_location("firered_source_of_truth", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = SimpleNamespace(**json.loads((weights / "config.json").read_text()))
    model = module.FireRedVadStreamModule.from_config(config).eval()
    state = load_file(str(weights / "model.safetensors"))
    model.load_state_dict({k.removeprefix("vad."): v for k, v in state.items()}, strict=True)
    torch.set_num_threads(1)
    return model


def source_forward(model, features, chunk):
    import torch
    caches, out = None, []
    with torch.inference_mode():
        for start in range(0, len(features), chunk):
            probs, caches = model(torch.from_numpy(features[start:start + chunk])[None], caches=caches)
            out.append(probs.numpy().reshape(-1))
    return np.concatenate(out)


@app.command()
def accuracy(models: list[Path] = typer.Option(..., "--model", envvar="VAD_EVAL_MODELS"),
             dataset: Path = typer.Option(..., envvar="VAD_DATASET_DIR"),
             output: Path = typer.Option(..., envvar="VAD_ACCURACY_OUTPUT"),
             source_repo: Path | None = typer.Option(None, envvar="VAD_SOURCE_REPO"),
             weights: Path | None = typer.Option(None, envvar="VAD_SOURCE_WEIGHTS")):
    """Evaluate held-out raw frame probabilities and production native segment output."""
    from fireredvad_ncnn.native import NativeEngine
    from fireredvad_ncnn.configs.stream_vad_config import FireRedStreamVadConfig
    output.mkdir(parents=True, exist_ok=False)
    runners = {p.name: NcnnModel(p) for p in models}
    if len(runners) != len(models):
        raise ValueError("Model directory names must be unique")
    oracle = source_model(source_repo, weights) if source_repo is not None and weights is not None else None
    aggregate = {name: {kind: {k: 0 for k in ("tp", "tn", "fp", "fn")}
                       for kind in ("raw", "segments")} for name in runners}
    torch_counts = {k: 0 for k in ("tp", "tn", "fp", "fn")}
    engines = {p.name: NativeEngine(p, cpu_budget=1, worker_count=1, threads_per_worker=1) for p in models}
    raw = []
    try:
        for path, labels in dataset_files(dataset, "test"):
            audio, rate = sf.read(path, dtype="int16")
            features = extract_features(audio, rate, models[0] / "cmvn.ark")
            times = np.arange(len(features)) * .01 + .0125
            reference, valid = binary_at_times(labels, times)
            oracle_probs = source_forward(oracle, features, runners[models[0].name].chunk) if oracle is not None else None
            if oracle_probs is not None:
                np.save(output / f"{path.stem}-source-fp32.npy", oracle_probs)
                counts = confusion(reference[valid], oracle_probs[valid] >= .5)
                for key in counts:
                    torch_counts[key] += counts[key]
            base_probs = None
            for name, runner in runners.items():
                probs = runner.forward(features)
                if base_probs is None:
                    base_probs = probs
                np.save(output / f"{path.stem}-{name}.npy", probs)
                events = engines[name].wait(engines[name].submit(audio, rate))
                segment_times = np.arange(math.floor(len(audio) / rate / .01)) * .01 + .005
                segment_ref, segment_valid = binary_at_times(labels, segment_times)
                hypothesis, _ = binary_at_times([(s, e, 1) for s, e, _ in events], segment_times)
                counts = {"raw": confusion(reference[valid], probs[valid] >= .5),
                          "segments": confusion(segment_ref[segment_valid], hypothesis[segment_valid])}
                delta = probs - (oracle_probs if oracle_probs is not None else base_probs)
                record = {"file": str(path), "audio_seconds": len(audio) / rate, "model": name,
                          "max_abs_probability_error": float(np.abs(delta).max()),
                          "mean_abs_probability_error": float(np.abs(delta).mean()),
                          "raw": metrics(counts["raw"]), "segments": metrics(counts["segments"]),
                          "events": events}
                raw.append(record)
                for kind in counts:
                    for key in counts[kind]:
                        aggregate[name][kind][key] += counts[kind][key]
    finally:
        for engine in engines.values():
            engine.close()
    write_json(output / "per_file.json", raw)
    summary = {name: {kind: metrics(counts) for kind, counts in kinds.items()} for name, kinds in aggregate.items()}
    if oracle is not None:
        summary["source_torch_fp32"] = {"raw": metrics(torch_counts)}
    write_json(output / "summary.json", summary)
    write_json(output / "methodology.json", {"dataset_provenance": dataset_manifest(dataset)[0], "split": "test", "threshold": .5,
        "raw_frame_time": "feature center = frame_index*10ms+12.5ms; only valid labeled frames scored",
        "segment_frame_time": "10ms grid midpoint; production native offline postprocessing; no collar",
        "postprocessing": asdict(FireRedStreamVadConfig()), "models": [str(p) for p in models],
        "source_repo": str(source_repo), "source_revision": subprocess.check_output(["git", "-C", str(source_repo), "rev-parse", "HEAD"], text=True).strip() if source_repo else None,
        "acceptance": "Task F1 loss <=0.005 absolute and no crash; performance candidate must also improve measured throughput"})


def resource_snapshot(pid=None):
    p = psutil.Process(pid)
    return {"cpu_s": sum(p.cpu_times()[:2]), "rss_bytes": p.memory_info().rss,
            "threads": p.num_threads(), "fds": p.num_fds()}


def measure_native(engine, audios, concurrency):
    """A closed-loop client keeps exactly concurrency native requests in flight."""
    before = resource_snapshot()
    metrics_before = engine.metrics()
    latency, pending, durations = [], {}, []
    peak = before.copy()
    stop = threading.Event()

    def sample():
        while not stop.wait(.02):
            current = resource_snapshot()
            for key in ("rss_bytes", "threads", "fds"):
                peak[key] = max(peak[key], current[key])

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.perf_counter()
    sent = completed = 0
    try:
        while completed < len(audios):
            while sent < len(audios) and len(pending) < concurrency:
                audio, rate = audios[sent]
                timestamp = time.perf_counter()
                request = engine.submit(audio, rate)
                pending[request] = (timestamp, len(audio) / rate)
                sent += 1
            result = engine.poll(0, wait_ms=10)
            if result is None:
                continue
            if result.status:
                raise RuntimeError(f"Native request failed: {result}")
            timestamp, duration = pending.pop(result.request_id)
            latency.append(time.perf_counter() - timestamp)
            durations.append(duration)
            completed += 1
        makespan = time.perf_counter() - started
    finally:
        stop.set()
        sampler.join()
    after = resource_snapshot()
    metrics_after = engine.metrics()
    metrics_delta = {k: metrics_after[k] - metrics_before[k] for k in metrics_after
                     if k not in {"sessions", "pending", "inflight_bytes", "queue_high_water"}}
    return {"requests": completed, "concurrency": concurrency, "makespan_s": makespan,
            "audio_seconds": sum(durations), "throughput_x": sum(durations) / makespan,
            "latency_s": latency, "p50_p95_p99_s": np.percentile(latency, [50, 95, 99]).tolist(),
            "cpu_core_seconds": after["cpu_s"] - before["cpu_s"],
            "mean_cpu_cores": (after["cpu_s"] - before["cpu_s"]) / makespan,
            "peak_resources": peak, "resource_before": before, "resource_after": after,
            "native_metrics_delta": metrics_delta, "native_metrics_after": metrics_after, "errors": 0}


@app.command()
def benchmark(models: list[Path] = typer.Option(..., "--model", envvar="VAD_EVAL_MODELS"),
              dataset: Path = typer.Option(..., envvar="VAD_DATASET_DIR"),
              output: Path = typer.Option(..., envvar="VAD_BENCHMARK_OUTPUT"),
              workers: int = typer.Option(4, min=1),
              cpu_affinity: str = typer.Option("", envvar="VAD_CPU_AFFINITY"),
              worker_cpu_affinity: str = typer.Option("", envvar="VAD_WORKER_CPU_AFFINITY"),
              repetitions: int = typer.Option(4, min=1),
              ready_file: Path = typer.Option(Path("/home/ascend-audio-eval-20261020/device-ready")),
              lock_file: Path = typer.Option(Path("/home/ascend-audio-eval-20261020/device.lock"))):
    """Isolated native PCM -> DSP -> NCNN -> segment end-to-end performance."""
    import fcntl
    from fireredvad_ncnn.native import NativeEngine, _load_library
    if not ready_file.is_file():
        raise RuntimeError("Device/CPU baseline isolation has not been established by the coordinator")
    output.mkdir(parents=True, exist_ok=False)
    with lock_file.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not ready_file.is_file():
            raise RuntimeError("Isolation readiness was revoked")
        if cpu_affinity:
            os.sched_setaffinity(0, {int(i) for i in cpu_affinity.split(",")})
        worker_cpus = sorted({int(i) for i in worker_cpu_affinity.split(",")}) if worker_cpu_affinity else []
        if worker_cpus and len(worker_cpus) < workers:
            raise ValueError("Need at least one assigned CPU per native worker")
        clips = [sf.read(path, dtype="int16") for path, _ in dataset_files(dataset, "test")]
        # Vary file order in deterministic cycles; clips are never concatenated.
        audios = clips * repetitions
        write_json(output / "environment.json", {"platform": platform.platform(), "client_cpu_affinity": sorted(os.sched_getaffinity(0)),
            "worker_cpu_affinity": worker_cpus,
            "python": sys.version, "build": _load_library().frv_build_info().decode(),
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "worktree": subprocess.check_output(["git", "status", "--short"], text=True),
            "dataset_provenance": dataset_manifest(dataset)[0], "split": "test", "rounds": 3,
            "formula": "sum(successful input audio seconds)/wall-clock makespan", "latency": "complete PCM request submission to complete native result; includes queue/DSP/inference/postprocess; excludes file IO/model loading/network",
            "workers": workers, "threads_per_worker": 1, "warmup_requests_per_case": 4,
            "model_files": {str(p): {f: sha256(p/f) for f in ("model.ncnn.param", "model.ncnn.bin", "metadata.json")} for p in models},
            "process_snapshot": [{"pid": p.pid, "name": p.name()} for p in psutil.process_iter()]})
        records = []
        with (output / "raw.jsonl").open("w") as raw:
            for level_index, concurrency in enumerate((1, 2, 4, 8, 16, 32)):
                order = models if level_index % 2 == 0 else list(reversed(models))
                for model in order:
                    before_tids = {t.id for t in psutil.Process().threads()}
                    with NativeEngine(model, cpu_budget=workers, worker_count=workers, threads_per_worker=1) as engine:
                        if worker_cpus:
                            native_tids = sorted({t.id for t in psutil.Process().threads()} - before_tids)
                            if len(native_tids) != workers:
                                raise RuntimeError("Unexpected native worker thread count; affinity assignment is ambiguous")
                            for index, tid in enumerate(native_tids):
                                os.sched_setaffinity(tid, {worker_cpus[index]})
                        for audio, rate in clips[:4]:
                            engine.wait(engine.submit(audio, rate))
                        for round_index in range(3):
                            record = {"model": model.name, "round": round_index,
                                      **measure_native(engine, audios, concurrency)}
                            records.append(record)
                            raw.write(json.dumps(record) + "\n")
                            raw.flush()
                            logger.info("%s concurrency=%d round=%d %.2fx P95=%.3fs", model.name,
                                        concurrency, round_index, record["throughput_x"], record["p50_p95_p99_s"][1])
        summary = []
        for model in models:
            for concurrency in (1, 2, 4, 8, 16, 32):
                group = [r for r in records if r["model"] == model.name and r["concurrency"] == concurrency]
                latencies = sum([r["latency_s"] for r in group], [])
                summary.append({"model": model.name, "concurrency": concurrency, "rounds": len(group),
                    "median_throughput_x": float(np.median([r["throughput_x"] for r in group])),
                    "min_max_throughput_x": [min(r["throughput_x"] for r in group), max(r["throughput_x"] for r in group)],
                    "pooled_p50_p95_p99_s": np.percentile(latencies, [50, 95, 99]).tolist(),
                    "mean_cpu_cores": float(np.mean([r["mean_cpu_cores"] for r in group])),
                    "peak_rss_bytes": max(r["peak_resources"]["rss_bytes"] for r in group), "errors": 0})
        write_json(output / "summary.json", summary)


def measure_grpc(stub, requests, durations, concurrency, server_pid, pool):
    before, peak = resource_snapshot(server_pid), resource_snapshot(server_pid)
    stop = threading.Event()

    def sample():
        while not stop.wait(.02):
            current = resource_snapshot(server_pid)
            for key in ("rss_bytes", "threads", "fds"):
                peak[key] = max(peak[key], current[key])

    def call(index):
        started = time.perf_counter()
        try:
            stub.Detect(requests[index], timeout=30)
            return {"latency_s": time.perf_counter() - started, "audio_seconds": durations[index], "error": None}
        except Exception as exc:
            return {"latency_s": time.perf_counter() - started, "audio_seconds": 0., "error": repr(exc)}

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        results = list(pool.map(call, range(len(requests))))
        makespan = time.perf_counter() - started
    finally:
        stop.set()
        sampler.join()
    after = resource_snapshot(server_pid)
    latencies = [r["latency_s"] for r in results]
    errors = [r["error"] for r in results if r["error"] is not None]
    seconds = sum(r["audio_seconds"] for r in results)
    return {"requests": len(results), "concurrency": concurrency, "makespan_s": makespan,
            "audio_seconds": seconds, "throughput_x": seconds / makespan,
            "latency_s": latencies, "p50_p95_p99_s": np.percentile(latencies, [50, 95, 99]).tolist(),
            "errors": len(errors), "error_details": errors,
            "cpu_core_seconds": after["cpu_s"] - before["cpu_s"],
            "mean_cpu_cores": (after["cpu_s"] - before["cpu_s"]) / makespan,
            "peak_resources": peak, "resource_before": before, "resource_after": after}


@app.command("grpc-benchmark")
def grpc_benchmark(models: list[Path] = typer.Option(..., "--model", envvar="VAD_EVAL_MODELS"),
                   dataset: Path = typer.Option(..., envvar="VAD_DATASET_DIR"),
                   output: Path = typer.Option(..., envvar="VAD_BENCHMARK_OUTPUT"),
                   workers: int = typer.Option(4, min=1),
                   client_cpus: str = typer.Option("52,53,54,55", envvar="VAD_CPU_AFFINITY"),
                   server_cpus: str = typer.Option("48,49,50,51", envvar="VAD_WORKER_CPU_AFFINITY"),
                   ready_file: Path = typer.Option(Path("/home/ascend-audio-eval-20261020/device-ready")),
                   lock_file: Path = typer.Option(Path("/home/ascend-audio-eval-20261020/device.lock"))):
    """Compare original and quantized artifacts over the real Python gRPC boundary."""
    import fcntl
    import grpc
    from grpc_health.v1 import health_pb2, health_pb2_grpc
    from fireredvad_ncnn.proto import ux_vad_pb2 as pb, ux_vad_pb2_grpc as rpc
    if not ready_file.is_file():
        raise RuntimeError("Coordinator isolation readiness is required")
    client_affinity = {int(v) for v in client_cpus.split(",")}
    server_affinity = {int(v) for v in server_cpus.split(",")}
    if client_affinity & server_affinity or len(server_affinity) < workers:
        raise ValueError("Server and client need disjoint CPU affinity and sufficient server CPUs")
    output.mkdir(parents=True, exist_ok=False)
    with lock_file.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not ready_file.is_file():
            raise RuntimeError("Coordinator isolation readiness was revoked")
        os.sched_setaffinity(0, client_affinity)
        requests, durations = [], []
        for path, _ in dataset_files(dataset, "test"):
            audio, rate = sf.read(path, dtype="int16")
            requests.append(pb.DetectRequest(config=pb.DetectionConfig(encoding=1, sample_rate_hertz=rate), audio=audio.tobytes()))
            durations.append(len(audio) / rate)
        requests, durations = requests * 4, durations * 4
        write_json(output / "environment.json", {"platform": platform.platform(), "argv": sys.argv,
            "client_cpus": sorted(client_affinity), "server_cpus": sorted(server_affinity),
            "workers": workers, "threads_per_worker": 1, "rounds": 3, "warmup_requests": 4,
            "dataset_provenance": dataset_manifest(dataset)[0], "split": "test",
            "formula": "successful input audio seconds / makespan; failed requests retained in errors and latency",
            "latency": "unary Detect call start to complete response, includes protobuf serialization, loopback HTTP2/gRPC, Python API, native scheduling/DSP/model/postprocessing; excludes disk IO and cold startup",
            "isolation": "coordinator stopped original services; exclusive shared lock; server48-51/client52-55; other research work0-31",
            "resource_scope": "server process only; peak RSS/threads/FD sampled every20ms",
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "worktree": subprocess.check_output(["git", "status", "--short"], text=True),
            "script_sha256": sha256(Path(__file__)),
            "model_files": {str(p): {f: sha256(p/f) for f in ("model.ncnn.param", "model.ncnn.bin", "metadata.json")} for p in models},
            "process_snapshot": [{"pid": p.pid, "name": p.name()} for p in psutil.process_iter()]})
        records, lifecycle = [], []
        with (output / "raw.jsonl").open("w") as raw:
            for level_index, concurrency in enumerate((1, 2, 4, 8, 16, 32)):
                order = models if level_index % 2 == 0 else list(reversed(models))
                for model in order:
                    with socket.socket() as probe:
                        probe.bind(("127.0.0.1", 0))
                        port = probe.getsockname()[1]
                    address = f"127.0.0.1:{port}"
                    command = ["taskset", "-c", server_cpus, sys.executable, "-m", "fireredvad_ncnn.commands.app", "serve",
                        str(model.resolve()), "--host", "127.0.0.1", "--port", str(port), "--num-threads", str(workers),
                        "--engine-workers", str(workers), "--threads-per-worker", "1", "--log-level", "INFO"]
                    log_path = output / f"{model.name}-{concurrency}.log"
                    with log_path.open("w") as log:
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                            env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))
                        try:
                            with grpc.insecure_channel(address) as channel:
                                health = health_pb2_grpc.HealthStub(channel)
                                deadline = time.monotonic() + 30
                                while True:
                                    if process.poll() is not None:
                                        raise RuntimeError(f"Service exited: {log_path.read_text()}")
                                    try:
                                        state = health.Check(health_pb2.HealthCheckRequest(), timeout=.2)
                                        if state.status == health_pb2.HealthCheckResponse.SERVING:
                                            break
                                    except grpc.RpcError:
                                        pass
                                    if time.monotonic() >= deadline:
                                        raise TimeoutError(f"Service did not become ready: {log_path}")
                                    time.sleep(.05)
                                if psutil.Process(process.pid).children(recursive=True):
                                    raise RuntimeError("Unexpected child processes in single-process native service")
                                stub = rpc.UxVoiceActivityDetectorStub(channel)
                                for request in requests[:4]:
                                    stub.Detect(request, timeout=30)
                                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                                    for round_index in range(3):
                                        record = {"model": model.name, "round": round_index,
                                            **measure_grpc(stub, requests, durations, concurrency, process.pid, pool)}
                                        records.append(record)
                                        raw.write(json.dumps(record) + "\n")
                                        raw.flush()
                                        logger.info("gRPC %s concurrency=%d round=%d %.2fx errors=%d", model.name,
                                            concurrency, round_index, record["throughput_x"], record["errors"])
                                        if record["errors"]:
                                            raise RuntimeError("Formal gRPC failures retained in raw.jsonl")
                        finally:
                            process.terminate()
                            try:
                                process.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=5)
                                raise RuntimeError("gRPC server failed graceful shutdown")
                        if process.returncode:
                            raise RuntimeError(f"gRPC server exit code {process.returncode}: {log_path}")
                    with socket.socket() as probe:
                        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        probe.bind(("127.0.0.1", port))
                    lifecycle.append({"model": model.name, "concurrency": concurrency, "command": command,
                        "pid": process.pid, "exit_code": process.returncode, "port_released": True,
                        "process_exists_after": psutil.pid_exists(process.pid)})
                    write_json(output / "lifecycle.json", lifecycle)
        summary = []
        for model in models:
            for concurrency in (1, 2, 4, 8, 16, 32):
                group = [r for r in records if r["model"] == model.name and r["concurrency"] == concurrency]
                summary.append({"model": model.name, "concurrency": concurrency, "rounds": len(group),
                    "median_throughput_x": float(np.median([r["throughput_x"] for r in group])),
                    "min_max_throughput_x": [min(r["throughput_x"] for r in group), max(r["throughput_x"] for r in group)],
                    "pooled_p50_p95_p99_s": np.percentile(sum([r["latency_s"] for r in group], []), [50, 95, 99]).tolist(),
                    "mean_cpu_cores": float(np.mean([r["mean_cpu_cores"] for r in group])),
                    "peak_rss_bytes": max(r["peak_resources"]["rss_bytes"] for r in group),
                    "errors": sum(r["errors"] for r in group)})
        write_json(output / "summary.json", summary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app()
