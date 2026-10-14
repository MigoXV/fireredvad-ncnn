"""Offline conversion; importing the serving package never imports Torch."""
from __future__ import annotations

import hashlib
import json
import shutil
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pnnx
import torch
from safetensors.torch import load_file

from .torch_model import FireRedVadStreamModule


def load_reference(source: Path):
    config = json.loads((source / "config.json").read_text())
    required = dict(idim=80, odim=1, R=8, P=128, N1=20, S1=1, N2=0)
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError(f"Unsupported stream model configuration; expected {required}")
    model = FireRedVadStreamModule.from_config(SimpleNamespace(**config))
    state = load_file(str(source / "model.safetensors"), device="cpu")
    model.load_state_dict({key.removeprefix("vad."): value for key, value in state.items()}, strict=True)
    return model.eval(), config


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, *caches):
        probs, states = self.model(features, caches=caches)
        return (probs, *states)


def export_model(source: Path, output: Path, chunk_frames: int = 10):
    if not 1 <= chunk_frames <= 19:
        raise ValueError("chunk_frames must be between 1 and 19")
    source, output = source.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty to avoid mixing model versions")
    model, config = load_reference(source)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    inputs = (torch.zeros(1, chunk_frames, 80), *(torch.zeros(1, 128, 19) for _ in range(8)))
    with torch.inference_mode():
        pnnx.export(ExportWrapper(model), str(output / "model.pt"), inputs=inputs,
                    fp16=False, device="cpu")
    for name in ["model.ncnn.param", "model.ncnn.bin"]:
        if not (output / name).is_file():
            raise RuntimeError(f"PNNX did not produce {name}")
    shutil.copyfile(source / "cmvn.ark", output / "cmvn.ark")
    metadata = {
        "format_version": 1, "chunk_frames": chunk_frames,
        "feature_dim": 80, "cache_shape": [1, 128, 19], "cache_count": 8,
        "inputs": [f"in{i}" for i in range(9)], "outputs": [f"out{i}" for i in range(9)],
        "dtype": "float32", "source_config": config,
        "source_sha256": hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest(),
        "versions": {name: version(name) for name in ["torch", "pnnx"]},
        "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                  for name in ["model.ncnn.param", "model.ncnn.bin", "cmvn.ark"]},
    }
    from fireredvad_ncnn.native import _load_library
    metadata["versions"]["native_engine"] = _load_library().frv_build_info().decode()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
