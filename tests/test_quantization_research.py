"""Failure checks for the additive INT8 research tools; no accelerator required."""
from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "quantization_research", Path(__file__).parents[1] / "scripts/quantization_research.py")
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


def test_labels_reject_gaps_overlaps_and_nonbinary_values(tmp_path):
    path = tmp_path / "labels.scv"
    for value in ("id,0,1,1,2,3,0", "id,0,2,1,1,3,0", "id,0,1,2", "id,0,nan,1"):
        path.write_text(value)
        with pytest.raises(ValueError):
            research.read_labels(path)


def test_frame_scoring_uses_half_open_intervals_and_known_region():
    labels = [(0., 1., 0), (1., 2., 1)]
    reference, valid = research.binary_at_times(labels, np.array([.5, 1., 1.5, 2.]))
    assert reference.tolist() == [False, True, True, False]
    assert valid.tolist() == [True, True, True, False]
    counts = research.confusion(reference[valid], [False, True, False])
    scores = research.metrics(counts)
    assert scores["precision"] == 1.
    assert scores["recall"] == .5
    assert scores["miss_rate"] == .5
    assert scores["false_alarm_rate"] == 0.
    assert scores["f1"] == pytest.approx(2/3)


def test_rttm_speaker_overlap_is_scored_once_and_silence_is_scored():
    labels = research.speech_union_labels([(1., 3.), (2., 4.), (5., 6.)], 7.)
    assert labels == [(0., 1., 0), (1., 4., 1), (4., 5., 0), (5., 6., 1), (6., 7., 0)]
    reference, valid = research.binary_at_times(labels, np.arange(7) + .5)
    assert valid.all()
    assert reference.sum() == 4


def test_gemm_rewrite_converts_bias_tag_and_preserves_weights():
    param = "7777517\n2 2\nInput in0 0 1 in0\nGemm fc 1 1 in0 out0 2=0 3=1 4=0 5=1 6=1 7=10 8=2 9=3 10=4\n"
    weight_bytes = struct.pack("<I6f", 0, 1, 2, 3, 4, 5, 6)
    binary = weight_bytes + struct.pack("<I2f", 0, .25, -.5)
    rewritten, converted, names = research.rewrite_gemm_to_innerproduct(param, binary)
    assert "InnerProduct fc 1 1 in0 out0 0=2 1=1 2=6" in rewritten
    assert names == ["fc"]
    assert converted == weight_bytes + struct.pack("<2f", .25, -.5)
    # Execute the two mathematical forms on a multi-row input.
    weights = np.frombuffer(converted, "<f4", count=6, offset=4).reshape(2, 3)
    bias = np.frombuffer(converted, "<f4", count=2, offset=28)
    inputs = np.array([[1, 3, -1], [0, 1, .5]], np.float32)
    assert np.allclose(inputs @ weights.T + bias, [[4.25, 12.5], [3.75, 7.5]])


def test_gemm_rewrite_rejects_unsupported_and_truncated_layout():
    param = "7777517\n1 1\nGemm fc 1 1 in0 out0 2=0 3=1 4=0 5=1 6=1 7=10 8=2 9=3 10=-1\n"
    binary = struct.pack("<I6f", 0, 1, 2, 3, 4, 5, 6)
    for bad in (binary[:-1], binary + b"x", struct.pack("<I", 0x01306B47) + binary[4:]):
        with pytest.raises(ValueError):
            research.rewrite_gemm_to_innerproduct(param, bad)
    with pytest.raises(ValueError):
        research.rewrite_gemm_to_innerproduct(param.replace("3=1", "3=0"), binary)


def test_constant_a_rewrite_preserves_binary_and_transposes_the_equation():
    param = "7777517\n1 1\nGemm fc 1 1 in0 out0 2=0 3=1 4=0 5=1 6=1 7=10 8=2 9=3 10=4\n"
    binary = struct.pack("<I6fI2f", 0, 1, 2, 3, 4, 5, 6, 0, .25, -.5)
    rewritten, same_binary, _ = research.rewrite_gemm_constant_a(param, binary)
    assert "4=1 5=0 6=1 7=2 8=10 9=3 10=1 14=1" in rewritten
    assert same_binary == binary
    x = np.array([[1, 3, -1], [0, 1, .5]], np.float32)
    w = np.frombuffer(binary, "<f4", count=6, offset=4).reshape(2, 3)
    b = np.frombuffer(binary, "<f4", count=2, offset=32)
    np.testing.assert_array_equal(x @ w.T + b, (w @ x.T + b[:, None]).T)
