"""Submission pipeline smoke tests.

AC-2: 全 ONNX が制約準拠 (1.44 MB / static shape / no banned op / kernel_time / multi-IO)
AC-5: submission.zip 内 file 構造 (task001.onnx ~ task400.onnx)
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import onnx
import pytest

from neurogolf_2026.build_submission import (
    BANNED_OPS,
    NUM_TASKS,
    ONNX_FILE_LIMIT_BYTES,
    _validate_onnx,
    build,
)
from neurogolf_2026.networks import REGISTRY, get_builder

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def all_models():
    """全 400 task の ONNX bytes (= 1 回だけ生成)。"""
    out = {}
    for n in range(1, NUM_TASKS + 1):
        tk = f"task{n:03d}"
        out[tk] = get_builder(tk)().SerializeToString()
    return out


# --- AC-2 ---
@pytest.mark.unit
def test_onnx_constraints(all_models):
    """全 400 task ONNX が制約準拠 (= AC-2 必須条件)."""
    bad = {}
    for tk, raw in all_models.items():
        v = _validate_onnx(tk, raw)
        if v:
            bad[tk] = v
    assert not bad, f"Constraint violations: {bad}"


@pytest.mark.unit
def test_onnx_size_under_limit(all_models):
    """各 ONNX が 1.44 MB 制限以下 (AC-2 内訳)."""
    over = {tk: len(raw) for tk, raw in all_models.items() if len(raw) > ONNX_FILE_LIMIT_BYTES}
    assert not over, f"Files over 1.44 MB: {over}"


@pytest.mark.unit
def test_no_banned_ops(all_models):
    """禁止 op (Loop/Scan/NonZero/Unique/Script/Function/Compress) 不使用."""
    for tk, raw in all_models.items():
        model = onnx.load_from_string(raw)
        op_types = {n.op_type for n in model.graph.node}
        banned = op_types & BANNED_OPS
        assert not banned, f"{tk}: banned op {banned}"


@pytest.mark.unit
def test_no_kernel_time_in_names(all_models):
    """tensor / node 名に 'kernel_time' 不使用 (2026-05-06 ban)."""
    for tk, raw in all_models.items():
        model = onnx.load_from_string(raw)
        names = (
            [n.name for n in model.graph.node]
            + [o for n in model.graph.node for o in n.output]
            + [i.name for i in model.graph.initializer]
        )
        for name in names:
            assert "kernel_time" not in name, f"{tk}: kernel_time in '{name}'"


@pytest.mark.unit
def test_single_input_output(all_models):
    """Multi-input / multi-output graph 不使用 (2026-05-06 ban)."""
    for tk, raw in all_models.items():
        model = onnx.load_from_string(raw)
        assert len(model.graph.input) == 1, f"{tk}: {len(model.graph.input)} inputs"
        assert len(model.graph.output) == 1, f"{tk}: {len(model.graph.output)} outputs"


@pytest.mark.unit
def test_static_input_shape(all_models):
    """input shape が [1, 10, 30, 30] static (= AC-2 内訳)."""
    expected = [1, 10, 30, 30]
    for tk, raw in all_models.items():
        model = onnx.load_from_string(raw)
        inp = model.graph.input[0]
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        assert dims == expected, f"{tk}: input shape {dims}, expected {expected}"


# --- AC-5 ---
@pytest.mark.integration
def test_zip_structure(tmp_path):
    """submission.zip が task001.onnx ~ task400.onnx を含む (= AC-5)."""
    out = tmp_path / "submission.zip"
    violations = build(out_path=out, num_tasks=NUM_TASKS, strict=True, verbose=False)
    assert not violations
    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        names = sorted(zf.namelist())
    expected = [f"task{n:03d}.onnx" for n in range(1, NUM_TASKS + 1)]
    assert names == expected, f"zip names mismatch: {set(names) ^ set(expected)}"
    # zip overhead 込みでも妥当な size (= 各 file 数 KB なので合計 < 数 MB)
    assert out.stat().st_size < 10 * 1024 * 1024, "submission.zip too large"


@pytest.mark.integration
def test_build_idempotent(tmp_path):
    """build を 2 回実行して同 byte の zip が出力される (= AC-6 冪等性)."""
    out1 = tmp_path / "sub1.zip"
    out2 = tmp_path / "sub2.zip"
    build(out_path=out1, num_tasks=10, strict=True)
    build(out_path=out2, num_tasks=10, strict=True)
    # zip metadata (timestamp) で完全 byte 一致は無理なので、内部 ONNX は一致を確認
    with zipfile.ZipFile(out1) as zf1, zipfile.ZipFile(out2) as zf2:
        for n in zf1.namelist():
            assert zf1.read(n) == zf2.read(n), f"{n}: bytes differ between builds"


# --- AC-1 (sanity for task276 only, not all 400) ---
@pytest.mark.integration
def test_task276_registered():
    """AC-1 baseline task が registry に登録されている."""
    assert "task276" in REGISTRY
    model = get_builder("task276")()
    assert isinstance(model, onnx.ModelProto)
