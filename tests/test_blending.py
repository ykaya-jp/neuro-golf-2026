"""Blending pipeline tests (= exp004 AC-1 / AC-2 / AC-3 / AC-5)."""
from __future__ import annotations

import zipfile
from pathlib import Path

import onnx
import pytest

from neurogolf_2026.blending import build_blended_zip, select_per_task
from neurogolf_2026.blending.argmin import Source
from neurogolf_2026.blending.source_pool import DEFAULT_SOURCES, inventory


REPO_ROOT = Path(__file__).resolve().parents[1]


# ----- AC-1 -----
@pytest.mark.integration
def test_source_pool_inventory():
    """各 source が data/external/ 下の path に解決され、 ONNX file 数 / size が取れる."""
    inv = inventory(REPO_ROOT, sources=DEFAULT_SOURCES)
    # 全 7 source が登録されている
    assert set(inv.keys()) == {s.name for s in DEFAULT_SOURCES}
    # 少なくとも 1 source が 200+ task を持つ (= konbu17 系)
    assert any(len(v) >= 200 for v in inv.values()), \
        f"no source has >= 200 tasks: {[(k, len(v)) for k, v in inv.items()]}"
    # 全 task カバー (= union が 400 件) を確認
    all_tasks = set()
    for per_task in inv.values():
        all_tasks.update(per_task.keys())
    assert len(all_tasks) >= 400, f"only {len(all_tasks)} tasks covered"


# ----- AC-2 (synthetic source で argmin の挙動を確認) -----
@pytest.mark.unit
def test_argmin_selects_min_size(tmp_path):
    """合成 source 2 件で argmin が小さい方を選ぶ."""
    import onnx as _onnx

    # 2 種類 size の異なる合成 ONNX を作る (= 1×1 conv identity)
    def make_synthetic_onnx(out_dir: Path, scale: int) -> Path:
        weights = [scale * 1.0 if i == j else 0.0
                   for i in range(10) for j in range(10) for _ in range(1)]
        x = _onnx.helper.make_tensor_value_info("input", _onnx.TensorProto.FLOAT, [1, 10, 30, 30])
        y = _onnx.helper.make_tensor_value_info("output", _onnx.TensorProto.FLOAT, [1, 10, 30, 30])
        w = _onnx.helper.make_tensor("W", _onnx.TensorProto.FLOAT, [10, 10, 1, 1], weights)
        node = _onnx.helper.make_node("Conv", ["input", "W"], ["output"],
                                     kernel_shape=[1, 1], pads=[0, 0, 0, 0])
        graph = _onnx.helper.make_graph([node], "g", [x], [y], [w])
        model = _onnx.helper.make_model(graph, opset_imports=[_onnx.helper.make_opsetid("", 10)])
        # padding 違いで size が変わるので、 仮 byte 違いを extra metadata で表現
        for k in range(scale):
            model.metadata_props.add(key=f"pad{k}", value="x" * 100)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "task001.onnx"
        path.write_bytes(model.SerializeToString())
        return path

    # 2 source を作成
    big_dir = tmp_path / "big"
    small_dir = tmp_path / "small"
    big_path = make_synthetic_onnx(big_dir, scale=20)  # 大量 metadata
    small_path = make_synthetic_onnx(small_dir, scale=1)  # 少量 metadata
    assert small_path.stat().st_size < big_path.stat().st_size

    # Source 定義 (= path_template は task001.onnx を直接指す)
    big_src = Source(name="big", path_template=str(big_dir.relative_to(tmp_path)) + "/task{n:03d}.onnx")
    small_src = Source(name="small", path_template=str(small_dir.relative_to(tmp_path)) + "/task{n:03d}.onnx")

    sel = select_per_task(1, [big_src, small_src], repo_root=tmp_path)
    assert sel is not None
    name, raw = sel
    assert name == "small"


# ----- AC-3 -----
@pytest.mark.unit
def test_self_preferred_when_smaller():
    """self_raw が一番小さければ self が選ばれる."""
    import onnx as _onnx
    weights = [1.0 if i == j else 0.0 for i in range(10) for j in range(10)]
    x = _onnx.helper.make_tensor_value_info("input", _onnx.TensorProto.FLOAT, [1, 10, 30, 30])
    y = _onnx.helper.make_tensor_value_info("output", _onnx.TensorProto.FLOAT, [1, 10, 30, 30])
    w = _onnx.helper.make_tensor("W", _onnx.TensorProto.FLOAT, [10, 10, 1, 1], weights)
    node = _onnx.helper.make_node("Conv", ["input", "W"], ["output"], kernel_shape=[1, 1], pads=[0]*4)
    graph = _onnx.helper.make_graph([node], "g", [x], [y], [w])
    model = _onnx.helper.make_model(graph, opset_imports=[_onnx.helper.make_opsetid("", 10)])
    self_raw = model.SerializeToString()
    self_size = len(self_raw)

    # Source は実在しないので 候補は self のみ
    sel = select_per_task(
        1, sources=[],  # 空 source list
        repo_root=Path("/tmp/nonexistent"),
        self_raw=self_raw, self_size=self_size,
    )
    assert sel is not None
    name, raw = sel
    assert name == "self"
    assert raw == self_raw


# ----- AC-5 -----
@pytest.mark.integration
def test_blended_submission_constraints(tmp_path):
    """build_blended_zip で構築した zip が制約準拠 (= 400 file, 個別 1.44MB 内, banned op なし)."""
    out = tmp_path / "submission.zip"
    summary = build_blended_zip(out, repo_root=REPO_ROOT)
    assert out.exists()

    # 400 files
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert len(names) == 400, f"expected 400, got {len(names)}"
        # 命名 task001.onnx ~ task400.onnx
        expected = {f"task{n:03d}.onnx" for n in range(1, 401)}
        assert set(names) == expected, f"name mismatch"

        # 個別 size limit (= 1.44 MB)
        for n in names:
            sz = zf.getinfo(n).file_size
            assert sz <= int(1.44 * 1024 * 1024), f"{n}: {sz} byte > 1.44 MB"

        # banned op check (= 採用された ONNX に scorer-poison ナシ確認は別途)
        # 本 test は build_blended_zip 内の _validate_onnx を信用
    assert summary["_meta"]["fallback_count"] >= 0
    # 全体 size < 100 MB
    assert out.stat().st_size < 100 * 1024 * 1024
