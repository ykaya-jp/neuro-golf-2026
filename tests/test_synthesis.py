"""LLM synthesis pipeline tests (= exp002 AC-1 / AC-2 / AC-3 / AC-5)."""
from __future__ import annotations

from pathlib import Path

import pytest

from neurogolf_2026.synthesis import DummyClient, run_synthesis
from neurogolf_2026.synthesis.extractor import ExtractError, extract_weight_fn
from neurogolf_2026.synthesis.runner import run_weight_fn


# --- AC-2: extractor 安全性 ---
@pytest.mark.unit
class TestExtractorSafety:
    """LLM 出力の unsafe code を AST whitelist で reject."""

    @pytest.mark.parametrize("forbidden", [
        # eval / exec
        '```python\ndef weight_fn(channel_out, channel_in, kernel_coord):\n    eval("1")\n    return 0.0\n```',
        '```python\ndef weight_fn(channel_out, channel_in, kernel_coord):\n    exec("x")\n    return 0.0\n```',
        # __import__
        '```python\ndef weight_fn(channel_out, channel_in, kernel_coord):\n    x = __import__\n    return 0.0\n```',
        # open / file I/O
        '```python\ndef weight_fn(channel_out, channel_in, kernel_coord):\n    open("/etc/passwd")\n    return 0.0\n```',
        # subprocess
        '```python\nimport subprocess\ndef weight_fn(channel_out, channel_in, kernel_coord):\n    return 0.0\n```',
        # os
        '```python\nimport os\ndef weight_fn(channel_out, channel_in, kernel_coord):\n    return 0.0\n```',
        # attribute access
        '```python\ndef weight_fn(channel_out, channel_in, kernel_coord):\n    return channel_out.bit_length()\n```',
        # function call (= ast.Call 禁止)
        '```python\ndef weight_fn(channel_out, channel_in, kernel_coord):\n    return float(channel_out)\n```',
    ])
    def test_unsafe_code_rejected(self, forbidden):
        with pytest.raises(ExtractError):
            extract_weight_fn(forbidden)

    def test_no_code_block_rejected(self):
        with pytest.raises(ExtractError):
            extract_weight_fn("just plain text without code block")

    def test_wrong_function_name_rejected(self):
        bad = '```python\ndef foo(channel_out, channel_in, kernel_coord):\n    return 0.0\n```'
        with pytest.raises(ExtractError):
            extract_weight_fn(bad)

    def test_wrong_signature_rejected(self):
        bad = '```python\ndef weight_fn(a, b):\n    return 0.0\n```'
        with pytest.raises(ExtractError):
            extract_weight_fn(bad)

    def test_safe_code_accepted(self):
        safe = '```python\ndef weight_fn(channel_out, channel_in, kernel_coord):\n    if kernel_coord != (0, 0):\n        return 0.0\n    if channel_out == channel_in:\n        return 1.0\n    return 0.0\n```'
        fn = extract_weight_fn(safe)
        assert callable(fn)
        # 動作確認
        assert fn(0, 0, (0, 0)) == 1.0
        assert fn(0, 1, (0, 0)) == 0.0
        assert fn(0, 0, (1, 1)) == 0.0


# --- AC-3: runner banned op check ---
@pytest.mark.integration
class TestRunnerBannedOp:
    """Runner が banned op (Loop / Compress 等) を含む ONNX を reject."""

    def test_runner_accepts_clean_weight_fn(self):
        """task276 reference を runner に通すと accepted = True."""
        def weight_fn(channel_out, channel_in, kernel_coord):
            if kernel_coord != (0, 0):
                return 0.0
            if channel_out == 6:
                return 0.0
            if channel_out == 2:
                if channel_in == 2 or channel_in == 6:
                    return 1.0
                return 0.0
            if channel_out == channel_in:
                return 1.0
            return 0.0
        rr = run_weight_fn("task276", weight_fn, kernel_size=1)
        assert rr.accepted, f"unexpected reject: {rr}"
        assert rr.functional_correct
        assert rr.scorer_ok
        assert rr.constraint_violations == []

    def test_runner_rejects_banned_op_via_synthetic(self, tmp_path):
        """合成的に banned op を含む ONNX を作って _validate_onnx 経由で reject されることを確認.

        runner が直接 banned op を含む ONNX を作る経路は無い (= single_layer_conv2d_network は
        Conv のみ生成) ので、build_submission._validate_onnx の負荷検査として直接 test。
        """
        import onnx
        from neurogolf_2026.build_submission import _validate_onnx

        # synthetic な banned op (Loop) を含む ONNX を構成
        x = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10, 30, 30])
        y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 10, 30, 30])
        loop_node = onnx.helper.make_node("Loop", ["input"], ["output"], name="bad_loop")
        graph = onnx.helper.make_graph([loop_node], "g", [x], [y])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 10)])
        violations = _validate_onnx("synthetic", model.SerializeToString())
        assert any("Loop" in v for v in violations), f"expected Loop violation, got {violations}"


# --- AC-1 + AC-5: pipeline end-to-end with DummyClient ---
@pytest.mark.integration
class TestPipelineDummy:
    def test_pipeline_dummy_end_to_end(self, tmp_path):
        """DummyClient で 3 task 試行、results.json + per-task raw が生成される."""
        out_dir = tmp_path / "synth_run"
        client = DummyClient()
        results = run_synthesis(client, ["task276", "task016", "task001"], out_dir)

        # results.json 生成
        results_path = out_dir / "results.json"
        assert results_path.exists()
        import json
        summary = json.loads(results_path.read_text())
        assert summary["_meta"]["task_count"] == 3
        assert summary["_meta"]["accepted_count"] >= 1  # task276 は dummy で通る

        # task276 が accepted
        task276 = next(r for r in results if r.task_id == "task276")
        assert task276.accepted, f"task276 not accepted: {task276}"
        assert task276.run_result["functional_correct"] is True

        # zero-template 系 (task016 / task001) は accepted=False (functional incorrect)
        for tk in ("task016", "task001"):
            r = next(r for r in results if r.task_id == tk)
            assert not r.accepted

        # raw 出力 file 生成
        assert (out_dir / "task276_raw.txt").exists()
