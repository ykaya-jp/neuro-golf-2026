"""ONNX builder helper.

公式 `data/raw/neurogolf_utils/neurogolf_utils.py:single_layer_conv2d_network` を
import 可能にし、independent 実装で重複を避ける (= 主催者 helper との一貫性)。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UTILS_PATH = _REPO_ROOT / "data" / "raw" / "neurogolf_utils" / "neurogolf_utils.py"


def _load_neurogolf_utils():
    if "neurogolf_utils" in sys.modules:
        return sys.modules["neurogolf_utils"]
    if not _UTILS_PATH.exists():
        raise FileNotFoundError(
            f"neurogolf_utils.py not found at {_UTILS_PATH}. "
            "Run `uv run kaggle competitions download -c neurogolf-2026 -p data/raw && "
            "unzip data/raw/*.zip -d data/raw/` to fetch it."
        )
    spec = importlib.util.spec_from_file_location("neurogolf_utils", _UTILS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load spec from {_UTILS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["neurogolf_utils"] = module
    spec.loader.exec_module(module)
    return module


utils = _load_neurogolf_utils()
single_layer_conv2d_network = utils.single_layer_conv2d_network
verify_subset = utils.verify_subset
score_network = utils.score_network
load_examples = utils.load_examples
convert_to_numpy = utils.convert_to_numpy
