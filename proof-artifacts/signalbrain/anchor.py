"""
signalbrain.anchor — Public interface to Merkle anchoring.

Re-exports build_merkle_root for proof-artifact verification scripts.
Requires the full SignalBrain-OS runtime.
"""
import importlib
import sys
from pathlib import Path

_ANCHOR_MODULE = None

def _resolve():
    global _ANCHOR_MODULE
    if _ANCHOR_MODULE is not None:
        return _ANCHOR_MODULE

    try:
        _ANCHOR_MODULE = importlib.import_module("neural_chat.merkle_anchor")
        return _ANCHOR_MODULE
    except ImportError:
        pass

    root = Path(__file__).resolve().parent.parent.parent
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        _ANCHOR_MODULE = importlib.import_module("neural_chat.merkle_anchor")
        return _ANCHOR_MODULE
    except ImportError:
        raise ImportError(
            "SignalBrain-OS runtime not found. "
            "Merkle anchor verification requires the full runtime."
        )


def build_merkle_root(leaves):
    """Recompute a Merkle root from a list of leaf hashes."""
    mod = _resolve()
    return mod.build_merkle_root(leaves)
