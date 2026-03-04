"""
signalbrain.anchor — Public interface to Merkle anchoring.

When running standalone (demo mode), provides a reference
Merkle root builder. In production, delegates to the real anchor module.
"""
import hashlib
import importlib
import sys
from pathlib import Path
from typing import List

_ANCHOR_MODULE = None
_DEMO_MODE = False

try:
    _ANCHOR_MODULE = importlib.import_module("neural_chat.merkle_anchor")
except ImportError:
    root = Path(__file__).resolve().parent.parent.parent
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        _ANCHOR_MODULE = importlib.import_module("neural_chat.merkle_anchor")
    except ImportError:
        _DEMO_MODE = True


def _demo_build_merkle_root(leaves: List[str]) -> str:
    """Reference implementation of Merkle root computation."""
    if not leaves:
        return hashlib.sha256(b"empty").hexdigest()

    layer = [h.encode() if isinstance(h, str) else h for h in leaves]
    while len(layer) > 1:
        next_layer = []
        for i in range(0, len(layer), 2):
            left = layer[i]
            right = layer[i + 1] if i + 1 < len(layer) else left
            combined = hashlib.sha256(left + right).digest()
            next_layer.append(combined)
        layer = next_layer
    return hashlib.sha256(layer[0]).hexdigest() if layer else ""


def build_merkle_root(leaves: List[str]) -> str:
    """Recompute a Merkle root from a list of leaf hashes."""
    if _DEMO_MODE:
        return _demo_build_merkle_root(leaves)
    return _ANCHOR_MODULE.build_merkle_root(leaves)
