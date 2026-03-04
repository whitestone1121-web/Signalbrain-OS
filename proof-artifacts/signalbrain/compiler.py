"""
signalbrain.compiler — Public interface to the Apex17 Policy Compiler.

This module re-exports the two public functions used by proof-artifact scripts:
  - draft(snapshot, agent_name) → PolicyDecision | None
  - get_stats() → dict

Requires the full SignalBrain-OS runtime to be installed.
"""
import importlib
import sys
from pathlib import Path

# Resolve the private runtime module
_RUNTIME_MODULE = None

def _resolve_runtime():
    global _RUNTIME_MODULE
    if _RUNTIME_MODULE is not None:
        return _RUNTIME_MODULE

    # Try standard install first
    try:
        _RUNTIME_MODULE = importlib.import_module("neural_chat.apex17_policy_compiler")
        return _RUNTIME_MODULE
    except ImportError:
        pass

    # Fallback: resolve from source tree relative to proof-artifacts/
    root = Path(__file__).resolve().parent.parent.parent
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        _RUNTIME_MODULE = importlib.import_module("neural_chat.apex17_policy_compiler")
        return _RUNTIME_MODULE
    except ImportError:
        raise ImportError(
            "SignalBrain-OS runtime not found. "
            "These proof artifacts require the full SignalBrain-OS runtime. "
            "Run inside the project root or Docker container."
        )


def draft(snapshot, agent_name):
    """Evaluate a policy draft for a market snapshot and agent.
    
    Returns a PolicyDecision with .signal_action, .confidence, .policy_hash
    or None if the compiler does not produce a bypass for this input.
    """
    mod = _resolve_runtime()
    return mod.draft(snapshot, agent_name)


def get_stats():
    """Return cumulative Apex17 compiler statistics."""
    mod = _resolve_runtime()
    return mod.get_stats()
