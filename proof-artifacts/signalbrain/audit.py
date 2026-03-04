"""
signalbrain.audit — Public interface to the USI Persistence Service.

Re-exports USIPersistenceService for proof-artifact verification scripts.
Requires the full SignalBrain-OS runtime.
"""
import importlib
import sys
from pathlib import Path

_USI_CLASS = None

def _resolve():
    global _USI_CLASS
    if _USI_CLASS is not None:
        return _USI_CLASS

    try:
        mod = importlib.import_module("agi_os_backend.usi_persistence_service")
        _USI_CLASS = mod.USIPersistenceService
        return _USI_CLASS
    except ImportError:
        pass

    root = Path(__file__).resolve().parent.parent.parent
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        mod = importlib.import_module("agi_os_backend.usi_persistence_service")
        _USI_CLASS = mod.USIPersistenceService
        return _USI_CLASS
    except ImportError:
        raise ImportError(
            "SignalBrain-OS runtime not found. "
            "USI audit verification requires the full runtime."
        )


class USIPersistenceService:
    """Proxy class that delegates to the real USIPersistenceService."""
    def __init__(self, *args, **kwargs):
        cls = _resolve()
        self._impl = cls(*args, **kwargs)

    def verify_archive(self, **kwargs):
        return self._impl.verify_archive(**kwargs)

    def shutdown(self):
        return self._impl.shutdown()
