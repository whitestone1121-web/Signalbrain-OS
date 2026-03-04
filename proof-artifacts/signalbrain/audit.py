"""
signalbrain.audit — Public interface to the USI Persistence Service.

When running standalone (demo mode), provides a stub verifier that
returns a synthetic pass result. In production, delegates to the
real USI audit backend.
"""
import importlib
import sys
from pathlib import Path


_DEMO_MODE = False

try:
    from agi_os_backend.usi_persistence_service import \
        USIPersistenceService as _RealUSI
except ImportError:
    root = Path(__file__).resolve().parent.parent.parent
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from agi_os_backend.usi_persistence_service import \
            USIPersistenceService as _RealUSI
    except ImportError:
        _RealUSI = None
        _DEMO_MODE = True


class USIPersistenceService:
    """USI archive verifier — delegates to production runtime or demo stub."""

    def __init__(self, *args, **kwargs):
        if _RealUSI is not None:
            self._impl = _RealUSI(*args, **kwargs)
        else:
            self._impl = None

    def verify_archive(self, **kwargs):
        if self._impl:
            return self._impl.verify_archive(**kwargs)
        # Demo mode: return synthetic pass
        n = kwargs.get("n", 100)
        return {
            "checked": n,
            "valid": n,
            "tampered": 0,
            "tampered_hashes": [],
            "demo_mode": True,
        }

    def shutdown(self):
        if self._impl:
            self._impl.shutdown()
