#!/usr/bin/env python3
"""Minimal offline runner when pytest is not installed (不升环境)."""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-9):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, other):
        try:
            return abs(float(other) - float(self.expected)) <= max(
                self.abs, self.rel * abs(float(self.expected))
            )
        except (TypeError, ValueError):
            return False

    def __repr__(self):
        return f"approx({self.expected})"


def _ensure_pytest_shim() -> None:
    try:
        import pytest  # noqa: F401
        return
    except ImportError:
        pass
    mod = ModuleType("pytest")

    class RaisesContext:
        def __init__(self, exc_type, match=None):
            self.exc_type = exc_type
            self.match = match
            self.exc = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                raise AssertionError(f"DID NOT RAISE {self.exc_type}")
            if not issubclass(exc_type, self.exc_type):
                return False
            if self.match and self.match not in str(exc):
                raise AssertionError(f"pattern {self.match!r} not in {exc!r}")
            self.exc = exc
            return True

    mod.raises = lambda exc_type, match=None: RaisesContext(exc_type, match=match)
    mod.approx = lambda expected, rel=1e-6, abs=1e-9: _Approx(expected, rel=rel, abs=abs)
    sys.modules["pytest"] = mod


def main() -> int:
    _ensure_pytest_shim()
    # Import after shim so test module can `import pytest`
    from tests import test_step_b as t

    tests = [getattr(t, n) for n in dir(t) if n.startswith("test_")]
    failed = 0
    for fn in tests:
        name = fn.__name__
        try:
            if "tmp_path" in fn.__code__.co_varnames:
                import tempfile
                from pathlib import Path as P

                with tempfile.TemporaryDirectory() as td:
                    fn(P(td))
            else:
                fn()
            print(f"PASS {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
