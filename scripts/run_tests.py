"""Minimal test runner for machines without pytest (the server): runs every fixture-free test_* function.

    PYTHONPATH=src python scripts/run_tests.py tests/test_scout.py [tests/test_structure.py ...]
"""
import importlib.util
import inspect
import sys
import time
import traceback
from pathlib import Path

failed = passed = skipped = 0
for path in sys.argv[1:]:
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_") or fn.__module__ != mod.__name__:
            continue
        if inspect.signature(fn).parameters:
            skipped += 1
            print(f"SKIP {path}::{name} (needs fixtures)")
            continue
        t0 = time.time()
        try:
            fn()
            passed += 1
            print(f"PASS {path}::{name} ({time.time() - t0:.1f} s)")
        except Exception:
            failed += 1
            print(f"FAIL {path}::{name}")
            traceback.print_exc()
print(f"{passed} passed, {failed} failed, {skipped} skipped")
sys.exit(1 if failed else 0)
