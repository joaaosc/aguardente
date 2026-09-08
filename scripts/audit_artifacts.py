"""Run export/runtime audit commands using the checked-out package."""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
command = sys.argv.pop(1)
if command not in {"local_export", "runtime_check"}:
    raise SystemExit("Use local_export or runtime_check")
runpy.run_module(f"aguardente.{command}", run_name="__main__")
