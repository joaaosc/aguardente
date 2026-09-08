"""Run the working tree using an existing pipeline Python environment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(line_buffering=True)
from aguardente.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
