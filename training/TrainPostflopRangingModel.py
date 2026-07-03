from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    if "--model-scope" not in sys.argv:
        sys.argv.extend(["--model-scope", "postflop"])
    runpy.run_path(
        str(Path(__file__).resolve().parent / "training" / "TrainRangingModel.py"),
        run_name="__main__",
    )
