#!/usr/bin/env python
"""Execute only the cells needed for the fixed-design verification section of
MicroGrids.ipynb, SKIPPING the free-sizing model-creation/solve/results/plots/timing cells
(11/13/15/17) -- those aren't needed to reach the fixed-design cells (19/20), which build
and solve a fresh, independent `model_fix` instance. This exists purely to avoid re-running
a ~40-minute free-sizing solve just to reproduce a fixed-design verification.

Usage: python run_notebook_cells_selective_fixdesign_only.py <notebook.ipynb>
Runs cells [2, 4, 5, 7, 9, 19, 20] in that order, in one shared namespace.
"""
import json
import sys
from pathlib import Path

CELL_INDICES = [2, 4, 5, 7, 9, 19, 20]

def main():
    nb_path = Path(sys.argv[1])
    nb = json.loads(nb_path.read_text())
    cells = nb["cells"]

    ns = {"__name__": "__main__"}
    for i in CELL_INDICES:
        cell = cells[i]
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if not src.strip():
            continue
        print(f"\n{'='*80}\n=== Executing cell {i} ===\n{'='*80}", flush=True)
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
        except SystemExit as e:
            print(f"SystemExit from cell {i}: {e}", flush=True)
            raise
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"\n!!! Cell {i} raised -- aborting run !!!", flush=True)
            raise

if __name__ == "__main__":
    main()
