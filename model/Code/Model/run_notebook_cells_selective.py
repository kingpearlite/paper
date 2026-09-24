#!/usr/bin/env python
"""Execute an explicit list of MicroGrids.ipynb code cells, in-process, in the order given.

Same in-process exec approach as run_notebook_cells.py (nbconvert/nbclient aren't installed in
myenv) but for a non-contiguous cell range -- e.g. setup cells + the fixed-design-verification
cells while skipping the free-sizing solve cells in between, for a quick regression check against
a design whose free-sizing result is already on record.

Usage: python run_notebook_cells_selective.py <notebook.ipynb> <comma-separated cell indices>
"""
import json
import sys
from pathlib import Path

def main():
    nb_path = Path(sys.argv[1])
    indices = [int(x) for x in sys.argv[2].split(",")]
    nb = json.loads(nb_path.read_text())
    cells = nb["cells"]

    ns = {"__name__": "__main__"}
    for i in indices:
        cell = cells[i]
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
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
