#!/usr/bin/env python
"""Execute a subset of MicroGrids.ipynb's code cells sequentially, in-process, in cell order.

Used instead of `jupyter nbconvert --execute` because nbconvert/nbclient are not installed in
the `myenv` conda environment (only ipykernel/jupyter_client are). This does the same thing
nbconvert would do for a linear "Run All" -- exec each code cell's source into one shared
namespace, in notebook order -- for the cell range this HPC batch actually needs (Path setup
through Timing, i.e. skipping the separate Fixed-design-verification section at the bottom,
per the checklist's established procedure of putting MGPY_FIX_* pins in the main Run
Configuration cell instead and re-running the same free-sizing cell range).

Usage: python run_notebook_cells.py <notebook.ipynb> <max_cell_index_inclusive>
"""
import json
import sys
from pathlib import Path

def main():
    nb_path = Path(sys.argv[1])
    max_idx = int(sys.argv[2])
    nb = json.loads(nb_path.read_text())
    cells = nb["cells"]

    ns = {"__name__": "__main__"}
    for i, cell in enumerate(cells):
        if i > max_idx:
            break
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
