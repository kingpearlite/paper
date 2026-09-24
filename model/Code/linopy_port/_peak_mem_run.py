"""Runs a Python script in this process and reports the process's peak working set (Windows),
which includes Gurobi's memory because linopy calls gurobipy in-process. Used for the laptop
smoke-test RAM figures in docs/microgridspy_pipeline_usage.tex (2026-09-11); the linopy scripts'
own resource monitor polls only every 60 s and misses short runs.

    python _peak_mem_run.py linopy_free_sizing_crosscheck.py      (env vars as for the script)
"""
import os
import runpy
import sys
import time

import psutil

script = os.path.abspath(sys.argv[1])
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(script))
t0 = time.time()
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit:
    pass
finally:
    mi = psutil.Process().memory_info()
    print(f"[peak memory] process peak working set {mi.peak_wset / 2**30:.2f} GiB, "
          f"wall {time.time() - t0:.0f} s", flush=True)
