import os
import sys
import time
from pathlib import Path

def _find_model_dir(start: Path, marker: str = "MicroGrids.py", max_up: int = 5) -> Path:
    """Search `start` and its ancestors (bounded) for the folder containing MicroGrids.py.

    Portable across machines/OSes on purpose: this notebook is meant to run unmodified
    both on a local checkout and on HPC (e.g. via a Strudel VS Code Server session),
    where the repo lives at a different absolute path. No machine-specific path is
    hardcoded here -- if the search fails, fix your kernel's working directory (or edit
    MODEL_DIR below by hand) rather than relying on a guessed fallback.
    """
    candidate = start
    for _ in range(max_up + 1):
        if (candidate / marker).exists():
            return candidate
        candidate = candidate.parent
    raise FileNotFoundError(
        f"Could not find {marker} in {start} or any of its parents.\n"
        "Set the notebook's kernel working directory to Code/Model, or set "
        "MODEL_DIR explicitly in this cell (e.g. MODEL_DIR = Path('/path/to/Code/Model'))."
    )

MODEL_DIR = _find_model_dir(Path.cwd())
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))  # lets the flat `from Model_Creation import ...`-style imports below resolve, same as running `python MicroGrids.py` from this folder
print(f"MODEL_DIR = {MODEL_DIR}")

#%% ---- next cell ----

RUN_TAG = "9sc_20y_capexp4_5kwgen_diagnostic"

MGPY_PARAMS = "Parameters_9sc_20y_gili_ketapang_capexp4_5kwgen_diagnostic_MILP_dieselops.dat"
MGPY_DEMAND = "Demand_9sc_20y_gili_ketapang.csv"
MGPY_RES = "RES_Time_Series_9sc_gili_ketapang.csv"
MGPY_WT_POWER_CURVE = None

MGPY_EXPECT_YEARS = 20
MGPY_EXPECT_SCENARIOS = 9
MGPY_EXPECT_MILP = 1

# 2026-08-30: DIAGNOSTIC -- reverts Generator_Nominal_Capacity_milp to the OLD 5kW default
# (was corrected to 470kW real physical unit size on 2026-08-23) to test the user's hypothesis:
# does a fine-grained generator unit size (like RES/Battery's own tiny per-unit sizes) make
# Generator_Units' LP relaxation nearly-integral on its own, avoiding the B&B stall seen at the
# real 470kW/unit resolution? Best-known recipe otherwise (RELAX_UC + BESS_FORM=nobinary +
# DegenMoves=0 + MIPFocus=3/Cuts=1/CutPasses=1). NOT a physically valid result if it solves --
# 5kW units are not real hardware, this is purely to isolate the granularity effect.
os.environ["MGPY_BESS_FORM"] = "nobinary"
os.environ["MGPY_MAX_BATTERY_KWH"] = "62031"
os.environ["MGPY_MAX_GENERATOR_KW"] = "9400"
os.environ["MGPY_MIPFOCUS"] = "3"
os.environ["MGPY_CUTS"] = "1"
os.environ["MGPY_CUTPASSES"] = "1"
os.environ["MGPY_RELAX_UC"] = "1"
os.environ["MGPY_DEGENMOVES"] = "0"
os.environ["MGPY_MEMLIMIT"] = "150"
os.environ["SLURM_CPUS_PER_TASK"] = "12"

for _env_var, _value in [
    ("MGPY_PARAMS", MGPY_PARAMS),
    ("MGPY_DEMAND", MGPY_DEMAND),
    ("MGPY_RES", MGPY_RES),
    ("MGPY_WT_POWER_CURVE", MGPY_WT_POWER_CURVE),
    ("MGPY_EXPECT_YEARS", MGPY_EXPECT_YEARS),
    ("MGPY_EXPECT_SCENARIOS", MGPY_EXPECT_SCENARIOS),
    ("MGPY_EXPECT_MILP", MGPY_EXPECT_MILP),
]:
    if _value is not None:
        os.environ[_env_var] = str(_value)

sys.argv = ["MicroGrids.py"] + ([RUN_TAG] if RUN_TAG else [])


#%% ---- next cell ----

from pyomo.environ import AbstractModel
from Model_Creation import Model_Creation
from Model_Resolution import Model_Resolution
from Results import ResultsSummary, TimeSeries, PrintResults
from Plots import DispatchPlot, SizePlot, CashFlowPlot
from run_id import RUN_ID  # same RUN_ID Results.py/Plots.py use to namespace Results/<RUN_ID>/ -- reused below to namespace the Gurobi log file too

#%% ---- next cell ----

class _Tee:
    """Duplicates writes to both an underlying stream and a log file, flushing every write
    so a concurrent `tail -f` sees output immediately instead of waiting on Python's own
    buffering."""
    def __init__(self, stream, filepath):
        self.stream = stream
        self.file = open(filepath, "a", buffering=1)  # line-buffered

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False  # some libraries (tqdm, etc.) probe this; keep it simple/false for a log file

LOGS_DIR = MODEL_DIR.parent.parent / "logs"  # same logs/ folder the .sbatch scripts' #SBATCH --output=logs/... writes into, at the repo root
LOGS_DIR.mkdir(exist_ok=True)                # matches sbatch behaviour (Slurm expects it to pre-exist there); harmless if it's already there

combined_logfile = LOGS_DIR / f"Combined_Output_{RUN_ID}.log"
sys.stdout = _Tee(sys.stdout, combined_logfile)
sys.stderr = _Tee(sys.stderr, combined_logfile)
print(f"Combined stdout/stderr log -> {combined_logfile}  (created now -- tail -f this from another terminal, same as sbatch's %j.out)")

#%% ---- next cell ----

import threading

RESOURCE_POLL_SECONDS = 60  # how often to sample; lower = finer detail, higher log volume

try:
    import psutil
    _resource_proc = psutil.Process(os.getpid())
    _HAVE_PSUTIL = True
except ImportError:
    _resource_proc = None
    _HAVE_PSUTIL = False
    print("psutil not available in this kernel -- resource monitor falls back to /proc reads "
          "for memory only (Linux only; CPU% will read as NaN, and on non-Linux this silently "
          "logs nothing).")

def _read_proc_rss_mb():
    """VmRSS from /proc/self/status, in MB. Returns None if unavailable (e.g. non-Linux)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except (FileNotFoundError, OSError):
        return None
    return None

_monitor_state = {}  # holds the active monitor's thread/stop-event/peak-RSS/label -- lets start/stop be called more than once per notebook run (once for the free-sizing solve, again for fixed-design verification)

def start_resource_monitor(label):
    """Start a background thread sampling RSS memory + CPU% every RESOURCE_POLL_SECONDS,
    writing to logs/Resource_Usage_<RUN_ID>_<label>.log and printing periodic summaries
    (captured into Combined_Output_<RUN_ID>.log via the Tee cell above). Call
    stop_resource_monitor() when the corresponding solve finishes."""
    if _monitor_state.get('thread') is not None:
        print("A resource monitor is already running -- call stop_resource_monitor() first.")
        return
    log_path = LOGS_DIR / f"Resource_Usage_{RUN_ID}_{label}.log"
    stop_event = threading.Event()
    peak_rss_mb = [0.0]

    def _loop():
        if _HAVE_PSUTIL:
            _resource_proc.cpu_percent(interval=None)  # first call always returns 0.0/garbage -- prime it, discard
        with open(log_path, 'a') as logf:
            logf.write("# time_s\trss_mb\tcpu_pct\n")
            t0 = time.time()
            while not stop_event.is_set():
                elapsed = time.time() - t0
                if _HAVE_PSUTIL:
                    rss_mb = _resource_proc.memory_info().rss / (1024**2)
                    cpu_pct = _resource_proc.cpu_percent(interval=None)
                    cpu_str = f"{cpu_pct:.1f}"
                else:
                    rss_mb = _read_proc_rss_mb()
                    cpu_str = "NaN"
                    if rss_mb is None:
                        stop_event.wait(RESOURCE_POLL_SECONDS)
                        continue
                peak_rss_mb[0] = max(peak_rss_mb[0], rss_mb)
                logf.write(f"{elapsed:.0f}\t{rss_mb:.1f}\t{cpu_str}\n")
                logf.flush()
                print(f"[resource monitor:{label}] t={elapsed:.0f}s  RSS={rss_mb:.0f} MB "
                      f"(peak so far {peak_rss_mb[0]:.0f} MB)  CPU={cpu_str}%")
                stop_event.wait(RESOURCE_POLL_SECONDS)

    thread = threading.Thread(target=_loop, daemon=True)
    _monitor_state.update(thread=thread, stop_event=stop_event, peak_rss_mb=peak_rss_mb,
                          label=label, log_path=log_path, start_time=time.time())
    thread.start()
    print(f"Resource monitor started ({label}) -> {log_path}  (polling every {RESOURCE_POLL_SECONDS}s)")

def stop_resource_monitor():
    """Stop the active resource monitor and print a final peak-usage summary."""
    if _monitor_state.get('thread') is None:
        print("No resource monitor is currently running.")
        return
    _monitor_state['stop_event'].set()
    _monitor_state['thread'].join(timeout=RESOURCE_POLL_SECONDS + 5)
    elapsed = time.time() - _monitor_state['start_time']
    print(f"\nResource monitor stopped ({_monitor_state['label']}). "
          f"Peak RSS observed: {_monitor_state['peak_rss_mb'][0]:.0f} MB over {elapsed:.0f}s. "
          f"Full trace -> {_monitor_state['log_path']}")
    _monitor_state.clear()

#%% ---- next cell ----

start = time.time()         # Start time counter
model = AbstractModel()     # Define type of optimization problem

#%% Processing

start_resource_monitor("freesizing")  # logs/Resource_Usage_<RUN_ID>_freesizing.log; stopped in the Timing cell below

Model_Creation(model) # Creation of the Sets, parameters and variables.

# Resolve the model instance
solver_logfile = str(LOGS_DIR / f"Solver_Output_{RUN_ID}.log")  # Gurobi's own LogFile -- separate from Combined_Output above; only appears once Gurobi's engine starts (after the temp .lp file finishes writing)
print(f"Gurobi-only log -> {solver_logfile}  (appears once solving actually starts; use Combined_Output_{RUN_ID}.log above to watch instance-build + solver output together, from the very start)")
instance = Model_Resolution(model, logfile=solver_logfile)

#%% ---- next cell ----

#%% Results

Time_Series       = TimeSeries(instance)
Optimization_Goal = instance.Optimization_Goal.extract_values()[None]
Results           = ResultsSummary(instance, Optimization_Goal,Time_Series) 

#%% ---- next cell ----

#%% Plot and print-out
PlotScenario = 1                     # Plot scenario
PlotDate = '01/01/2023 00:00:00'     # Month-Day-Year. If devoid of meaning: Day-Month-Year
PlotTime = 3                         # Number of days to be shown in the plot
PlotFormat = 'png'                   # Desired extension of the saved file (Valid formats: png, svg, pdf)
PlotResolution = 400                 # Plot resolution in dpi (useful only for .png files, .svg and .pdf output a vector plot)
'''
PlotScenario1 = 1                    # Plot scenario
PlotDate1 = '01/01/2042 00:00:00'    # Month-Day-Year. If devoid of meaning: Day-Month-Year
PlotTime1 = 3                        # Number of days to be shown in the plot
PlotFormat1 = 'png'                  # Desired extension of the saved file (Valid formats: png, svg, pdf)
PlotResolution1 = 400                # Plot resolution in dpi (useful only for .png files, .svg and .pdf output a vector plot)

PlotScenario2 = 1                    # Plot scenario
PlotDate2 = '01/01/2032 00:00:00'    # Month-Day-Year. If devoid of meaning: Day-Month-Year
PlotTime2 = 3                        # Number of days to be shown in the plot
PlotFormat2 = 'png'                  # Desired extension of the saved file (Valid formats: png, svg, pdf)
PlotResolution2 = 400                # Plot resolution in dpi (useful only for .png files, .svg and .pdf output a vector plot)

PlotScenario3 = 1                    # Plot scenario
PlotDate3 = '01/01/2038 00:00:00'    # Month-Day-Year. If devoid of meaning: Day-Month-Year 
PlotTime3 = 3                        # Number of days to be shown in the plot
PlotFormat3 = 'png'                  # Desired extension of the saved file (Valid formats: png, svg, pdf)
PlotResolution3 = 400                # Plot resolution in dpi (useful only for .png files, .svg and .pdf output a vector plot)
'''
DispatchPlot(instance,Time_Series,PlotScenario,PlotDate,PlotTime,PlotResolution,PlotFormat)
'''
DispatchPlot1(instance,TimeSeries,PlotScenario1,PlotDate1,PlotTime1,PlotResolution1,PlotFormat1)

DispatchPlot2(instance,TimeSeries,PlotScenario2,PlotDate2,PlotTime2,PlotResolution2,PlotFormat2)
DispatchPlot3(instance,TimeSeries,PlotScenario3,PlotDate3,PlotTime3,PlotResolution3,PlotFormat3)
'''
CashFlowPlot(instance,Results,PlotResolution,PlotFormat)
SizePlot(instance,Results,PlotResolution,PlotFormat)

PrintResults(instance, Results)  

#%% ---- next cell ----

#%% Timing
end = time.time()
elapsed = end - start
print('\n\nModel run complete (overall time: ',round(elapsed,0),'s,',round(elapsed/60,1),' m)\n')

stop_resource_monitor()  # prints peak RSS for the free-sizing solve; logs/Resource_Usage_<RUN_ID>_freesizing.log has the full trace