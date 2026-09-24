"""
Hour-of-day-aligned moving-block bootstrap, shared by the 8760h scenario
generators (scenario_generator_8760h.py, scenario_generator_8760h_pv.py) and
their sensitivity checks (inflation_factor_sensitivity.py,
block_length_sensitivity.py).

Both residual series are deviations from an hour-of-day (load) or
month-and-hour (PV) climatology, so a resampled block must land on the same
hours of day it was drawn from. The first implementation drew block starts at
arbitrary row indices of the observed series, which shifted daytime PV
residuals onto night hours (non-zero PV at night) and, for load, let blocks
run across missing hours and the Dec 2024 -> Apr 2026 collection gap. Here:

  - residuals live on a regular hourly grid split into contiguous segments
    (one per load collection period, one per calendar month for PV);
  - a block placed at output position p is drawn only from start hours whose
    hour of day (and, optionally, group such as calendar month) matches p's,
    and must lie entirely inside one segment.
"""
import numpy as np
import pandas as pd


def regular_hourly_grid(df, value_col, max_gap_h=168):
    """Reindexes an hourly series (columns: timestamp, value_col) onto a regular
    hourly grid. Runs of observations separated by more than max_gap_h hours
    become separate segments; hours missing inside a segment get NaN."""
    df = df.sort_values("timestamp")
    segment_id = (df["timestamp"].diff() > pd.Timedelta(hours=max_gap_h)).cumsum()
    parts = []
    for seg, g in df.groupby(segment_id):
        grid = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq="h", name="timestamp")
        part = g.set_index("timestamp")[[value_col]].reindex(grid).reset_index()
        part["segment"] = seg
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def aligned_start_pools(hour, segment, block_len, group=None):
    """Valid block start indices keyed by (hour of day, group). A start s is
    valid if [s, s + block_len) lies inside one segment; with group=None all
    starts share the key (hour, None)."""
    hour, segment = np.asarray(hour), np.asarray(segment)
    n = len(hour)
    inside = np.zeros(n, dtype=bool)
    if n >= block_len:
        inside[: n - block_len + 1] = segment[: n - block_len + 1] == segment[block_len - 1:]
    pools = {}
    for g in ([None] if group is None else np.unique(group)):
        mask = inside if g is None else inside & (np.asarray(group) == g)
        for h in range(24):
            pools[(h, g)] = np.flatnonzero(mask & (hour == h))
    return pools


def aligned_block_bootstrap(residuals, pools, block_len, n_hours, target_hour, rng, target_group=None):
    """One resampled residual series of length n_hours: consecutive blocks,
    each drawn from the pool matching the hour of day (and group) of the
    position where it is placed."""
    out = np.empty(n_hours)
    pos = 0
    while pos < n_hours:
        key = (int(target_hour[pos]), None if target_group is None else target_group[pos])
        start = rng.choice(pools[key])
        take = min(block_len, n_hours - pos)
        out[pos:pos + take] = residuals[start:start + take]
        pos += take
    return out
