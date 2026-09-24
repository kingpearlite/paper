"""Can HOMER's battery-free design serve the deterministic demand hour by hour?

    python _check_homer_h2_parallel.py [PV_KW] [BATTERY_KWH]      # default: the 2026-09-14 H2 design

Defaults are HOMER's original H2 (1,140 kW, 137 kWh); the 2026-09-21 re-run on the corrected
inputs is 1,316 kW and 110 kWh.
"""
import sys
import pandas as pd, numpy as np
pv_kw = float(sys.argv[1]) if len(sys.argv) > 1 else 1140.0
batt = float(sys.argv[2]) if len(sys.argv) > 2 else 137.0
d = pd.read_csv('Demand_1sc_20y_gili_ketapang_merged.csv', sep=';', decimal=',',
                index_col=0).to_numpy()
pvk = pd.read_csv('RES_Time_Series_1sc_gili_ketapang_merged.csv',
                  index_col=0).to_numpy()[:, 0] * 0.98 / 0.33
res_u = round(pv_kw/0.33); pv = pvk[:, None] * res_u * 0.33
bmax = batt / 4
def gmax_default(x):
    return np.where(x >= 587.5, np.minimum(x, 940),
                    np.where(x >= 117.5, np.minimum(x, 470), 0.0))
def gmax_parallel(x):
    # two units in parallel share the load: one unit covers [117.5, 470], two cover [235, 940]
    return np.where(x >= 117.5, np.minimum(x, 940), 0.0)
print(f'pins: RES {res_u} units ({res_u*0.33:.2f} kW), battery {batt:.0f} kWh '
      f'(discharge cap {bmax:.2f} kW), 2 gensets')
for label, gmax in (('default (at most one unit below full)', gmax_default),
                    ('parallel load sharing', gmax_parallel)):
    short = d - gmax(d) - pv - bmax
    bad = short > 1e-9
    print(f'{label}: hours no dispatch can serve (ignoring SOC): {bad.sum():,} of {d.size:,}; '
          f'in year 19: {bad[:, 18].sum():,}; worst gap {short.max():.1f} kW')
bad = (d - gmax_parallel(d) - pv - bmax) > 1e-9
yrs = np.nonzero(bad.any(axis=0))[0] + 1
print(f'parallel: unservable hours by year: ' +
      ', '.join(f'y{y}: {bad[:, y-1].sum()}' for y in yrs))
pvb = np.broadcast_to(pv, d.shape)
print(f'parallel: demand in those hours {d[bad].min():.1f}-{d[bad].max():.1f} kW, PV there {pvb[bad].max():.1f} kW max')
print(f'hours with demand below 117.5 kW: {(d < 117.5).sum()}')
