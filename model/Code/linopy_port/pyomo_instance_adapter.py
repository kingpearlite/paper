"""Generic Pyomo-`instance`-compatible adapter over a SOLVED linopy model, so the real
project's own `Results.py` (Code/Model/Results.py -- ResultsSummary/TimeSeries/PrintResults)
can be called UNMODIFIED against a linopy solve and produce the exact same
Code/Results/<RUN_ID>/Results_Summary.xlsx + Time_Series_SC_*.xlsx output Pyomo runs produce.

WHY THIS EXISTS: the linopy_port crosscheck scripts only ever printed NPC + sizing to stdout --
never wired into the project's actual reporting pipeline. Results.py accesses everything via
Pyomo's `instance.X.value` / `.get_values()` / `.extract_values()` idiom. Rather than duplicate
Results.py's ~2000 lines of table-building logic (LP/MILP, Greenfield/Brownfield, grid-
connected/not, land-use/not, multi-objective/not branches) for linopy, this module builds a
lightweight adapter object that answers the SAME calls, backed by dicts assembled from the
solved linopy model's `.solution` values -- so Results.py's own functions run unchanged.

GENERALITY: the adapter mechanism itself (the two proxy classes below) is not hardcoded to any
one parameter or variable name -- it just serves whatever dict of {name: data} it's given. What
IS scoped to this project's actual configuration (Gili Ketapang: brownfield, MILP, partial-load,
Model_Components=0, Grid_Connection=0, Land_Use=0, Multiobjective_Optimization=0) is which
values `linopy_results_export.py` populates -- Results.py's own `if instance.Grid_Connection...`
etc. branches already gate dead code paths off when those flags are 0, so nothing beyond this
project's real parameter set needs to be supplied for a correct result. Extending to a
grid-connected/land-use/multi-objective run later just means loading the extra parameters those
branches read (see the comments in linopy_results_export.py) -- the adapter itself needs no
changes.
"""
import numpy as np
import pandas as pd
import xarray as xr


class _ParamProxy:
    """Mimics a Pyomo scalar-or-indexed Param: `.value` (scalar/dict[None]), `.extract_values()`
    (always a dict), and numeric/comparison operators (Results.py compares some Params directly,
    e.g. `instance.Grid_Connection == 1`, without `.value` -- real Pyomo Param objects support
    that via NumericValue; this replicates just enough of it)."""

    def __init__(self, data):
        self._data = data  # scalar, or dict keyed by index (int/tuple) or None for a "scalar Param stored as dict"

    @property
    def value(self):
        if isinstance(self._data, dict):
            if None in self._data:
                return self._data[None]
            if len(self._data) == 1:
                return next(iter(self._data.values()))
        return self._data

    def extract_values(self):
        if isinstance(self._data, dict):
            return self._data
        return {None: self._data}

    def __call__(self):  # old-style Pyomo Param access, e.g. instance.StartDate()
        return self.value

    def _other_value(self, other):
        return other.value if isinstance(other, _ParamProxy) else other

    def __eq__(self, other):
        return self.value == self._other_value(other)

    def __ne__(self, other):
        return self.value != self._other_value(other)

    def __lt__(self, other):
        return self.value < self._other_value(other)

    def __le__(self, other):
        return self.value <= self._other_value(other)

    def __gt__(self, other):
        return self.value > self._other_value(other)

    def __ge__(self, other):
        return self.value >= self._other_value(other)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __bool__(self):
        return bool(self.value)

    # Arithmetic operators (Stage 5, full-parity roadmap): Results.py's grid-connection code
    # paths (e.g. YearlyCosts() at L1274/1279: `El_Purchased_Price = instance.
    # Grid_Purchased_El_Price` then `... * El_Purchased_Price` with no `.value`) use a raw Param
    # directly in arithmetic, exactly like the comparison operators above already handle for
    # `==`/`<`/etc. -- real Pyomo Param objects support this via NumericValue; previously missing
    # here because no earlier code path (all off-grid) ever hit a raw-Param-in-arithmetic case.
    def __mul__(self, other):
        return self.value * self._other_value(other)

    def __rmul__(self, other):
        return self._other_value(other) * self.value

    def __add__(self, other):
        return self.value + self._other_value(other)

    def __radd__(self, other):
        return self._other_value(other) + self.value

    def __sub__(self, other):
        return self.value - self._other_value(other)

    def __rsub__(self, other):
        return self._other_value(other) - self.value

    def __truediv__(self, other):
        return self.value / self._other_value(other)

    def __rtruediv__(self, other):
        return self._other_value(other) / self.value

    def __neg__(self):
        return -self.value

    def __pos__(self):
        return self.value

    def __repr__(self):
        return f"_ParamProxy({self._data!r})"


class _VarProxy:
    """Mimics a Pyomo scalar-or-indexed Var: `.value` and `.get_values()` (always a dict)."""

    def __init__(self, data):
        self._data = data

    @property
    def value(self):
        if isinstance(self._data, dict):
            if None in self._data:
                return self._data[None]
            if len(self._data) == 1:
                return next(iter(self._data.values()))
        return self._data

    def get_values(self):
        if isinstance(self._data, dict):
            return self._data
        return {None: self._data}

    def extract_values(self):
        # Real Pyomo Var objects support BOTH .get_values() and .extract_values() (identical
        # result for a Var, unlike Param where extract_values() always returns a dict but
        # .value unwraps a scalar) -- Results.py's LP-mode branch calls .extract_values() on
        # Battery_Nominal_Capacity/Generator_Nominal_Capacity, which this port registers as
        # variables (Stage 8, full-parity roadmap: found via a real lp_mode export run,
        # 2026-09-04) even though the MILP branch's equivalent
        # Battery_Nominal_Capacity_milp/Generator_Nominal_Capacity_milp are registered as params.
        return self.get_values()

    def __repr__(self):
        return f"_VarProxy(n={len(self._data) if isinstance(self._data, dict) else 1})"


class _ObjectiveProxy:
    """Mimics Pyomo's ObjectiveFuntion: `.expr()` returns the solved objective value."""

    def __init__(self, value):
        self._value = value

    def expr(self):
        return self._value


class PyomoInstanceAdapter:
    """Drop-in stand-in for a solved Pyomo ConcreteModel `instance`, as consumed by Results.py.

    `params`: dict of {name: scalar-or-dict} for Pyomo Params (accessed via .value/.extract_values()).
    `variables`: dict of {name: scalar-or-dict} for Pyomo Vars (accessed via .value/.get_values()).
    `objective_value`: the solved objective's value, exposed as `instance.ObjectiveFuntion.expr()`.
    """

    def __init__(self, params, variables, objective_value=None):
        self._params = {k: _ParamProxy(v) for k, v in params.items()}
        self._vars = {k: _VarProxy(v) for k, v in variables.items()}
        if objective_value is not None:
            self._vars["__objective__"] = _ObjectiveProxy(objective_value)

    def __getattr__(self, name):
        if name == "ObjectiveFuntion" and "__objective__" in self._vars:
            return self._vars["__objective__"]
        if name in self._vars:
            return self._vars[name]
        if name in self._params:
            return self._params[name]
        raise AttributeError(
            f"PyomoInstanceAdapter has no '{name}' -- add it to the params/variables dict "
            f"passed into the adapter (see linopy_results_export.py's build_adapter())."
        )


# ---------------------------------------------------------------------------
# Reshaping helpers: turn a solved linopy variable (an xarray DataArray of
# .solution values) into the same {index: value} dict shape Pyomo's Var.get_values()
# would produce.
# ---------------------------------------------------------------------------

def da_to_pyomo_dict(da, dim_order=None, inject=None):
    """Convert a linopy-solved xarray DataArray into a Pyomo-style {key: float} dict.

    `dim_order`: the FULL tuple of index names in Pyomo's key order (e.g. ("s","y","r","t")).
    Any name in `dim_order` not present in `da.dims` is treated as an injected constant index
    (see `inject`) -- this project's linopy models drop the r/g (RES/generator "type") axis
    entirely since R=G=1 always (single RES source, single generator type), so Pyomo's 4-tuple
    keys need that constant re-inserted.
    `inject`: {name: constant_value} for names in dim_order missing from da.dims (typically {"r": 1}
    or {"g": 1}).

    Single-dim (or dim_order of length 1) results collapse to plain (non-tuple) keys, matching
    Pyomo's own Var.get_values() convention for a Var with only one index set.
    """
    inject = inject or {}
    series = da.to_series()  # vectorized -- avoids a slow python-level iterrows() loop on million-row arrays
    if dim_order is None:
        dim_order = list(da.dims)
    present_dims = list(da.dims)

    if len(present_dims) == 0:
        return {None: float(da.values)}

    result = {}
    if len(present_dims) == 1 and len(dim_order) == 1:
        for idx, v in series.items():
            result[idx] = float(v)
        return result

    for idx, v in series.items():
        idx_t = idx if isinstance(idx, tuple) else (idx,)
        present_iter = iter(idx_t)
        key = tuple(inject[d] if d not in present_dims else next(present_iter) for d in dim_order)
        result[key] = float(v)
    return result


def scalar_from_solution(var):
    """Extract a plain python float from a linopy Variable/expression's solved `.solution`
    (handles both a bare scalar DataArray and a size-1 array)."""
    val = var.solution
    if isinstance(val, xr.DataArray):
        return float(val.values.reshape(-1)[0]) if val.size else float(val.values)
    return float(val)
