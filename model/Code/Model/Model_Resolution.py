"""
MicroGridsPy - Multi-year capacity-expansion (MYCE)

Linear Programming framework for microgrids least-cost sizing,
able to account for time-variable load demand evolution and capacity expansion.

"""


from pyomo.environ import *                 # Pyomo modeling objects: Constraint, Objective, Var, Param, minimize, value, etc.
from pyomo.opt import SolverFactory          # Factory used to instantiate solver interfaces (gurobi, glpk, cplex)
import matplotlib
from matplotlib import pyplot as plt         # Used to draw and save the Pareto curve plot
import re                                    # Used to extract numeric values from the .dat parameter file via regex
import os
import time                                  # Used to compute remaining wall-clock time before the Slurm walltime limit
import getpass                               # Resolves the username for the scratch2 node-file path (robust to $USER being unset)
import params_path                           # Resolves which parameters file this run reads (MGPY_PARAMS)
from run_id import RUN_ID                    # Unique identifier for this run, used to name the Results output folder

matplotlib.use('Agg')  # Switch to 'Agg' backend to prevent GUI operations

current_directory = os.path.dirname(os.path.abspath(__file__))          # Directory containing this script
inputs_directory = os.path.join(current_directory, '..', 'Inputs')      # Path to the Inputs folder
data_file_path = params_path.PARAMS_PATH                                # AMPL/Pyomo .dat parameter file, resolved once via MGPY_PARAMS (see params_path.py)


# =============================================================================
# DESIGN VERIFICATION -- pin the sizing decisions to a reported design and
# re-solve as a dispatch-only problem. Opt-in: none of this runs unless an
# MGPY_FIX_* variable is set, so every existing workflow is unaffected.
# =============================================================================
# WHY IT EXISTS. A reported objective on this model is not self-verifying. Job
# 58519303 reported NPC 109,629.05 as proven optimal, gap 0.0000%; job 58547964
# later found a feasible 103,035.96, which a minimisation cannot do below its
# own optimum -- so the earlier BOUND was invalid. Any check that consults the
# bound therefore proves nothing. Fixing the sizing to a reported design asks a
# question the bound cannot contaminate: can THIS combination of PV, battery and
# genset actually serve the demand, and at what dispatch cost?
#
# WHICH VARIABLES. Under MILP_Formulation = 1 the sizing decisions are integer
# UNIT COUNTS, not capacities:
#     RES_Units_milp[step, source] x RES_Nominal_Capacity[source]
#     Battery_Units[step]          x Battery_Nominal_Capacity_milp
#     Generator_Units[step, type]  x Generator_Nominal_Capacity_milp[type]
# model.RES_Units, model.Battery_Nominal_Capacity and model.Generator_Nominal_
# Capacity belong to the LP formulation and occur ZERO times in
# Constraints_Greenfield_Milp (checked, 2026-07-27). Fixing those under a MILP
# run is a silent no-op that leaves the design free to move -- the run would
# look like it verified something and would not have. Hence the split below on
# MILP_Formulation.
#
# UNITS. The model holds power and energy natively in kW and kWh (converted from
# W/Wh on 2026-07-28 -- see the unit-conversion plan record for detail), matching
# Results.py's Size sheet with no scaling in between. These env vars take the
# kW/kWh figures exactly as printed there. With Parameters_6sc_5y_MILP.dat one PV
# unit is 1 kW, one battery unit 1 kWh and one genset unit 5 kW, so
# "PV 48 kW / Battery 308 kWh / Diesel 5 kW" is 48 / 308 / 1 units.
#
#   MGPY_FIX_RES_KW          one value per RES source, in index order   "48,0"
#   MGPY_FIX_BATTERY_KWH     installed battery capacity                 "308"
#   MGPY_FIX_GEN_KW          one value per generator type               "5"
#   MGPY_FIX_RELAX_BINARIES  1 = relax the leftover battery binaries (see below)
#
# MULTI-STEP DESIGNS (capacity expansion, Steps_Number > 1). Each env var above
# takes one ';'-separated group PER INVESTMENT STEP, each group itself a
# comma-separated list in the same index order as the single-step case, e.g. a
# 2-step, 1-source design that builds 2187.9 kW of PV at step 1 and another
# 3031.38 kW at step 2 is MGPY_FIX_RES_KW="2187.9;3031.38". These are PER-STEP
# INCREMENTS, matching exactly what Results.py's Size sheet prints as its
# "Step 1 / Step 2 / Total" columns -- do not add previous steps in yourself.
# IMPORTANT: this is the INTERFACE convention, not the model's internal one.
# RES_Units_milp[step,r]/Battery_Units[step]/Generator_Units[step,g] themselves
# hold the CUMULATIVE total as of that step (confirmed from two places that
# derive an increment BY SUBTRACTING consecutive steps: Constraints.py's
# Inv_Ren/Inv_Bat/Inv_Gen investment-cost formulas, and Results.py's own
# "(Units[st]-Units[st-1])*Nominal_Capacity # incremental capacity added at
# step st"). Fix_Design accumulates the per-step increments you provide into
# running totals internally before pinning any variable -- fixing a step to
# its own increment instead of the running total under-provisions every step
# after the first and silently makes the design infeasible.
# Single-step designs are unaffected: omit the ';' and give one comma-separated
# group as before, unit size per component (RES_Nominal_Capacity, Generator_
# Nominal_Capacity_milp, Battery_Nominal_Capacity_milp) is a step-independent
# constant, so only the per-step unit COUNT differs -- nothing else about the
# fixing logic changes between step counts.
#
# WHY RELAXING THE LEFTOVER BINARIES IS STILL A PROOF. Fixing the sizing removes
# only 4 integers; job 58547964 logged "262804 integer (262800 binary)", and the
# remaining 262,800 are Single_Flow_BESS -- one per scenario/year/hour. The fixed
# model is therefore still a MILP, not the ~5-minute LP this check is meant to
# be. Those binaries appear in exactly two constraints and nowhere in the
# objective:
#     Battery_Outflow[t] <= z[t]     * Battery_Maximum_Discharge_Power * Delta_Time
#     Battery_Inflow[t]  <= (1-z[t]) * Battery_Maximum_Charge_Power    * Delta_Time
# So for any given dispatch an integral z exists if and only if no period both
# charges and discharges: z[t]=1 covers Inflow[t]=0, z[t]=0 covers Outflow[t]=0.
# Relax them, solve the LP, then check min(Inflow, Outflow) == 0 in every period.
# If it holds, the relaxed solution IS a feasible point of the full MILP with the
# same objective -- a complete certificate, obtained without ever reading a
# bound. Report_Fixed_Design runs that test automatically and prints the verdict.
#
# Battery_Maximum_Charge_Power and Battery_Maximum_Discharge_Power are fixed too,
# to exactly the value their own defining equality prescribes. Without that they
# stay free variables and z*Power is a genuine bilinear term -- Gurobi logged
# 1,051,200 SOS constraints for it -- which would survive the relaxation as a
# non-convex quadratic instead of collapsing to a linear big-M.
# =============================================================================

FIX_DESIGN_ENV_VARS = ('MGPY_FIX_RES_KW', 'MGPY_FIX_BATTERY_KWH', 'MGPY_FIX_GEN_KW')


def Fix_Design_Requested():
    """True when any MGPY_FIX_* sizing variable is set, i.e. verification mode."""
    return any(os.environ.get(name) for name in FIX_DESIGN_ENV_VARS)


def _read_fixed_sizes(env_var, count, what, n_steps=1):
    """Parse env_var into a list of length n_steps, each a list of `count` floats.

    Single-step (n_steps=1): 'a,b,c' -- unchanged from the original single-step-only
    format, no ';' expected or allowed.
    Multi-step (n_steps>1): 'a,b,c;d,e,f;...' -- one ';'-separated group PER STEP,
    each a per-step INCREMENT in the same comma-separated index order as the
    single-step case. See the header comment above for why increments, not totals.
    """
    raw = os.environ.get(env_var)
    if raw is None:
        raise ValueError(
            "Design fixing is active but {} is not set. Every sizing decision has "
            "to be pinned: an unpinned one stays a free variable, and the run then "
            "verifies some other design while looking like it verified yours. "
            "Expected {} comma-separated value(s) per step, one per {}{}."
            .format(env_var, count, what, " (';'-separated groups, one per investment step)"
                    if n_steps > 1 else ""))

    step_groups = raw.split(';')
    if n_steps == 1:
        if len(step_groups) != 1:
            raise ValueError(
                "{}='{}' contains ';' but this instance has only 1 investment step. "
                "Remove the ';' and give one comma-separated value per {}."
                .format(env_var, raw, what))
    elif len(step_groups) != n_steps:
        raise ValueError(
            "{}='{}' has {} ';'-separated group(s) but this instance has {} investment "
            "steps. Give one ';'-separated group per step (in step order), each a "
            "comma-separated PER-STEP INCREMENT with one value per {} -- see the header "
            "comment above Fix_Design_Requested for the exact format and an example."
            .format(env_var, raw, len(step_groups), n_steps, what))

    sizes_by_step = []
    for step_num, group in enumerate(step_groups, start=1):
        chunks = [chunk.strip() for chunk in group.split(',') if chunk.strip()]
        if len(chunks) != count:
            raise ValueError(
                "{}{} gives {} value(s) but this model has {} {}(s). Give one value "
                "per {} in index order, using 0 for the ones the design does not build."
                .format(env_var, " step {} ('{}')".format(step_num, group) if n_steps > 1 else "='{}'".format(raw),
                        len(chunks), count, what, what))
        try:
            sizes_by_step.append([float(chunk) for chunk in chunks])
        except ValueError:
            raise ValueError("{} step {} ('{}') is not a comma-separated list of numbers."
                             .format(env_var, step_num, group))
    return sizes_by_step


def _unit_count(size, unit_capacity, env_var, label, unit):
    """Turn a kW/kWh figure as printed by Results.py into a whole number of units."""
    exact = size  / unit_capacity
    count = int(round(exact))
    if abs(exact - count) > 1e-6:
        raise ValueError(
            "{}: {:g} {} is {:.6f} units of {:g} {} -- not a whole number of units, "
            "so the MILP cannot express it. Cross-check {} against the Size sheet of "
            "Results_Summary.xlsx.".format(label, size, unit, exact, unit_capacity,
                                           unit.replace('k', ''), env_var))
    return count


def Fix_Design(instance, MILP_Formulation, Model_Components):
    """Pin RES / battery / generator sizing to the design given in MGPY_FIX_*.

    Supports both single-step and multi-step (capacity-expansion) instances. For
    multi-step instances, each MGPY_FIX_* env var carries one ';'-separated group
    of PER-STEP INCREMENTS per investment step -- see the format comment above
    Fix_Design_Requested and _read_fixed_sizes' docstring.

    CUMULATIVE VS INCREMENTAL. The env-var *interface* takes increments (matching
    how Results.py's own Size sheet prints "Step 1 / Step 2 / Total" -- the natural
    way a human reads a capacity-expansion result), but the model's actual decision
    variables (RES_Units_milp[step,r], Battery_Units[step], Generator_Units[step,g])
    hold the CUMULATIVE total installed as of that step, not that step's own
    increment -- confirmed directly from two places the model derives an increment
    BY SUBTRACTING consecutive steps: Constraints.py's Inv_Ren/Inv_Bat/Inv_Gen
    ("(Units[ut] - Units[ut-1]) * cost"), and Results.py's own Size-sheet builder
    ("(RES_Units_milp[st,r]-RES_Units_milp[st-1,r])*Nominal_Capacity  # incremental
    capacity added at step st"). Fixing each step to its own increment (rather than
    the running total) silently under-provisions every step after the first --
    caught by a local smoke test that went straight to Infeasible in 2.2s on a
    design that Gurobi itself had just reported feasible and optimal minutes
    earlier. This function therefore accumulates each component's per-step
    increments into a running total BEFORE fixing any variable.

    Returns a list of human-readable lines describing exactly what was pinned, so
    the log records the design that was actually verified rather than the one that
    was intended.
    """
    steps = list(instance.steps)
    n_steps = len(steps)
    has_battery = Model_Components in (0, 1)
    has_generator = Model_Components in (0, 2)
    summary = []
    step_tag = lambda step: 'step {} '.format(step) if n_steps > 1 else ''

    "Renewables"
    sources = list(instance.renewable_sources)
    res_names = instance.RES_Names.extract_values()
    res_kw_by_step = _read_fixed_sizes('MGPY_FIX_RES_KW', len(sources), 'RES source', n_steps=n_steps)
    res_cumulative = [0.0] * len(sources)
    for step, res_kw_increment in zip(steps, res_kw_by_step):
        for idx, (increment, source) in enumerate(zip(res_kw_increment, sources)):
            res_cumulative[idx] += increment                                # running total as of this step
            size = res_cumulative[idx]
            nominal = value(instance.RES_Nominal_Capacity[source])          # kW per unit
            name = res_names.get(source, 'RES {}'.format(source))
            if MILP_Formulation:
                units = _unit_count(size, nominal, 'MGPY_FIX_RES_KW', name, 'kW')
                instance.RES_Units_milp[step, source].fix(units)
                summary.append('  {}{:<20} {:>10.4g} kW cum.  -> RES_Units_milp[{},{}] = {:g}   (unit = {:g} kW)'
                               .format(step_tag(step), name, size, step, source, units, nominal))
            else:
                units = size  / nominal
                instance.RES_Units[step, source].fix(units)
                summary.append('  {}{:<20} {:>10.4g} kW cum.  -> RES_Units[{},{}] = {:g}   (unit = {:g} kW)'
                               .format(step_tag(step), name, size, step, source, units, nominal))

    "Battery bank"
    if has_battery:
        battery_kwh_by_step = _read_fixed_sizes('MGPY_FIX_BATTERY_KWH', 1, 'battery bank', n_steps=n_steps)
        battery_cumulative = 0.0
        for step, vals in zip(steps, battery_kwh_by_step):
            battery_cumulative += vals[0]                                   # running total as of this step
            battery_kwh = battery_cumulative
            if MILP_Formulation:
                nominal = value(instance.Battery_Nominal_Capacity_milp)     # kWh per unit
                units = _unit_count(battery_kwh, nominal, 'MGPY_FIX_BATTERY_KWH', 'Battery bank', 'kWh')
                instance.Battery_Units[step].fix(units)
                capacity = units * nominal                                  # cumulative kWh installed as of this step
                summary.append('  {}{:<20} {:>10.4g} kWh cum. -> Battery_Units[{}] = {:g}   (unit = {:g} kWh)'
                               .format(step_tag(step), 'Battery bank', battery_kwh, step, units, nominal))
            else:
                capacity = battery_kwh
                instance.Battery_Nominal_Capacity[step].fix(capacity)
                summary.append('  {}{:<20} {:>10.4g} kWh cum. -> Battery_Nominal_Capacity[{}] = {:g} kWh'
                               .format(step_tag(step), 'Battery bank', battery_kwh, step, capacity))
            # Pinned to exactly what Max_Power_Battery_Charge/Discharge already force
            # them to, which adds no restriction but makes the Single_Flow_BESS products
            # linear instead of bilinear. See the header note. capacity here is the
            # CUMULATIVE bank as of this step, so charge/discharge power scales with
            # the full accumulated bank, not just this step's own increment.
            charge_power = capacity / value(instance.Maximum_Battery_Charge_Time)
            discharge_power = capacity / value(instance.Maximum_Battery_Discharge_Time)
            instance.Battery_Maximum_Charge_Power[step].fix(charge_power)
            instance.Battery_Maximum_Discharge_Power[step].fix(discharge_power)
            summary.append('  {}{:<20} {:>10} '
                           '     -> Battery_Maximum_Charge_Power[{}] = {:g} kW, Discharge = {:g} kW'
                           .format(step_tag(step), '(implied by above)', '', step, charge_power, discharge_power))

    "Diesel generators"
    if has_generator:
        generators = list(instance.generator_types)
        gen_names = instance.Generator_Names.extract_values()
        gen_kw_by_step = _read_fixed_sizes('MGPY_FIX_GEN_KW', len(generators), 'generator type', n_steps=n_steps)
        gen_cumulative = [0.0] * len(generators)
        for step, gen_kw_increment in zip(steps, gen_kw_by_step):
            for idx, (increment, generator) in enumerate(zip(gen_kw_increment, generators)):
                gen_cumulative[idx] += increment                            # running total as of this step
                size = gen_cumulative[idx]
                name = gen_names.get(generator, 'Generator {}'.format(generator))
                if MILP_Formulation:
                    nominal = value(instance.Generator_Nominal_Capacity_milp[generator])   # kW per unit
                    units = _unit_count(size, nominal, 'MGPY_FIX_GEN_KW', name, 'kW')
                    instance.Generator_Units[step, generator].fix(units)
                    summary.append('  {}{:<20} {:>10.4g} kW cum.  -> Generator_Units[{},{}] = {:g}   (unit = {:g} kW)'
                                   .format(step_tag(step), name, size, step, generator, units, nominal))
                else:
                    capacity = size
                    instance.Generator_Nominal_Capacity[step, generator].fix(capacity)
                    summary.append('  {}{:<20} {:>10.4g} kW cum.  -> Generator_Nominal_Capacity[{},{}] = {:g} kW'
                                   .format(step_tag(step), name, size, step, generator, capacity))

    return summary


def Battery_Single_Flow_Form(MILP_Formulation):
    """Validate MGPY_BESS_FORM and announce it. Returns the mode string.

    'linear' needs MGPY_MAX_BATTERY_KWH because there is no data-derived ceiling on
    battery INFLOW -- RES and generator output both scale with unbounded integer unit
    counts. Discharge needs no such input: Max_Bat_out already caps outflow at demand,
    which is an exact big-M for free. Failing here costs a second; failing after the
    model is built costs however long presolve took.
    """
    form = os.environ.get('MGPY_BESS_FORM', 'bilinear').strip().lower()
    if form not in ('bilinear', 'linear', 'nobinary'):
        raise ValueError(
            "MGPY_BESS_FORM='{}' is not recognised. Use 'bilinear' (default, the "
            "historical formulation), 'linear', or 'nobinary'.".format(form))
    if form != 'bilinear' and not MILP_Formulation:
        raise ValueError(
            "MGPY_BESS_FORM='{}' only applies to the MILP formulation; this run has "
            "MILP_Formulation = 0. The LP branch has no single-flow binaries.".format(form))
    if form == 'linear' and not os.environ.get('MGPY_MAX_BATTERY_KWH'):
        raise ValueError(
            "MGPY_BESS_FORM=linear needs MGPY_MAX_BATTERY_KWH: the linearised charge "
            "gate is a big-M constraint and nothing in the model bounds battery inflow. "
            "Set it well above any plausible design -- 5x the expected battery size is "
            "ample. Check_Battery_Bound() warns if the answer lands on the ceiling.")
    if form != 'bilinear':
        print('\n' + '=' * 78)
        print('BATTERY SINGLE-FLOW FORM: {}  (default is bilinear)'.format(form.upper()))
        if form == 'linear':
            print('  Cap and mode gate split into separate linear constraints.')
            print('  Expect ZERO quadratic constraints and no SOS reformulation.')
            print('  Same feasible set as bilinear for integral binaries.')
        else:
            print('  Mode gate and its 262,800 binaries DROPPED; only the C-rate cap remains.')
            print('  This is NOT an equivalent model -- it permits simultaneous charge and')
            print('  discharge, which is assumed uneconomic. The post-solve check below is')
            print('  what makes the result valid; a non-zero overlap invalidates the run.')
        print('=' * 78)
    return form


def Bound_Battery_Size(instance, MILP_Formulation):
    """Put a finite upper bound on the battery sizing variable. Returns a description.

    WHY. Battery_Units is declared NonNegativeIntegers with no upper bound, and
    Battery_Maximum_Discharge_Power is NonNegativeReals with no upper bound. Those
    two appear multiplied by the Single_Flow_BESS binary, which is what makes
    525,600 constraints quadratic. Linearising a binary-times-continuous product
    needs a finite ceiling on the continuous factor; with none available Gurobi
    falls back to an SOS reformulation (it logged 1,051,200 of them) and derives
    whatever bounds presolve can infer.

    That matters because the invalid bounds on this model are produced by ROOT CUT
    GENERATION, not by branching (measured 2026-07-27): jobs 58553291 and 58547964
    both start from a valid root relaxation of 96,153.90 and end with bounds of
    100,290.56 and 100,408.67, having explored 1 and 11 nodes respectively. A
    feasible point exists at 96,532.96, so the maximum valid movement was +379 and
    the cuts moved +4,140 -- an overshoot of roughly 10x. The cut families
    responsible are MIR and Relax-and-lift, both derived by rounding coefficients,
    on a matrix spanning 7.5e11. Unbounded big-M values are a standard cause of
    exactly that failure, so bounding the size is the cheapest thing to try.

    Opt-in via MGPY_MAX_BATTERY_KWH; unset leaves the model exactly as before.
    Pick a value well above any plausible design -- 5x the verified optimum is
    still finite enough to help. Check_Battery_Bound() warns after the solve if the
    answer sits ON the bound, which would mean it is cutting off real solutions.
    """
    raw = os.environ.get('MGPY_MAX_BATTERY_KWH')
    if not raw:
        return None
    max_kwh = float(raw)
    for step in instance.steps:
        if MILP_Formulation:
            nominal = value(instance.Battery_Nominal_Capacity_milp)      # kWh per unit
            instance.Battery_Units[step].setub(int(max_kwh  / nominal))
        else:
            instance.Battery_Nominal_Capacity[step].setub(max_kwh )
        # The power variables inherit the ceiling through their own defining
        # equalities, but stating it explicitly is what actually gives the
        # bilinear terms a finite big-M rather than leaving presolve to infer one.
        capacity = max_kwh 
        instance.Battery_Maximum_Charge_Power[step].setub(
            capacity / value(instance.Maximum_Battery_Charge_Time))
        instance.Battery_Maximum_Discharge_Power[step].setub(
            capacity / value(instance.Maximum_Battery_Discharge_Time))
    return max_kwh


def Check_Battery_Bound(instance, max_kwh, MILP_Formulation):
    """Warn loudly if the solution sits on the artificial ceiling set above."""
    if not max_kwh:
        return
    for step in instance.steps:
        if MILP_Formulation:
            nominal = value(instance.Battery_Nominal_Capacity_milp)
            installed = value(instance.Battery_Units[step]) * nominal 
        else:
            installed = value(instance.Battery_Nominal_Capacity[step]) 
        if installed >= max_kwh * (1 - 1e-6):
            print('\n' + '!' * 78)
            print('MGPY_MAX_BATTERY_KWH IS BINDING at step {}: installed {:g} kWh == the '
                  'ceiling.'.format(step, installed))
            print('The bound is cutting off designs the model wanted. Raise it and re-run;')
            print('this result is not trustworthy as it stands.')
            print('!' * 78 + '\n')


# Discrete variables Relax_Discrete_Variables must NOT relax (2026-09-11).
# Salvage_Positive_Flag (the salvage floor-at-zero linearization, added 2026-08-28) sits inside a
# big-M pair, Salvage_Value <= raw + M*(1-f) and Salvage_Value <= M*f. With f continuous the
# solver picks f = (raw+M)/(2M) and books Salvage_Value = (raw+M)/2 -- about M/2 of fictitious
# salvage, subtracted straight off the NPC. Found by the linopy integral cross-check on
# smoketest_6sc_2y_dieselops: relaxed NPC 1,992,187.07 with Salvage Value 1,198.22 kUSD =
# (396.43 + 2,000)/2 exactly, against 2,793,966.33 from linopy (whose flag stays binary). Unlike
# Single_Flow_BESS, no post-solve check catches this, so every MGPY_FIX_RELAX_BINARIES=1 result
# since 2026-08-28 needs re-checking. It is one scalar binary, so keeping it integral costs
# nothing.
KEEP_INTEGRAL_WHEN_RELAXING = ('Salvage_Positive_Flag',)


def Relax_Discrete_Variables(instance):
    """Relax every still-free integer/binary variable onto its continuous interval,
    except those listed in KEEP_INTEGRAL_WHEN_RELAXING (see the note above it).

    Only variables left unfixed are touched, so the pinned design stays integral.
    Returns the number relaxed.
    """
    relaxed = 0
    for var in instance.component_objects(Var, active=True):
        if var.local_name in KEEP_INTEGRAL_WHEN_RELAXING:
            print('  Keeping {} integral (a big-M binary; relaxing it is not lossless).'.format(var.local_name))
            continue
        for index in var:
            var_data = var[index]
            if var_data.fixed or not (var_data.is_binary() or var_data.is_integer()):
                continue
            lower, upper = var_data.lb, var_data.ub   # Binary -> (0, 1); NonNegativeIntegers -> (0, None)
            var_data.domain = Reals
            var_data.setlb(lower)
            var_data.setub(upper)
            relaxed += 1
    return relaxed


def Report_Fixed_Design(instance, relaxed_count, Model_Components):
    """Print the bound-free verdict for a fixed-design solve.

    Reports the dispatch cost, the unserved energy per scenario, and -- when the
    battery binaries were relaxed -- whether the answer is a genuine feasible
    point of the full MILP (no period charging and discharging at once).
    """
    print('\n' + '=' * 78)
    print('FIXED-DESIGN VERIFICATION -- RESULT')
    print('=' * 78)

    objective = value(instance.ObjectiveFuntion)
    print('  Objective            = {:,.2f} USD   ({:,.4f} kUSD)'.format(objective, objective / 1e3))

    "Unserved energy -- must be zero when Lost_Load_Fraction is 0"
    lost_load = instance.Lost_Load.get_values()
    demand = instance.Energy_Demand.extract_values()
    per_scenario_lost, per_scenario_demand = {}, {}
    for (s, y, t), val in lost_load.items():
        per_scenario_lost[s] = per_scenario_lost.get(s, 0.0) + (val or 0.0)
    for (s, y, t), val in demand.items():
        per_scenario_demand[s] = per_scenario_demand.get(s, 0.0) + (val or 0.0)
    print('  Lost load by scenario [kWh]:')
    for s in sorted(per_scenario_lost):
        share = per_scenario_lost[s] / per_scenario_demand[s] if per_scenario_demand.get(s) else 0.0
        print('    scenario {:<3} {:>18,.6f}   ({:.3e} of its demand)'.format(
            s, per_scenario_lost[s] , share))

    "Integrality certificate for the relaxed battery binaries"
    if relaxed_count and Model_Components in (0, 1):
        Check_Simultaneous_Flows(instance, 'relaxed')
    elif not relaxed_count:
        print('  Binaries were left integral, so this is a solution of the full MILP as formulated.')

    # ADDED 2026-08-25 (backfilled 2026-08-27 after a local linopy-port cross-check --
    # see Code/linopy_port/linopy_core_model_partial_load_fixed_design.py -- found this
    # gap for real): Relax_Discrete_Variables relaxes EVERY remaining discrete variable,
    # not just Single_Flow_BESS. Under Generator_Partial_Load=1, that also includes
    # Generator_Full (a COUNT of fully-loaded generator units) and Generator_Partial.
    # Unlike Single_Flow_BESS, neither has a complementary-structure guarantee that a
    # fractional relaxed value is losslessly roundable: Generator_Full appears directly
    # in the fuel-cost objective and four other constraints, so this certificate had a
    # real blind spot -- a design could print "CERTIFIED FEASIBLE" (battery check only)
    # while its relaxed Generator_Full sat at a physically meaningless fractional count
    # of whole machines (e.g. 0.49995), understating the true dispatch cost. Checked
    # independently, whether or not the battery check above passed.
    if relaxed_count and Model_Components in (0, 2) and bool(value(instance.Generator_Partial_Load)):
        Check_Generator_Commitment_Integrality(instance, 'relaxed')
    print('=' * 78 + '\n')


def Check_Simultaneous_Flows(instance, context):
    """Report whether any period both charges and discharges. Returns the offender count.

    This is the whole basis on which two different shortcuts are legitimate:

      * MGPY_FIX_RELAX_BINARIES relaxes Single_Flow_BESS to [0,1]. Those binaries only
        forbid simultaneous flows and appear nowhere in the objective, so if no period
        does both, every relaxed value rounds to 0/1 with no change to dispatch or cost
        -- the relaxed answer IS a feasible point of the full MILP.
      * MGPY_BESS_FORM=nobinary removes the gate constraints outright, betting that
        simultaneous flows are uneconomic (round-trip losses, plus battery replacement
        cost charged on BOTH inflow and outflow). That bet is an argument, not a proof,
        so it has to be checked on every single run rather than once.

    A non-zero count invalidates the run. It does not mean the model is wrong -- it
    means the shortcut was not applicable this time.
    """
    inflow = instance.Battery_Inflow.get_values()
    outflow = instance.Battery_Outflow.get_values()
    tolerance = 0.001        # kWh (= 1 Wh); below this the overlap is solver noise, not a real simultaneous flow
    worst, worst_key, offenders = 0.0, None, 0
    for key, in_val in inflow.items():
        overlap = min(in_val or 0.0, outflow.get(key) or 0.0)
        if overlap > worst:
            worst, worst_key = overlap, key
        if overlap > tolerance:
            offenders += 1
    print('  Simultaneous charge/discharge check over {:,} periods:'.format(len(inflow)))
    print('    worst overlap      = {:.6g} kWh   at {}'.format(worst, worst_key))
    print('    periods above {:g} kWh = {:,}'.format(tolerance, offenders))
    if offenders == 0:
        print('  VERDICT: CERTIFIED FEASIBLE. No period both charges and discharges, so')
        print('           every Single_Flow_BESS value can be rounded to 0/1 without')
        print('           changing the dispatch or the objective. This solution is a')
        print('           feasible point of the full MILP, established without reading')
        print('           any solver bound.')
    else:
        print('  VERDICT: INVALID FOR THIS SHORTCUT. Some periods charge and discharge at')
        print('           once, which no integral Single_Flow_BESS can reproduce, so this')
        print('           objective is only a LOWER BOUND on the true dispatch cost.')
        if context == 'nobinary':
            print('           Re-run with MGPY_BESS_FORM=linear (or unset) to get a real answer.')
        else:
            print('           Re-run without MGPY_FIX_RELAX_BINARIES to settle it.')
    return offenders


def Check_Generator_Commitment_Integrality(instance, context):
    """Report whether the relaxed Generator_Full/Generator_Partial commitment variables
    came back near-integral. Returns the combined offender count.

    ADDED 2026-08-25 (backfilled 2026-08-27). Under Generator_Partial_Load=1,
    Relax_Discrete_Variables relaxes Generator_Full (a COUNT of generator units running
    at exactly 100% rated output) and Generator_Partial (whether one additional unit is
    running below full load) alongside Single_Flow_BESS. Generator_Full has NO
    complementary-structure guarantee the way Single_Flow_BESS does: it appears directly
    in the fuel-cost objective (Total_Fuel_Cost_Act/NonAct) and in four other constraints
    (Minimum/Maximum_Generator_Energy_Partial, Generator_Energy_Total, Generator_Units_
    Total), so a fractional relaxed value is not obviously roundable. A fractional
    Generator_Full does NOT mean "one unit at partial load" -- that state is already
    represented by Generator_Partial/Generator_Energy_Partial (real gensets genuinely can
    run below full load; that's the whole point of Generator_Partial_Load=1). A fractional
    COUNT of fully-loaded machines has no physical meaning at all, and the solver can use
    it to shave cost off the relaxed objective with no way to realize that saving on real
    hardware.

    A local linopy-port cross-check (Code/linopy_port/linopy_core_model_partial_load_
    fixed_design.py) found a real instance of exactly this: a design whose relaxed
    Generator_Full came back at 0.49995 despite the battery check (Check_Simultaneous_
    Flows) certifying feasible, understating the true (integral-resolve) dispatch cost by
    about 0.14%. This function is the missing check that would have caught it.
    """
    full = instance.Generator_Full.get_values()
    partial = instance.Generator_Partial.get_values()
    tolerance = 1e-4  # fractional units; below this the deviation is solver noise, not a real fractional commitment

    def worst_fraction(values):
        worst, worst_key, offenders = 0.0, None, 0
        for key, val in values.items():
            val = val or 0.0
            frac = abs(val - round(val))
            if frac > worst:
                worst, worst_key = frac, key
            if frac > tolerance:
                offenders += 1
        return worst, worst_key, offenders

    full_worst, full_key, full_offenders = worst_fraction(full)
    partial_worst, partial_key, partial_offenders = worst_fraction(partial)
    offenders = full_offenders + partial_offenders

    print('  Generator commitment integrality check (Generator_Full / Generator_Partial):')
    print('    Generator_Full:    worst fractional count = {:.6g}   at {}   ({:,}/{:,} above {:g})'.format(
        full_worst, full_key, full_offenders, len(full), tolerance))
    print('    Generator_Partial: worst fractional count = {:.6g}   at {}   ({:,}/{:,} above {:g})'.format(
        partial_worst, partial_key, partial_offenders, len(partial), tolerance))
    if offenders == 0:
        print('  VERDICT: Generator commitment relaxation came back integral -- no evidence of')
        print('           the fractional-full-unit artifact this check exists to catch.')
    else:
        print('  VERDICT: NOT INTEGRAL. Some relaxed Generator_Full/Generator_Partial values are')
        print('           genuinely fractional. Unlike Single_Flow_BESS, there is no proof this')
        print('           rounds losslessly, so the relaxed objective may UNDERSTATE the true')
        print('           dispatch cost -- even where the battery check above says feasible.')
        print('           Re-run without MGPY_FIX_RELAX_BINARIES to settle it.')
    return offenders


def Model_Resolution(model, datapath=data_file_path, options_string="mipgap=0.05",
                     warmstart=False, keepfiles=False, load_solutions=False, logfile="Solver_Output.log"):

    Data_import = open(data_file_path).readlines()  # Read the .dat file line by line to manually parse config flags

    for i in range(len(Data_import)):                        # Scan every line of the .dat file for known parameter names
        if "param: Renewable_Penetration" in Data_import[i]:
            Renewable_Penetration = float((re.findall("\d+\.\d+|\d+",Data_import[i])[0]))   # Minimum required share of renewables (float, e.g. 0.5)
        if "param: Battery_Independence" in Data_import[i]:      
            Battery_Independence = int((re.findall('\d+',Data_import[i])[0]))   # Days of battery autonomy required (0 = constraint disabled)
        if "param: Greenfield_Investment" in Data_import[i]:  
            Greenfield_Investment = int((re.findall('\d+',Data_import[i])[0]))  # 1 = new system from scratch, 0 = brownfield (existing capacity)
        if "param: Multiobjective_Optimization" in Data_import[i]:      
            Multiobjective_Optimization = int((re.findall('\d+',Data_import[i])[0]))  # 1 = run cost/emissions epsilon-constraint Pareto analysis
        if "param: Optimization_Goal" in Data_import[i]:
            Optimization_Goal = int((re.findall('\d+',Data_import[i])[0]))      # 1 = minimize NPC, 0 = minimize total variable/operation cost
        if "param: MILP_Formulation" in Data_import[i]:      
            MILP_Formulation = int((re.findall('\d+',Data_import[i])[0]))       # 1 = use mixed-integer (unit commitment) constraints, 0 = pure LP
        if "param: Plot_Max_Cost" in Data_import[i]:      
            Plot_Max_Cost = int((re.findall('\d+',Data_import[i])[0]))          # Whether to include the max-cost point when building Pareto steps
        if "param: Generator_Partial_Load" in Data_import[i]:      
            Generator_Partial_Load = int((re.findall('\d+',Data_import[i])[0])) # 1 = allow generators to run below full load (adds partial-load constraints)
        if "param: Model_Components" in Data_import[i]:      
            Model_Components = int((re.findall('\d+',Data_import[i])[0]))       # Which subsystems are modeled: 0=RES+BESS+GEN, 1=RES+BESS, 2=RES+GEN
        if "param: Land_Use" in Data_import[i]:      
            Land_Use = int((re.findall('\d+',Data_import[i])[0]))               # 1 = enforce a maximum land-area constraint for renewables
        if "param: Solver" in Data_import[i]:      
            Solver = int((re.findall('\d+',Data_import[i])[0]))                 # Which solver to use: 0=Gurobi, 1=GLPK, 2=CPLEX
        if "param: Grid_Connection " in Data_import[i]:      
            Grid_Connection = int((re.findall('\d+',Data_import[i])[0]))        # 1 = microgrid can exchange power with the main grid
        if "param: Grid_Connection_Type " in Data_import[i]:      
            Grid_Connection_Type = int((re.findall('\d+',Data_import[i])[0]))   # 0 = bidirectional (can export/sell), otherwise import-only
        if "param: Pareto_points" in Data_import[i]:      
            n = int((re.findall('\d+',Data_import[i])[0]))                      # Number of points to sample along the Pareto front
        if "param: Pareto_solution" in Data_import[i]:      
            p = int((re.findall('\d+',Data_import[i])[0]))                      # Index of the specific Pareto point to solve/return
        
    # Validate the battery formulation choice BEFORE building anything -- 'linear'
    # without its big-M would otherwise fail only after presolve.
    bess_form = Battery_Single_Flow_Form(MILP_Formulation)

    # Pick the constraint module matching the chosen investment mode (green/brownfield) and formulation (MILP/LP)
    if Greenfield_Investment == 1 and MILP_Formulation == 1 :
        from Constraints import Constraints_Greenfield_Milp as C
    if Greenfield_Investment == 0 and MILP_Formulation == 1 :
        from Constraints import Constraints_Brownfield_Milp as C
    if Greenfield_Investment == 1 and MILP_Formulation == 0 :
        from Constraints import Constraints_Greenfield as C
    if Greenfield_Investment == 0 and MILP_Formulation == 0 : 
        from Constraints import Constraints_Brownfield as C   
    
 
    
#%% Economic constraints
    model.NetPresentCost = Constraint(rule=C.Net_Present_Cost)     # Total NPC = sum of all discounted cost components
    model.CO2emission = Constraint(rule=C.CO2_emission)            # Total lifetime CO2 emissions across all sources
        
    model.ScenarioCO2emission    = Constraint(model.scenarios, 
                                              rule=C.Scenario_CO2_emission)       # CO2 emissions broken down per demand/RES scenario
    model.ScenarioNetPresentCost = Constraint(model.scenarios, 
                                              rule=C.Scenario_Net_Present_Cost)   # NPC broken down per scenario, weighted by scenario probability
    model.VariableCostNonAct = Constraint(rule=C.Total_Variable_Cost)            # Non-actualized (undiscounted) total variable cost, for reporting
    
    "Investment cost"
    model.InvestmentCost = Constraint(rule=C.Investment_Cost)      # Capex of RES/BESS/generator capacity installed across all steps
    if Optimization_Goal == 0:
        model.InvestmentCostLimit = Constraint(rule=C.Investment_Cost_Limit)   # Caps capex when minimizing operating cost instead of NPC

    "Fixed costs"    
    model.OperationMaintenanceCostAct = Constraint(rule=C.Operation_Maintenance_Cost_Act)         # Discounted O&M cost
    model.OperationMaintenanceCostNonAct = Constraint(rule=C.Operation_Maintenance_Cost_NonAct)   # Undiscounted O&M cost, for reporting

    "Variable costs"
    model.TotalVariableCostAct         = Constraint(rule=C.Total_Variable_Cost_Act)   # Discounted sum of fuel/electricity/battery/lost-load costs
    model.ScenarioVariableCostAct      = Constraint(model.scenarios,
                                                    rule=C.Scenario_Variable_Cost_Act)     # Discounted variable cost per scenario  
    model.ScenarioVariableCostNonAct   = Constraint(model.scenarios,
                                                    rule=C.Scenario_Variable_Cost_NonAct)  # Undiscounted variable cost per scenario
    # Fuel
    if Model_Components == 0 or Model_Components == 2:   # Only relevant when generators are part of the modeled system
        model.FuelCostTotalAct             = Constraint(model.scenarios, 
                                                model.generator_types,
                                                rule=C.Total_Fuel_Cost_Act)          # Discounted fuel cost per scenario/generator type
        model.FuelCostTotalNonAct          = Constraint(model.scenarios, 
                                                        model.generator_types,
                                                        rule=C.Total_Fuel_Cost_NonAct)  # Undiscounted fuel cost, for reporting
    # Grid Connection
    if Grid_Connection == 1:
        model.TotalElectricityCostAct      = Constraint(model.scenarios,
                                                        rule=C.Total_Electricity_Cost_Act)    # Discounted cost of energy imported from the grid   
        model.TotalRevenuesAct             = Constraint(model.scenarios,
                                                        rule=C.Total_Revenues_Act)            # Discounted revenue from energy exported to the grid
        if Grid_Connection_Type == 0:   # Bidirectional connection: exporting/selling back to the grid is allowed
            model.TotalRevenuesNonAct          = Constraint(model.scenarios,
                                                            rule=C.Total_Revenues_NonAct)          # Undiscounted export revenue, for reporting
            model.TotalElectricityCostNonAct   = Constraint(model.scenarios,
                                                            rule=C.Total_Electricity_Cost_NonAct)  # Undiscounted import cost, for reporting
    # Battery Replacement
    if Model_Components == 0 or Model_Components == 1:   # Only relevant when a battery (BESS) is part of the modeled system
        
        model.BatteryReplacementCostAct    = Constraint(model.scenarios,
                                                        rule=C.Battery_Replacement_Cost_Act)      # Discounted cost of replacing degraded battery capacity
        model.BatteryReplacementCostNonAct = Constraint(model.scenarios,
                                                        rule=C.Battery_Replacement_Cost_NonAct)   # Undiscounted battery replacement cost, for reporting
    # Land Use for Renewables
    if Land_Use == 1:
        model.RenewablesMaxLandUse         = Constraint(model.steps,
                                                        rule=C.Renewables_Max_Land_Use)   # Caps total land area occupied by renewable installations
    # Lost Load
    model.ScenarioLostLoadCostAct      = Constraint(model.scenarios, 
                                                    rule=C.Scenario_Lost_Load_Cost_Act)      # Discounted penalty cost for unserved/lost load
    model.ScenarioLostLoadCostNonAct   = Constraint(model.scenarios, 
                                                    rule=C.Scenario_Lost_Load_Cost_NonAct)   # Undiscounted lost-load penalty, for reporting     

    "Salvage value"
    # Floor-at-zero fix (2026-08-28): a single equality here used to pin Salvage_Value exactly
    # to the raw vintage-by-vintage residual-value formula, which has no floor of its own and
    # can go negative for a component whose lifetime is much shorter than the project horizon
    # and gets front-loaded (installed early rather than deferred) -- e.g. a 6-year generator
    # against a 20-year horizon. Since Salvage_Value's own domain is NonNegativeReals, a
    # negative raw value made the model infeasible outright instead of flooring at zero (found
    # via IIS on a real capexp5_6sc design: single constraint + single bound, raw -$8,538.38).
    # Replaced with the standard MILP linearization of Salvage_Value = max(0, raw), using the
    # new Salvage_Positive_Flag binary (Model_Creation.py). No-op whenever raw >= 0 (every
    # design solved so far at 9sc scale), since the objective already always wants
    # Salvage_Value as large as feasible (it's subtracted from NPC).
    model.SalvageValueUpperRaw  = Constraint(rule=C.Salvage_Value_Upper_Raw)    # Salvage_Value <= raw formula, only when Salvage_Positive_Flag=1
    model.SalvageValueUpperFlag = Constraint(rule=C.Salvage_Value_Upper_Flag)   # Salvage_Value <= 0, forced when Salvage_Positive_Flag=0 (i.e. whenever raw < 0)
    
#%% Brownfield additional constraints
    
    if Greenfield_Investment == 0:   # Brownfield mode: account for pre-existing capacity already installed before the optimization horizon
        model.REScapacity     = Constraint(model.scenarios, model.years_steps, 
                                           model.renewable_sources,
                                           model.periods,
                                           rule=C.RES_Capacity)   # Available RES capacity = existing + newly installed, net of any decommissioning
        if Model_Components == 0 or Model_Components == 1:
            model.BESScapacity    = Constraint(model.steps,
                                               rule=C.BESS_Capacity)   # Available battery capacity = existing + newly installed
        if Model_Components == 0 or Model_Components == 2:
            model.GENcapacity     = Constraint(model.steps,                                                
                                               model.generator_types,
                                               rule=C.GEN_Capacity)   # Available generator capacity = existing + newly installed


#%% Electricity generation system constraints 
    model.EnergyBalance = Constraint(model.scenarios,
                                     model.years_steps, 
                                     model.periods, 
                                     rule=C.Energy_balance)   # Supply (RES+BESS+GEN+grid) must meet demand in every scenario/step/period

    "Renewable Energy Sources constraints"
    model.RenewableEnergy = Constraint(model.scenarios,
                                       model.years_steps, 
                                       model.renewable_sources,
                                       model.periods, 
                                       rule=C.Renewable_Energy)  # Energy output of the solar panels
    model.ResMinStepUnits = Constraint(model.years_steps,
                                       model.renewable_sources, 
                                       rule=C.Renewables_Min_Step_Units)   # Enforces minimum RES capacity added per capacity-expansion step
    
    if Renewable_Penetration > 0:
        model.RenewableEnergyPenetration = Constraint(model.steps, 
                                                      rule=C.Renewable_Energy_Penetration)   # Enforces the minimum renewable-share target

    "Battery Energy Storage constraints"
    if Model_Components == 0 or Model_Components == 1:
        model.StateOfCharge            = Constraint(model.scenarios, 
                                                    model.years_steps,
                                                    model.periods, 
                                                    rule=C.State_of_Charge) # State of Charge of the battery
        model.MaximumCharge            = Constraint(model.scenarios,
                                                    model.years_steps, 
                                                    model.periods, 
                                                    rule=C.Maximum_Charge) # Maximun state of charge of the Battery
        model.MinimumCharge            = Constraint(model.scenarios, 
                                                    model.years_steps,
                                                    model.periods,
                                                    rule=C.Minimum_Charge) # Minimun state of charge
        model.MaxPowerBatteryCharge    = Constraint(model.steps, 
                                                    rule=C.Max_Power_Battery_Charge)  # Max power battery charge constraint
        model.MaxPowerBatteryDischarge = Constraint(model.steps,
                                                    rule=C.Max_Power_Battery_Discharge)    # Max power battery discharge constraint

        if MILP_Formulation:   # Integer formulation needs binary variables to forbid simultaneous charge/discharge
            # In the historical bilinear form ONE constraint carries both the C-rate
            # cap and the mode gate. The alternative forms split them, so the cap must
            # be registered explicitly -- omit it and the battery could discharge its
            # entire capacity in a single hour with no error raised anywhere.
            if bess_form in ('linear', 'nobinary'):
                model.BatteryFlowDischargeCap = Constraint(model.scenarios,
                                                        model.years_steps,
                                                        model.periods,
                                                        rule=C.Max_Bat_flow_out_milp)   # C-rate limit on discharge, independent of mode
                model.BatteryFlowChargeCap    = Constraint(model.scenarios,
                                                        model.years_steps,
                                                        model.periods,
                                                        rule=C.Max_Bat_flow_in_milp)    # C-rate limit on charge, independent of mode
            if bess_form != 'nobinary':   # 'nobinary' drops the gate, and with it the binaries
                model.BatterySingleFlowDischarge = Constraint(model.scenarios,
                                                        model.years_steps,
                                                        model.periods,
                                                        rule=C. Battery_Single_Flow_Discharge)   # Discharge only allowed when the discharge binary is active
                model.BatterySingleFlowCharge = Constraint(model.scenarios,
                                                        model.years_steps,
                                                        model.periods,
                                                        rule=C. Battery_Single_Flow_Charge)      # Charge only allowed when the charge binary is active
        else:
            model.BatteryFlowCharge                     = Constraint(model.scenarios,
                                                                    model.years_steps,
                                                                    model.periods, 
                                                                    rule=C.Max_Bat_flow_in) # Minimun flow of energy for the charge fase
            model.BatteryFlowDischarge                 = Constraint(model.scenarios,
                                                                    model.years_steps,
                                                                    model.periods, 
                                                                    rule=C.Max_Bat_flow_out) # Minimun flow of energy for the discharge fase
        model.Maxbatout                = Constraint(model.scenarios, 
                                                    model.years_steps, 
                                                    model.periods,
                                                    rule=C.Max_Bat_out) #minimun flow of energy for the discharge fase
        model.BatteryMinStepCapacity   = Constraint(model.years_steps,                                             
                                                    rule=C.Battery_Min_Step_Capacity)   # Enforces minimum battery capacity added per capacity-expansion step
            
        if Battery_Independence > 0:
            model.BatteryMinCapacity   = Constraint(model.steps, 
                                                    rule=C.Battery_Min_Capacity)   # Enforces the required autonomy (days of storage) target

    "Diesel generator constraints"
    if Model_Components == 0 or Model_Components == 2:
        
        if MILP_Formulation == 1 and Generator_Partial_Load == 1:   # MILP with generators allowed to run below full load
           model.MinimumGeneratorEnergyPartial     = Constraint(model.scenarios, 
                                                                model.years_steps, 
                                                                model.generator_types,
                                                                model.periods, 
                                                                rule=C.Minimum_Generator_Energy_Partial)   # Lower bound on output while a generator unit is committed
           model.MaximumGeneratorEnergyPartial     = Constraint(model.scenarios, 
                                                                model.years_steps, 
                                                                model.generator_types,
                                                                model.periods, 
                                                                rule=C.Maximum_Generator_Energy_Partial)   # Upper bound on output while a generator unit is committed
           model.MaximumGeneratorEnergyTotal1       = Constraint(model.scenarios,
                                                                model.years_steps, 
                                                                model.generator_types,
                                                                model.periods, 
                                                                rule=C.Maximum_Generator_Energy_Total_1)   # Links total output to number of committed units (bound 1)
           model.MaximumGeneratorEnergyTotal2       = Constraint(model.scenarios,
                                                                model.years_steps, 
                                                                model.generator_types,
                                                                model.periods, 
                                                                rule=C.Maximum_Generator_Energy_Total_2)   # Links total output to number of committed units (bound 2)
           model.GeneratorEnergyTotal              = Constraint(model.scenarios,
                                                                model.years_steps, 
                                                                model.generator_types,
                                                                model.periods, 
                                                                rule=C.Generator_Energy_Total)   # Total generator energy = sum of full-load + partial-load output
           model.GeneratorUnitsTotal               = Constraint(model.scenarios,
                                                                model.years_steps, 
                                                                model.generator_types,
                                                                model.periods, 
                                                                rule=C.Generator_Units_Total)   # Total committed units cannot exceed installed generator capacity
           model.GeneratorMinStepCapacity          = Constraint(model.years_steps,
                                                                model.generator_types,
                                                                rule=C.Generator_Min_Step_Capacity)   # Enforces minimum generator capacity added per capacity-expansion step
           model.GeneratorMaxUnits                 = Constraint(model.years_steps,
                                                                model.generator_types,
                                                                rule=C.Generator_Max_Units)   # ADDED (2026-08-23): opt-in ceiling on installed generator capacity, via MGPY_MAX_GENERATOR_KW -- no-op unless set
        elif MILP_Formulation == 1 and Generator_Partial_Load == 0:   # MILP but generators must run at full load or off
           model.MaximumGeneratorEnergyTotal1       = Constraint(model.scenarios, 
                                                                model.years_steps, 
                                                                model.generator_types,
                                                                model.periods, 
                                                                rule=C.Maximum_Generator_Energy_Total_1)   # Links total output to number of committed units (bound 1)
           model.MaximumGeneratorEnergyTotal2       = Constraint(model.scenarios, 
                                                                model.years_steps, 
                                                                model.generator_types,
                                                                model.periods, 
                                                                rule=C.Maximum_Generator_Energy_Total_2)   # Links total output to number of committed units (bound 2)
           model.GeneratorMinStepCapacity = Constraint(model.years_steps,
                                                       model.generator_types,
                                                       rule=C.Generator_Min_Step_Capacity)   # Enforces minimum generator capacity added per capacity-expansion step
           model.GeneratorMaxUnits = Constraint(model.years_steps,
                                               model.generator_types,
                                               rule=C.Generator_Max_Units)   # ADDED (2026-08-23): opt-in ceiling on installed generator capacity, via MGPY_MAX_GENERATOR_KW -- no-op unless set
        else:   # Pure LP: generator output is a continuous variable with no commitment logic
           model.MaximumGeneratorEnergy1        = Constraint(model.scenarios, 
                                                       model.years_steps, 
                                                       model.generator_types,
                                                       model.periods, 
                                                       rule=C.Maximum_Generator_Energy_1)   # Output cannot exceed installed generator capacity (bound 1)
           model.MaximumGeneratorEnergy2        = Constraint(model.scenarios, 
                                                       model.years_steps, 
                                                       model.generator_types,
                                                       model.periods, 
                                                       rule=C.Maximum_Generator_Energy_2)   # Output cannot exceed installed generator capacity (bound 2)
           model.GeneratorMinStepCapacity = Constraint(model.years_steps, 
                                                       model.generator_types, 
                                                       rule=C.Generator_Min_Step_Capacity)   # Enforces minimum generator capacity added per capacity-expansion step         
    "Grid constraints" 
    if Grid_Connection == 1:
        model.MaximumPowerFromGrid     = Constraint(model.scenarios,
                                                model.years,
                                                model.periods,
                                                rule=C.Maximum_Power_From_Grid)   # Caps power imported from the grid to the connection's rated capacity
        if Grid_Connection_Type == 0:
            model.MaximumPowerToGrid       = Constraint(model.scenarios,
                                                model.years,
                                                model.periods,
                                                rule=C.Maximum_Power_To_Grid)   # Caps power exported/sold to the grid to the connection's rated capacity
        if MILP_Formulation:   # Prevent simultaneous import and export in the same period via binary flow-direction variables
            if Grid_Connection_Type == 0:
                model.SingleFlowEnergyToGrid      = Constraint(model.scenarios,
                                                     model.years_steps,
                                                     model.periods,
                                                     rule=C.Single_Flow_Energy_To_Grid)   # Export only allowed when the export binary is active
            model.SingleFlowEnergyFromGrid        = Constraint(model.scenarios,
                                                     model.years_steps,
                                                     model.periods,
                                                     rule=C.Single_Flow_Energy_From_Grid)   # Import only allowed when the import binary is active

    "Lost load constraints"
    model.MaximumLostLoad = Constraint(model.scenarios, model.years, 
                                       rule=C.Maximum_Lost_Load) # Maximum permissible lost load

    "Emission constrains"
    model.RESemission    = Constraint(rule=C.RES_emission)   # Lifecycle emissions attributable to renewable generation (usually near zero)
    if Model_Components == 0 or Model_Components == 2:
        model.GENemission    = Constraint(rule=C.GEN_emission)   # Lifecycle/manufacturing emissions attributable to generator capacity
        model.FUELemission   = Constraint(model.scenarios, 
                                          model.years_steps, 
                                          model.generator_types,
                                          model.periods,
                                          rule=C.FUEL_emission)   # Combustion emissions from fuel burned by generators
        model.ScenarioFUELemission = Constraint(model.scenarios,
                                                rule=C.Scenario_FUEL_emission)   # Total fuel emissions aggregated per scenario
    if Model_Components == 0 or Model_Components == 1:
        model.BESSemission   = Constraint(rule=C.BESS_emission)   # Lifecycle/manufacturing emissions attributable to battery capacity
    
    if Grid_Connection == 1:
        model.GRIDemission = Constraint(model.scenarios, 
                                    model.years,
                                    model.periods,
                                    rule=C.GRID_emission)   # Emissions embedded in energy imported from the grid
        model.ScenarioGRIDemission = Constraint(model.scenarios,
                                            rule=C.Scenario_GRID_emission)   # Total grid emissions aggregated per scenario

##############################################################################################################################################################
     
    if Multiobjective_Optimization == 0:   # Single-objective mode: solve once for cost only (no cost/emissions Pareto sweep)
        if Optimization_Goal == 1:
            model.ObjectiveFuntion = Objective(rule=C.Net_Present_Cost_Obj, 
                                               sense = minimize)   # Minimize total discounted Net Present Cost
        elif Optimization_Goal == 0:
            model.ObjectiveFuntion = Objective(rule=C.Total_Variable_Cost_Obj, 
                                               sense = minimize)   # Minimize total discounted variable/operating cost only

        instance = model.create_instance(datapath) # load parameters

        print('\nInstance created')

        # Opt-in design verification -- see the block above this function for why
        # this exists and why the relaxation is still a proof. Unset MGPY_FIX_*
        # leaves everything below untouched.
        # Opt-in finite ceiling on battery size -- see Bound_Battery_Size for why
        # the unbounded default is suspected of producing the invalid root cuts.
        max_battery_kwh = Bound_Battery_Size(instance, MILP_Formulation)
        if max_battery_kwh:
            print('\nMGPY_MAX_BATTERY_KWH={:g}: battery sizing bounded above (was unbounded).'
                  .format(max_battery_kwh))

        fix_design = Fix_Design_Requested()
        relaxed_count = 0
        if fix_design:
            print('\n' + '=' * 78)
            print('FIXED-DESIGN VERIFICATION -- SIZING PINNED, DISPATCH FREE')
            print('=' * 78)
            for line in Fix_Design(instance, MILP_Formulation, Model_Components):
                print(line)
            if os.environ.get('MGPY_FIX_RELAX_BINARIES') == '1':
                relaxed_count = Relax_Discrete_Variables(instance)
                print('  Relaxed {:,} remaining discrete variables to continuous.'.format(relaxed_count))
                print('  The post-solve check certifies whether that relaxation was lossless.')
            else:
                print('  Remaining discrete variables left integral (still a MILP).')
            print('=' * 78)


        if Solver == 0:
            opt = SolverFactory('gurobi') # Solver use during the optimization
            # Match Gurobi's thread count to whatever Slurm actually allocated, rather than
            # hardcoding 12. Previously this was a fixed 12, which silently ignored any larger
            # --cpus-per-task request (job 58513174 logged "Thread count was 12 (of 128
            # available processors)" while asking for 12 cores). Falls back to 12 for local runs
            # where SLURM_CPUS_PER_TASK is unset.
            opt.options['Threads'] = int(os.environ.get('SLURM_CPUS_PER_TASK', 12))

            # Setting options for Gurobi
            if MILP_Formulation:
                opt.options['Method'] = 2             # barrier for the root relaxation; Method=3 (concurrent) was tested and found to waste wall-clock time racing a simplex thread that barrier reliably beats on this problem
                opt.options['BarHomogeneous'] = 1     # Use the more robust homogeneous barrier algorithm (handles infeasible/unbounded root relaxations better)
                # CROSSOVER. Default 1 = historical behaviour, unchanged.
                # The earlier revert note stands for the case it was measured on: Crossover=0
                # (with NodeMethod=2 to fully disable it) was trialed at 9-scenario/2-year and made
                # cut-generation progress worse, not faster -- gap froze at 24.2% for 43+ minutes with
                # zero movement, versus the Cuts-only trial closing 94.1%->13.6% over a comparable
                # window. That trial measured cut generation INSIDE B&B, where crossover had already
                # finished cheaply and the cost of a non-basic solution showed up later.
                # It does NOT transfer to 6sc/10y, where the phase breakdown (2026-07-26) puts
                # crossover + root simplex at ~29,000s against barrier's 1,220s -- ~96% of root time,
                # with B&B never starting at all. Hence env-configurable rather than hardcoded.
                #   MGPY_CROSSOVER    1 = historical default; 0 = skip the push to a basic solution
                #   MGPY_NODEMETHOD   unset = Gurobi automatic (default); 2 = barrier at every node
                # NOTE the coupling: for a MIP, Crossover=0 on its own is not enough -- node LPs are
                # warm-started from a basis, so Gurobi still crosses over at the root unless
                # NodeMethod=2 also removes the need for one. Setting only MGPY_CROSSOVER=0 is
                # therefore likely to be a no-op. Setting both genuinely skips it, at the price of
                # solving every B&B node by barrier, which does not warm-start. See the sbatch
                # scripts for why that trade is worth testing anyway.
                opt.options['Crossover'] = int(os.environ.get('MGPY_CROSSOVER', 1))
                _node_method = os.environ.get('MGPY_NODEMETHOD')
                if _node_method:
                    opt.options['NodeMethod'] = int(_node_method)
                # MGPY_DEGENMOVES (2026-08-30) -- disables Gurobi's post-root-relaxation
                # "degenerate simplex moves" polishing phase (see Gurobi support: a model with an
                # optimal LP face rather than a single vertex can spend excessive time here before
                # cut generation/B&B even starts). Added after the linopy cross-check port found
                # this genuinely helped its own free-sizing degeneracy stall (gili_ketapang_
                # checklist_dlensemble.md §8) -- this project's Pyomo pipeline never had this
                # lever at all before now. Unset (default) leaves Gurobi's own default behaviour
                # unchanged, same "only touch if set" pattern as MGPY_NODEMETHOD above.
                _degen_moves = os.environ.get('MGPY_DEGENMOVES')
                if _degen_moves:
                    opt.options['DegenMoves'] = int(_degen_moves)
                # MIPFocus/Cuts are env-configurable so competing tunings can run as concurrent jobs
                # without editing this file between submissions (which would race). Defaults reproduce
                # the historical settings exactly, so omitting the env vars changes nothing.
                #   MIPFocus 1 = find good incumbents (default, tuned for the 9sc/20y stall)
                #   MIPFocus 3 = focus on the best bound -- the actual bottleneck at 6sc/5y, where the
                #                incumbent improved 31.6% while the bound moved 0.65% (job 58513174)
                opt.options['MIPFocus'] = int(os.environ.get('MGPY_MIPFOCUS', 1))
                # Both default to 1e-3, the historical values, so nothing changes when
                # they are unset. They are env-configurable because a fixed-design
                # verification run (MGPY_FIX_*) is answering a different question from a
                # sizing run: there the objective is the answer, not a waypoint, and it
                # should not be left to a tolerance loose enough to stop simplex early.
                opt.options['BarConvTol'] = float(os.environ.get('MGPY_BARCONVTOL', 1e-3))    # Relative barrier convergence tolerance for the root LP relaxation
                opt.options['OptimalityTol'] = float(os.environ.get('MGPY_OPTTOL', 1e-3))     # Simplex dual feasibility (reduced cost) tolerance
                # NUMERICAL CONDITIONING. "Matrix range [8e-09, 3e+06]" (~15 orders of
                # magnitude) was the PRE-conversion figure, from before the 162-site kW/kWh
                # unit conversion (Full_Progress_Report/main.tex sec:units) was applied to
                # this codebase. Post-conversion, free-sizing diagnostic runs on
                # 2026-08-24 (see logs/*_scale_diagnostic*, e.g.
                # 20260824_182648_6sc_10y_gili_ketapang_MILP_dieselops_scale_diagnostic)
                # measure "Matrix range [3e-03, 3e+05]" to "[3e-03, 5e+05]" -- a ~1e8 span,
                # down 6 orders of magnitude, with no NumericFocus advisory from Gurobi on
                # any of these runs. Still above Gurobi's "ideally below 1e6" guidance, but
                # no longer the ~15-order-of-magnitude problem described below. CORRECTED
                # (2026-08-24, see Full_Progress_Report/main.tex
                # sec:m-certificate/sec:m-cause -- this comment originally reasoned the wrong way):
                # job 58513174's incumbent of 106,342 being BELOW job 58519303's reported "proven"
                # 109,629 (bound == objective, gap 0.0000%) was NOT evidence 106,342 was infeasible --
                # it was evidence 109,629 was never actually optimal. It wasn't: 109,629 was itself an
                # invalid bound, produced by the bilinear big-M battery constraint generating
                # numerically invalid MIR/relax-and-lift cuts (root-caused and fixed via
                # MGPY_BESS_FORM=linear, sec:m-cause). The project's own fixed-design verification
                # later established the true optimum as 96,532.96 -- below every one of these
                # reported numbers, confirming the "lower incumbent must be infeasible" reasoning was
                # backwards, not just wrong about this one case. Moral for this codebase generally: a
                # 0.0000% gap is evidence, not proof, on this model -- see sec:trust.
                #   MGPY_NUMERICFOCUS  0 = Gurobi automatic (default, = historical behaviour)
                #                      1-3 = progressively more careful arithmetic, progressively slower
                #   MGPY_FEASTOL       1e-4 = historical default here; 1e-6 is Gurobi's own default
                # Defaults preserve past behaviour so earlier runs stay reproducible; set both
                # explicitly in the sbatch script for any run whose numbers you intend to publish.
                opt.options['FeasibilityTol'] = float(os.environ.get('MGPY_FEASTOL', 1e-4))
                _numeric_focus = int(os.environ.get('MGPY_NUMERICFOCUS', 0))
                if _numeric_focus:
                    opt.options['NumericFocus'] = _numeric_focus
                # Stop once the remaining optimality gap is this small. Default 0.03
                # preserves historical behaviour; loosened from 0.01 originally because
                # the 9-scenario/20-year run stalled for hours in root-node cut
                # generation before B&B even started.
                #
                # MEASURED COST OF THIS TOLERANCE (job 58547964, 6sc/5y, 2026-07-26):
                #   gap 4.91% reached at  4,099s  with objective 105,473.16
                #   gap 2.55% reached at 33,805s  with objective 103,035.96
                # i.e. the last ~2.4% of objective cost ~8x the wall-clock time. Use
                # 0.05 to screen a new site quickly and 0.03 or tighter for a number
                # that gets published.
                #
                # NOTE the gap is only as trustworthy as the BOUND behind it. Job
                # 58519303 reported gap 0.0000% with a bound of 109,629.05, and a
                # feasible solution at 103,035.96 was later found -- so the bound was
                # wrong and the "proven optimum" was not optimal. A small gap is
                # evidence, not proof, on this badly-conditioned model.
                opt.options['MIPGap'] = float(os.environ.get('MGPY_MIPGAP', 0.03))
                # Once B&B node storage exceeds this many GB, spill node data to disk instead of
                # RAM, trading some speed for not crashing on large models. Default 0.5 preserves
                # historical behaviour. Env-configurable (2026-08-25) because #13
                # (gili_ketapang_9sc_20y_MILP_dieselops) has OOM-crashed 5x deep in branching even
                # at a 350GB ceiling (checklist item #14's retry) -- if spillover isn't keeping pace
                # with how fast Gurobi opens nodes at this scale, a much lower threshold (e.g. 0.05)
                # forces more aggressive disk offloading, trading per-node speed for headroom, without
                # capping the search itself the way MGPY_TIMELIMIT banking would. Untested as of this
                # writing -- try a low value (MGPY_NODEFILESTART=0.05) on #13 before assuming it helps.
                opt.options['NodefileStart'] = float(os.environ.get('MGPY_NODEFILESTART', 0.5))
                # NodefileStart decides *when* to spill; NodefileDir decides *where*. Without this,
                # Gurobi writes node files into the current working directory -- job 58517429 logged
                # "Created node file directory './grbnodes0'" under Code/Model/, which on HPC sits
                # inside the 15GB HOME quota. scratch2 has 3TB and is the intended location for large
                # regenerable intermediates. Guarded so local (non-HPC) runs skip it entirely and fall
                # back to Gurobi's default rather than creating a stray /scratch2 path off the drive root.
                _scratch_root = '/scratch2/<project>'
                if os.path.isdir(_scratch_root):
                    try:
                        # RUN_ID keeps concurrent jobs from sharing a node-file directory -- two
                        # simultaneous solves both writing 'grbnodes0' under the same parent would
                        # collide. RUN_ID is timestamp+tag, so it is unique per submission.
                        _nodefile_dir = os.path.join(_scratch_root, getpass.getuser(), 'grbnodes', RUN_ID)
                        os.makedirs(_nodefile_dir, exist_ok=True)
                        opt.options['NodefileDir'] = _nodefile_dir
                    except OSError:
                        pass                  # not writable (quota, permissions) -- leave Gurobi's default in place
                # Cuts 1 = conservative (default here): dialled back from Gurobi's aggressive root-node
                # cut generation, which is what the stalled 9-scenario/20-year run was stuck in for 9+
                # hours with zero nodes explored. Note the tension: cuts are what TIGHTEN THE BOUND, so
                # at 6sc/5y -- where B&B runs fine and the bound is the bottleneck -- Cuts=2 (aggressive)
                # or -1 (automatic) may be the better setting. Env-configurable for the same reason as
                # MIPFocus above.
                opt.options['Cuts'] = int(os.environ.get('MGPY_CUTS', 1))
                # CutPasses: max number of cutting-plane rounds at the root (and, with
                # Gurobi's default GomoryPasses etc., at nodes). Left at Gurobi's automatic
                # choice (-1) unless explicitly set. Added 2026-08-25, env-configurable for
                # the same reason as MGPY_CUTS -- a low value (e.g. 1) caps how much time/
                # memory root cut generation can consume before B&B proper starts, worth
                # testing on #14 (gili_ketapang_6sc_20y_MILP_dieselops) alongside Cuts=1.
                _cut_passes = os.environ.get('MGPY_CUTPASSES')
                if _cut_passes:
                    opt.options['CutPasses'] = int(_cut_passes)
                # Heuristics: fraction of run time Gurobi spends on its own internal
                # rounding/diving heuristics, independent of Cuts/Crossover/NodeMethod.
                # Left at Gurobi's own default (~0.05) unless explicitly overridden --
                # same "only touch if set" pattern as MGPY_NUMERICFOCUS above, since this
                # is a diagnostic, not a tuned default. Added 2026-07-28 to test a NEW
                # stall pattern seen at 6sc/20y (job 58592236): unlike the previously-
                # solved crossover-stall and MIR-cut-corruption issues (both already
                # guarded against above), that run sat at the root node (Depth 0,
                # Expl=Unexpl=0) for 3.5+ hours with the bound frozen, while repeatedly
                # finding new incumbents via Gurobi's own heuristics (the "H" lines in
                # the log) -- consistent with heuristic search itself being the time
                # sink, not cut generation. MGPY_HEURISTICS=0 tests that hypothesis
                # directly; not yet confirmed to help, this is the first test of it.
                _heuristics = os.environ.get('MGPY_HEURISTICS')
                if _heuristics:
                    opt.options['Heuristics'] = float(_heuristics)
                # Presolve=0 was tested here 2026-07-24 (6sc/10y size) and made things worse, not
                # better: without presolve, Gurobi can't simplify the model's 1,051,200 quadratic
                # constraints (generator/battery commitment) and instead reformulates them into a
                # much larger problem (6.5M continuous + 1.87M integer vars, up from 4.7M/525K) --
                # ran out of memory in 47s. Unlike the LP case, presolve is load-bearing here. Do
                # not re-add without new evidence.
                # NoRelHeurTime=300 was trialed 2026-07-20 (job 58401804, 9sc/2y size) on top of Cuts=1/MIPGap=0.03: gave a much better/faster initial incumbent (108,689 by t=713s vs. 1.5M/94.1% gap for Cuts-only), but overall converged worse -- hit its own TimeLimit at 33,682s (~9.36h) still at 6.20% gap, while the Cuts-only baseline (58370842) fully converged to 2.81% in 19,873s (~5.52h). Root bound froze at 89,872.05 from t=8,719s to the end (~6.9h of zero bound movement) -- all late-stage gap closing came from incumbent-side heuristics only. Net negative at this trial size; reverted. Don't re-add without new evidence.
            else:
                opt.options['Method'] = 2             # Barrier method for solving the pure LP
                opt.options['BarHomogeneous'] = 0     # Standard (non-homogeneous) barrier algorithm is sufficient for the LP relaxation
                opt.options['Crossover'] = 0           # Skip crossover to simplex basis; keep the barrier's interior-point solution directly
                # Env-configurable. Default 1e-4 preserves historical behaviour. Note the
                # LP branch also sets Crossover=0, so the reported answer is an INTERIOR
                # point, not a vertex, accurate only to this relative tolerance. Two
                # algebraically identical models can therefore land on objectives that
                # differ at the 1e-4 level without either being "wrong". Tighten this
                # before concluding that a modelling change altered the optimum.
                opt.options['BarConvTol'] = float(os.environ.get('MGPY_BARCONVTOL', 1e-4))       # Relative barrier convergence tolerance
                opt.options['OptimalityTol'] = 1e-4    # Simplex dual feasibility (reduced cost) tolerance
                # Env-configurable, matching the MILP branch. Default 1e-4 preserves
                # historical behaviour. Tightening this is how you test whether a
                # result depends on tolerance: an answer that MOVES when the tolerance
                # is tightened was exploiting slack in a badly-scaled constraint.
                opt.options['FeasibilityTol'] = float(os.environ.get('MGPY_FEASTOL', 1e-4))   # Primal constraint feasibility tolerance
                # Default 0 preserves historical behaviour (disabled), set at 5y/10y scale where
                # Presolve=1 gave near-identical timing to automatic -- inconclusive on aggressiveness,
                # never re-tested against the true default (-1) at larger size. Unlike the MILP branch
                # (where Presolve=0 is the deliberate choice -- that model's 1,051,200 quadratic
                # constraints from the bilinear battery formulation are what presolve chokes on), this
                # LP branch has ZERO quadratic constraints, so that rationale doesn't transfer here.
                # At 6sc/20y the model reaches barrier fully unreduced at 12.6M rows, and per-iteration
                # cost was observed climbing sharply near the (apparently degenerate) optimal face --
                # worth testing whether presolve shrinking/simplifying the model first changes that.
                opt.options['Presolve'] = int(os.environ.get('MGPY_PRESOLVE', 0))

            # SCALING -- applies to BOTH formulations, which is the point of putting it
            # here rather than inside either branch. [1e-08, 3e+06] (ratio 3e14) was the
            # PRE-conversion coefficient spread, from power/energy held in W/Wh against
            # cost given per kW/kWh. DONE: the 162-site kW/kWh unit conversion (see
            # Full_Progress_Report/main.tex sec:units) has since been applied across
            # Constraints/Results/Model_Resolution/Plots and verified to 9 significant
            # figures against a 920-cell golden-output diff -- this was "the real fix"
            # this comment used to point at as future work; it is not future work
            # anymore. Post-conversion free-sizing diagnostics (2026-08-24, see
            # logs/*_scale_diagnostic*) measure a residual [3e-03, 5e+05] span (~1e8,
            # still above Gurobi's "ideally below 1e6" guidance but 6 orders of
            # magnitude better), with no NumericFocus advisory from Gurobi on any of
            # those runs. ScaleFlag remains available below as a zero-risk lever for
            # that residual gap if a future run does trigger a numerical warning: no
            # data changes, no results changes, no plot changes.
            #
            # Because it is set for both branches, the LP can be used as the cheap test:
            # it solves in ~6 min against the MILP's ~3 h, on the same coefficient range,
            # with a known reference answer of NPC 96,136.
            #   unset / -1 = automatic (default, = historical behaviour)
            #   0 = none, 1 = equilibrium, 2 = geometric mean, 3 = aggressive
            _scale_flag = os.environ.get('MGPY_SCALEFLAG')
            if _scale_flag:
                opt.options['ScaleFlag'] = int(_scale_flag)

            # SOLUTION EXPORT. Set MGPY_WRITE_SOL=1 to have Gurobi write the full
            # solution vector to Solution_<RUN_ID>.sol alongside the run.
            #
            # This exists because a reported objective is not self-verifying on this
            # model. Job 58519303 reported 109,629.05 as proven optimal with gap
            # 0.0000%; a feasible solution at 103,035.96 was later found, so its BOUND
            # was wrong. The independent check is to take the solution values, fix
            # every variable to them, re-solve, and confirm feasibility -- a check that
            # never consults the bound. That needs the solution vector, which is
            # otherwise discarded: Results.py writes aggregated figures, not the
            # variable values.
            #
            # Cheap to enable (one file write at the end) and worth setting on any run
            # whose numbers are intended for publication.
            if os.environ.get('MGPY_WRITE_SOL') == '1':
                opt.options['ResultFile'] = os.path.abspath(
                    os.path.join(current_directory, 'Solution_{}.sol'.format(RUN_ID)))

            # Stop Gurobi cleanly before the Slurm walltime kills it outright.
            # scancel/SIGINT was tried twice this session to recover a partial
            # solution from a stalled solve and never worked -- confirmed via web
            # research this is a known Gurobi+Slurm gap: the Python gurobipy API
            # doesn't listen for SIGINT unless the application installs its own
            # signal handler calling model.terminate(), and Slurm's signal often
            # doesn't even reach the child process through a plain bash sbatch
            # wrapper. TimeLimit sidesteps both problems entirely: Gurobi checks
            # its own clock as a normal part of the solve loop and always returns
            # its best incumbent when it hits the limit, no external signal needed.
            slurm_end_time = os.environ.get('SLURM_JOB_END_TIME')  # epoch seconds; set by Slurm for batch jobs, absent for local/interactive runs
            if slurm_end_time is not None:
                safety_margin = 1800  # stop 30 min before the walltime kill, leaving time for opt.solve() to return and the results-writing pipeline (Results.py, PrintResults, plots) to run
                remaining = int(slurm_end_time) - time.time()
                if remaining > safety_margin:
                    opt.options['TimeLimit'] = remaining - safety_margin
            # Explicit limit in seconds, for runs with no Slurm clock to read. This
            # exists for LOCAL MILP runs: memory here grows with the B&B tree, and
            # the machine has 31.6 GB against the 42 GB a full 6sc/5y solve reached
            # on the cluster. MIPGap is not a safe stopping rule for that -- a run
            # whose bound stays honest does NOT hit the gap early and keeps
            # branching. Overrides the Slurm-derived value when both are set.
            _time_limit = os.environ.get('MGPY_TIMELIMIT')
            if _time_limit:
                opt.options['TimeLimit'] = float(_time_limit)

            # Gurobi checks its own memory usage periodically and stops gracefully,
            # returning the best incumbent found, once it exceeds this -- the same
            # graceful-stop pattern as TimeLimit above, but for memory instead of
            # wall-clock. Exists because an OOM kill from the OS/cgroup (job 59314325,
            # 2026-08-20: killed at ~208.7GB against a 200G Slurm allocation, root node
            # never left, Cuts=1) returns nothing at all -- no incumbent, no log tail, no
            # Results/ folder, the entire run is a total loss. Set a bit below your actual
            # cgroup/Slurm memory limit (e.g. 180 against a 200G allocation) so Gurobi's
            # own check fires before the OS's does. Unset (default) preserves historical
            # behaviour -- Gurobi's own default is unlimited.
            _mem_limit = os.environ.get('MGPY_MEMLIMIT')
            if _mem_limit:
                opt.options['MemLimit'] = float(_mem_limit)

            # Diagnostic only (added 2026-08-15, opt-in): preserve the .lp/.sol files Pyomo
            # normally deletes after solve, in a known local directory instead of a random
            # system temp path, so they can be independently re-solved via raw gurobipy --
            # the only way to tell whether a solver-reported value was already wrong inside
            # Gurobi's own solution, versus corrupted afterward during Pyomo's read-back of
            # that solution file. Off by default; zero effect on any existing run.
            if os.environ.get('MGPY_KEEPFILES') == '1':
                from pyomo.common.tempfiles import TempfileManager
                keepfiles_dir = os.environ.get('MGPY_KEEPFILES_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keepfiles_diagnostic'))
                os.makedirs(keepfiles_dir, exist_ok=True)
                TempfileManager.tempdir = keepfiles_dir
                keepfiles = True
                print(f'MGPY_KEEPFILES=1: solver .lp/.sol files will be kept in {keepfiles_dir}')

            print('Calling GUROBI solver...')
            results = opt.solve(instance, tee=True, warmstart=warmstart, keepfiles=keepfiles,
                            load_solutions=load_solutions, logfile=logfile) # Solving a model instance

        elif Solver == 1:
           opt = SolverFactory('glpk') # Solver use during the optimization
           timelimit = 10000
           opt.options['tmlim'] = timelimit   # Wall-clock time limit in seconds
           if MILP_Formulation: 
               opt.options['mipgap'] = 0.01      # Set relative gap tolerance for MIP
               opt.options['clq_cuts'] = 'on'  # Enable clique cuts
           
           print('Calling GLPK solver...')
           results = opt.solve(instance, tee=True, keepfiles=keepfiles, logfile=logfile) # Solving a model instance
           
        elif Solver == 2:
            opt = SolverFactory('cplex') # Solver used during the optimization

            # Setting options for CPLEX
            if MILP_Formulation:
                opt.options['mip.tolerances.mipgap'] = 0.01        # Relative MIP optimality gap tolerance
                opt.options['mip.cuts.cliques'] = 2                # Aggressive clique-cut generation
                opt.options['mip.tolerances.absmipgap'] = 1e-3     # Absolute MIP optimality gap tolerance
                opt.options['mip.tolerances.integrality'] = 1e-4   # Tolerance for a variable to be considered integral
            else:
                opt.options['lpmethod'] = 2                              # Barrier method for solving the LP
                opt.options['barrier.convergetol'] = 1e-4                # Barrier convergence tolerance
                opt.options['simplex.tolerances.optimality'] = 1e-4      # Simplex dual feasibility (reduced cost) tolerance
                opt.options['simplex.tolerances.feasibility'] = 1e-4     # Primal constraint feasibility tolerance

            opt.options['timelimit'] = 10000   # Wall-clock time limit in seconds

            print('Calling CPLEX solver...')
            results = opt.solve(instance, tee=True, warmstart=warmstart, keepfiles=keepfiles,
                            load_solutions=load_solutions, logfile=logfile) # Solving a model instance
           
        print('Instance solved')

        # A fixed-design run that comes back infeasible is a RESULT, not a crash, so
        # say what happened before Pyomo raises on the empty solution set. Note the
        # ordering advice: a wrong variable or wrong units look exactly like a
        # genuinely infeasible design, so re-read the SIZING PINNED block above first.
        if fix_design:
            termination = str(results.solver.termination_condition)
            if termination not in ('optimal', 'feasible', 'globallyOptimal', 'locallyOptimal', 'maxTimeLimit'):
                print('\n' + '=' * 78)
                print('FIXED-DESIGN VERIFICATION -- NO SOLUTION ({})'.format(termination))
                print('=' * 78)
                print('  Before concluding the design is infeasible, check the SIZING PINNED')
                print('  block above: a wrong variable or a factor of 1000 presents identically.')
                print('  To locate the binding constraints, set MGPY_FIX_DUMP_LP=1 and re-run,')
                print('  then load the .lp in gurobipy and call Model.computeIIS().')
                if os.environ.get('MGPY_FIX_DUMP_LP') == '1':
                    lp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Verification_{}.lp'.format(RUN_ID))
                    instance.write(lp_path, io_options={'symbolic_solver_labels': True})
                    print('  Wrote {}'.format(lp_path))
                print('=' * 78 + '\n')
                raise SystemExit(2)
        else:
            # Same gap as above, general case. Confirmed 2026-07-27: 20y LP job 58578242
            # ran 8835s, reached barrier iteration 253 with Compl converging normally
            # (2.39e-05), then hit TimeLimit and crashed on the unconditional load_from()
            # below with "Cannot load a SolverResults object with bad status: aborted".
            # Crossover=0 means an interrupted mid-iteration interior point is never
            # finalized into a loadable solution -- unlike a MILP timeout, which usually
            # has a storable incumbent, an LP barrier timeout can have nothing Pyomo can
            # load at all. The iteration progress is real and visible in the tee'd solver
            # log above; it just never reaches Results/.
            termination = str(results.solver.termination_condition)
            if termination not in ('optimal', 'feasible', 'globallyOptimal', 'locallyOptimal', 'maxTimeLimit'):
                print('\n' + '=' * 78)
                print('NO LOADABLE SOLUTION ({})'.format(termination))
                print('=' * 78)
                print('  The solver did not reach a result Pyomo can load into the model.')
                print('  Common cause: TimeLimit reached before the solver\'s own convergence')
                print('  check, especially LP/Crossover=0 barrier runs, where a mid-solve')
                print('  interior-point iterate is not a finalized solution to fall back on.')
                print('  The best iterate values are visible in this run\'s solver log above')
                print('  (Primal/Dual objective, Compl residual) -- just not exported here.')
                print('  Increase SITE_TIME, or loosen MGPY_BARCONVTOL/MGPY_FEASTOL, and re-run.')
                print('=' * 78 + '\n')
                raise SystemExit(2)

        instance.solutions.load_from(results)  # Loading solution into instance

        if fix_design:
            Report_Fixed_Design(instance, relaxed_count, Model_Components)
        # In 'nobinary' mode this check is not diagnostics -- it is the only thing
        # standing between the reported number and a model that permits physically
        # impossible dispatch. Run it unconditionally, whatever else happened.
        if bess_form == 'nobinary' and Model_Components in (0, 1):
            print('\n' + '=' * 78)
            print('MGPY_BESS_FORM=nobinary -- MANDATORY VALIDITY CHECK')
            print('=' * 78)
            Check_Simultaneous_Flows(instance, 'nobinary')
            print('=' * 78 + '\n')
        Check_Battery_Bound(instance, max_battery_kwh, MILP_Formulation)

        return instance
        
    else:   # Multiobjective_Optimization == 1: run the epsilon-constraint method to trace a cost-vs-CO2 Pareto front
        if Optimization_Goal == 1:
            model.f1 = Var()   # Auxiliary variable representing the NPC objective
            model.C_f1 = Constraint(expr = model.f1 == model.Net_Present_Cost)
            model.ObjectiveFuntion = Objective(expr = model.f1, 
                                                  sense=minimize)   # First-stage objective: minimize NPC
            model.f2 = Var()   # Auxiliary variable representing the CO2 emissions objective
            model.C_f2 = Constraint(expr = model.f2 == model.CO2_emission)
            model.ObjectiveFuntion1 = Objective(expr = model.f2, 
                                                   sense=minimize)   # Second-stage objective: minimize CO2 emissions
        
            # n = int(input("please indicate how many points (n) you want to analyse: "))
            
            #NPC min and CO2 emission max calculation
            model.ObjectiveFuntion1.deactivate()    # Only the NPC objective is active for this solve
            instance = model.create_instance(datapath)
            if Solver == 0:
                opt = SolverFactory('gurobi')
                if MILP_Formulation:
                    opt.set_options('Method=3 BarHomogeneous=1 Crossover=1 MIPfocus=1 BarConvTol=1e-3 OptimalityTol=1e-3 FeasibilityTol=1e-4 TimeLimit=10000 MIPGap=0.01 NodefileStart=0.5')
                else:
                    opt.set_options('Method=2 BarHomogeneous=0 Crossover=0 BarConvTol=1e-4 OptimalityTol=1e-4 FeasibilityTol=1e-4 IterationLimit=1000')
                print('Optimizing only for minimum NPC...')
            elif Solver == 1:
                opt = SolverFactory('glpk')
                timelimit = 10000
                opt.options['tmlim'] = timelimit
                if MILP_Formulation: 
                    opt.options['mipgap'] = 0.01      # Set relative gap tolerance for MIP
                    opt.options['clq_cuts'] = 'on'    # Enable clique cuts
                
                print('Optimizing only for minimum NPC...')
            
            opt.solve(instance, tee=True)
            print('Instance solved') 
            NPC_min = value(instance.ObjectiveFuntion)          # Best-case (minimum) NPC, ignoring emissions
            CO2emission_max = value(instance.ObjectiveFuntion1) # CO2 emissions resulting from the cost-optimal solution (Pareto front upper bound)
            print('NPC_min [kUSD] = ' +str(NPC_min/1e3),'CO2emission_max [ton] = ' +str(CO2emission_max/1e3))
           
        
            #NPC max and CO2 emission min calculation
            model.ObjectiveFuntion.deactivate()    # Switch to optimizing emissions only
            model.ObjectiveFuntion1.activate()
            instance = model.create_instance(datapath)
            print('Optimizing only for minimum CO2 emissions...')
            results = opt.solve(instance, tee=True)  # Solve the model instance
            instance.solutions.load_from(results)    # Load the solution into the instance
            print('Instance solved') 

            NPC = value(instance.ObjectiveFuntion)
            CO2emission_min = value(instance.f2)   # Best-case (minimum) achievable CO2 emissions
            print('NPC [kUSD] = ' +str(NPC/1e3),'CO2emission_min [ton] = ' +str(CO2emission_min/1e3))
           

            # Second Optimization: Minimize cost while constraining emissions to the minimum value found
            model.ObjectiveFuntion.activate()     # Reactivate cost minimization objective
            model.ObjectiveFuntion1.deactivate()  # Ensure emissions objective is deactivated
            instance = model.create_instance(datapath)
            instance.CO2 = Param(initialize=CO2emission_min, mutable=True)         # Fix the emissions target at its minimum value
            instance.CO2_fixed = Constraint(expr = instance.f2 == instance.CO2)    # Force emissions to equal that target

            print('Optimizing for cost with minimum CO2 emissions constraint...')
            results = opt.solve(instance, tee=True)     # Solve the model instance again with the new constraint
            instance.solutions.load_from(results)       # Load the new solution into the instance
            NPC_max = value(instance.ObjectiveFuntion)  # Worst-case (maximum) NPC needed to reach zero-emission-min target (Pareto front upper cost bound)
            print('NPC_max [kUSD] = ' +str(NPC_max/1e3),'with CO2emission_fixed [ton] = ' +str(CO2emission_min/1e3))

            #normal eps method
            model.ObjectiveFuntion.activate()
            model.ObjectiveFuntion1.deactivate()        

            instance = model.create_instance(datapath)
            instance.e = Param(initialize=0, mutable=True)           # Mutable epsilon parameter swept across the CO2 range
            instance.C_e = Constraint(expr = instance.f2 == instance.e)   # Force emissions to equal the current epsilon value
            
            if Plot_Max_Cost:   # Include the max-cost/min-emission point in the sampled steps
                step = int((CO2emission_max - CO2emission_min)/(n-1))
                steps = list(range(int(CO2emission_min),int(CO2emission_max),step)) 
            else:
                step = int((CO2emission_max - CO2emission_min)/n)
                steps = list(range(int(CO2emission_min),int(CO2emission_max),step)) 
                steps.pop(0)          
                
            f1_l,f2_l = [],[]   # Collected (NPC, CO2) pairs along the Pareto front
            for i in steps:  
                instance.e = i   # Set the emissions cap for this Pareto point
                #print(value(instance.e))
                print('Calling solver...')
                results = opt.solve(instance, tee=True) # Solving a model instance
                print('Instance solved')  # Loading solution into instance
                f1_l.append(value(instance.f1)/1e3)
                f2_l.append(value(instance.f2)/1e3)
                instance.solutions.load_from(results)
                print('NPC [kUSD] = ' +str(value(instance.f1)/1e3),'CO2 emission [ton] = ' +str(value(instance.f2)/1e3))

            if len(f1_l)<n:   # Ensure the min-cost/max-emission endpoint is included in the plotted series
                 f1_l.append(NPC_min/1e3)
                 f2_l.append(CO2emission_max/1e3)        
            
            print ('\nNPC [kUSD] =' +str(f1_l))
            print ('\nCO2 emission [ton] =' +str(f2_l))
            CO2=(CO2emission_max-CO2emission_min)/1e3   # Total CO2 abated between the two extreme solutions
            NPC=f1_l[0]*1000-NPC_min                    # Extra cost paid to abate that CO2
            print('Cost CO2 avoided [USD/ton] =' +str(round(NPC/CO2,3)))    # Marginal abatement cost
            
            ##################################################################################
            def save_pareto_curve(f1_l, f2_l, plot_title, plot_xlabel, plot_ylabel, plot_path):
                fig, ax = plt.subplots(figsize=(15, 10))
                ax.plot(f1_l, f2_l, 'o-', c='r', label='Pareto optimal front')
                ax.set_title(plot_title, fontsize=22)
                ax.set_xlabel(plot_xlabel, fontsize=20)
                ax.set_ylabel(plot_ylabel, fontsize=20)
                ax.grid(True)
                ax.legend(loc='best', fontsize=20)
                plt.tight_layout()

                # Save plot to file
                plt.savefig(plot_path, dpi=400, bbox_inches='tight')
                plt.close()
            
                
            print('Plotting Pareto curve...')

            current_directory = os.path.dirname(os.path.abspath(__file__))
            results_directory = os.path.join(current_directory, '..', 'Results', RUN_ID, 'Plots')
            os.makedirs(results_directory, exist_ok=True)
            plot_path = os.path.join(results_directory, 'ParetoCurve.png')
            save_pareto_curve(f1_l, f2_l, "Pareto Curve - NPC", "CO2 Emissions [ton]", "Net Present Costs [kUSD]",plot_path)

            print('Pareto curve plot saved.')
            #################################################################################################
            
            # Calculate the step size
            step = int((CO2emission_max - CO2emission_min) / (n - 1))

            # Generate steps from CO2emission_min up to (but not including) CO2emission_max, with an interval of 'step'
            steps = list(range(int(CO2emission_min), int(CO2emission_max), step))

            # Insert 0 at the beginning of the list to represent the baseline or control scenario
            steps.insert(0, 0)             
                
            instance.e = steps[p]     # Select the p-th Pareto point requested by the user/config
            print('Calling solver...')
            results = opt.solve(instance, tee=True) # Solving a model instance
            print('Instance solved')  # Loading solution into instance
            instance.solutions.load_from(results)
            print('NPC [kUSD] = ' +str(value(instance.f1)/1e3),'CO2 emission [ton] = ' +str(value(instance.f2)/1e3))
            return instance
        
        elif Optimization_Goal == 0:
            model.f1 = Var()   # Auxiliary variable representing the total variable/operation cost objective
            model.C_f1 = Constraint(expr = model.f1 == model.Total_Variable_Cost)
            model.ObjectiveFuntion = Objective(expr = model.f1, 
                                                  sense=minimize)   # First-stage objective: minimize operation cost
            model.f2 = Var()   # Auxiliary variable representing the CO2 emissions objective
            model.C_f2 = Constraint(expr = model.f2 == model.CO2_emission)
            model.ObjectiveFuntion1 = Objective(expr = model.f2, 
                                                   sense=minimize)   # Second-stage objective: minimize CO2 emissions
        
            # n = int(input("please indicate how many points (n) you want to analyse: "))
                        
            #NPC min and CO2 emission max calculation
            model.ObjectiveFuntion1.deactivate()    # Only the operation-cost objective is active for this solve
            instance = model.create_instance(datapath)
            if Solver == 0:
                opt = SolverFactory('gurobi')
                if MILP_Formulation:
                    opt.set_options('Method=3 BarHomogeneous=1 Crossover=1 MIPfocus=1 BarConvTol=1e-3 OptimalityTol=1e-3 FeasibilityTol=1e-4 TimeLimit=10000 MIPGap=0.01 NodefileStart=0.5')
                else:
                    opt.set_options('Method=2 BarHomogeneous=0 Crossover=0 BarConvTol=1e-4 OptimalityTol=1e-4 FeasibilityTol=1e-4 IterationLimit=1000')
                print('Optimizing only for minimum Operation Costs...')
            elif Solver == 1:
                opt = SolverFactory('glpk')
                timelimit = 10000
                opt.options['tmlim'] = timelimit
                if MILP_Formulation: 
                    opt.options['mipgap'] = 0.01      # Set relative gap tolerance for MIP
                    opt.options['clq_cuts'] = 'on'    # Enable clique cuts
                
                print('Optimizing only for minimum Operation Costs...')

            OperationCost_min = value(instance.ObjectiveFuntion)          # Best-case (minimum) operation cost, ignoring emissions
            CO2emission_max = value(instance.ObjectiveFuntion1)           # CO2 emissions resulting from the cost-optimal solution (Pareto front upper bound)
            print('OperationCost_min [kUSD] = ' +str(OperationCost_min/1e3),'CO2emission_max [ton] = ' +str(CO2emission_max/1e3))
           
        
            #NPC max and CO2 emission min calculation
            model.ObjectiveFuntion.deactivate()    # Switch to optimizing emissions only
            model.ObjectiveFuntion1.activate()
            instance = model.create_instance(datapath)
            print('Calling solver...')
            print('Optimizing only for minimum CO2 emissions...')
            opt.solve(instance, tee=True)
            print('Instance solved') 
            OperationCost = value(instance.ObjectiveFuntion)
            CO2emission_min = value(instance.ObjectiveFuntion1)    # Best-case (minimum) achievable CO2 emissions
            print('OperationCost [kUSD] = ' +str(OperationCost/1e3),'CO2emission_min [ton] = ' +str(CO2emission_min/1e3))
            r=CO2emission_max-CO2emission_min   # Total CO2 range spanned by the Pareto front (unused downstream, kept for reference)

            # Second Optimization: Minimize cost while constraining emissions to the minimum value found
            model.ObjectiveFuntion.activate()     # Reactivate cost minimization objective
            model.ObjectiveFuntion1.deactivate()  # Ensure emissions objective is deactivated
            instance = model.create_instance(datapath)
            instance.CO2 = Param(initialize=CO2emission_min, mutable=True)         # Fix the emissions target at its minimum value
            instance.CO2_fixed = Constraint(expr = instance.f2 == instance.CO2)    # Force emissions to equal that target

            print('Optimizing for cost with minimum CO2 emissions constraint...')
            results = opt.solve(instance, tee=True)     # Solve the model instance again with the new constraint
            instance.solutions.load_from(results)       # Load the new solution into the instance
            OperationCost_max = value(instance.ObjectiveFuntion)   # Worst-case (maximum) operation cost needed to reach the min-emission target
            print('OperationCost_max [kUSD] = ' +str(OperationCost_max/1e3),'with CO2emission_fixed [ton] = ' +str(CO2emission_min/1e3))

            
            #normal eps method
            model.ObjectiveFuntion.activate()
            model.ObjectiveFuntion1.deactivate()

            instance = model.create_instance(datapath)
            instance.e = Param(initialize=0, mutable=True)                    # Mutable epsilon parameter swept across the CO2 range
            instance.C_e = Constraint(expr = instance.f2 == instance.e)       # Force emissions to equal the current epsilon value

            if Plot_Max_Cost:   # Include the max-cost/min-emission point in the sampled steps
                step = int((CO2emission_max - CO2emission_min)/(n-1))
                steps = list(range(int(CO2emission_min),int(CO2emission_max),step)) 
            else:
                step = int((CO2emission_max - CO2emission_min)/n)
                steps = list(range(int(CO2emission_min),int(CO2emission_max),step)) 
                steps.pop(0)     

            f1_l,f2_l = [],[]    # Collected (operation cost, CO2) pairs along the Pareto front
            for i in steps:  
               instance.e = i    # Set the emissions cap for this Pareto point
               #print(value(instance.e))
               print('Calling solver...')
               results = opt.solve(instance, tee=True) # Solving a model instance
               print('Instance solved')  # Loading solution into instance
               f1_l.append(value(instance.f1)/1e3)
               f2_l.append(value(instance.f2)/1e3)
               instance.solutions.load_from(results)
               print('Operation Cost [kUSD] = ' +str(value(instance.f1)/1e3),'CO2 emission [ton] = ' +str(value(instance.f2)/1e3))

            if len(f1_l)<n:   # Ensure the min-cost/max-emission endpoint is included in the plotted series
                f1_l.append(OperationCost_min/1e3)
                f2_l.append(CO2emission_max/1e3) 

            print ('\nOperation Cost [kUSD] =' +str(f1_l))
            print ('\nCO2 emission [ton] =' +str(f2_l))  

            #################################################################################################
            
            print('Plotting Pareto curve...')

            current_directory = os.path.dirname(os.path.abspath(__file__))
            results_directory = os.path.join(current_directory, '..', 'Results', RUN_ID, 'Plots')
            os.makedirs(results_directory, exist_ok=True)
            plot_path = os.path.join(results_directory, 'ParetoCurve.png')
            save_pareto_curve(f1_l, f2_l, "Pareto Curve - Operation Costs", "CO2 Emissions [ton]", "Operation Costs [kUSD]", plot_path)

            print('Pareto curve plot saved.')

            #################################################################################################
            
            steps = list(range(int(CO2emission_min),int(CO2emission_max),step))
            
            if len(steps)<=n:
                steps.append(CO2emission_max)

            if Plot_Max_Cost:
                steps.insert(0,0) 

            # i = int(input("please indicate which solution you prefer (starting from 1 to n in CO2 emission): ")) #asks the user how many profiles (i.e. code runs) he wants
                           
            instance.e = steps[p]    # Select the p-th Pareto point requested by the user/config
            #print(value(instance.e))
            print('Calling solver...')
            results = opt.solve(instance, tee=True) # Solving a model instance
            print('Instance solved')  # Loading solution into instance
            instance.solutions.load_from(results)
            print('Operation Cost [kUSD] = ' +str(value(instance.f1)/1e3),'CO2 emission [ton] = ' +str(value(instance.f2)/1e3))
            return instance
        


           
          
        
    
                
                    
        
                
           
