import pandas as pd                                                       # tabular data structures used throughout for building result tables
import matplotlib.pyplot as plt                                           # plotting (used elsewhere in the module for charts)
import matplotlib.patches as mpatches                                     # legend patch helpers for plots
import matplotlib.lines as mlines                                         # legend line helpers for plots
from pandas import ExcelWriter                                            # writer object used to export multiple sheets to a single .xlsx file
import re                                                                 # regular expressions (utility import)
import os                                                                 # filesystem path handling for output directories
import warnings; warnings.simplefilter(action='ignore', category=FutureWarning)  # silence pandas FutureWarning noise in console output
from run_id import RUN_ID                                                 # unique identifier for the current run, used to namespace output folders

#%% Results summary
def ResultsSummary(instance, Optimization_Goal, TimeSeries):
    # Orchestrates all the individual result-extraction functions, writes them to an Excel
    # workbook and returns everything bundled into a single dictionary for later use (e.g. PrintResults).

    from Results import EnergySystemCost, EnergySystemSize, YearlyCosts, YearlyEnergyParams, YearlyEnergyParamsSC, EnergySystemLandUse  # re-import locally (shadows module-level names below)
    
    print('Results: exporting economic results...')
    EnergySystemCost                              = EnergySystemCost(instance, Optimization_Goal)   # build cost breakdown table
    YearlyCost                                    = YearlyCosts(instance)                            # build year-by-year cash flow table
    print('         exporting technical results...')
    EnergySystemSize                              = EnergySystemSize(instance)                       # build installed-capacity table
    YearlyEnergyParams, RenewablePenetration      = YearlyEnergyParams(instance, TimeSeries)          # yearly aggregated energy shares (weighted across scenarios)
    YearlyEnergyParamsSC, RenewablePenetrationSC  = YearlyEnergyParamsSC(instance, TimeSeries)        # same energy shares but kept per-scenario
    if instance.Land_Use.value == 1: EnergySystemLandUse = EnergySystemLandUse(instance)              # land-use table, only if land-use accounting is enabled
    
    current_directory = os.path.dirname(os.path.abspath(__file__))                                   # folder containing this script
    run_results_directory = os.path.join(current_directory, '..', 'Results', RUN_ID)                 # per-run results folder (Code/Results/<RUN_ID>)
    os.makedirs(run_results_directory, exist_ok=True)                                                # create it if it doesn't exist yet
    results_directory = os.path.join(run_results_directory, 'Results_Summary.xlsx')                  # path to the summary workbook

    Excel = ExcelWriter(results_directory)                                # open the workbook for writing multiple sheets

    EnergySystemSize.to_excel(Excel, sheet_name='Size')                                              # write installed size table
    if instance.Land_Use.value == 1: EnergySystemLandUse.to_excel(Excel, sheet_name='Land Use')       # write land-use table if applicable
    EnergySystemCost.to_excel(Excel, sheet_name='Cost')                                               # write cost breakdown table
    YearlyCost.to_excel(Excel, sheet_name='Yearly cash flows')                                        # write yearly cash-flow table
    YearlyEnergyParams.to_excel(Excel, sheet_name='Yearly energy parameters')                         # write yearly aggregated energy-share table
    YearlyEnergyParamsSC.to_excel(Excel, sheet_name='Yearly energy parameters SC')                    # write per-scenario yearly energy-share table

    Excel.close()                                                        # flush and close the workbook file

    Results = {
        'Costs': EnergySystemCost,                                                                    # cost breakdown DataFrame
        'Size': EnergySystemSize,                                                                     # installed-capacity DataFrame
        'Land Use': EnergySystemLandUse if instance.Land_Use.value == 1 else None,                    # land-use DataFrame or None
        'Yearly cash flows': YearlyCost,                                                              # yearly cash-flow DataFrame
        'Yearly energy parameters': YearlyEnergyParams,                                               # yearly aggregated energy-share DataFrame
        'Renewables Penetration': RenewablePenetration,                                                # yearly weighted renewable penetration series
        'Yearly energy parameters SC': YearlyEnergyParamsSC,                                          # per-scenario yearly energy-share DataFrame
        'Renewables Penetration SC': RenewablePenetrationSC,                                          # per-scenario renewable penetration series
    }
    
    return Results  # dictionary consumed by PrintResults() and by callers that export/plot results

#%% TimeSeries generation
def TimeSeries(instance):
    # Builds an hourly time-series DataFrame per scenario/year from the solved Pyomo instance
    # and exports one Excel workbook per scenario (one sheet per year).

    print('\nResults: exporting time-series...')
    "Importing parameters"
    S  = int(instance.Scenarios.extract_values()[None])       # number of scenarios
    P  = int(instance.Periods.extract_values()[None])         # number of time periods (hours) per year
    Y  = int(instance.Years.extract_values()[None])            # project lifetime in years
    ST = int(instance.Steps_Number.extract_values()[None])     # number of investment steps
    R  = int(instance.RES_Sources.extract_values()[None])      # number of renewable source types
    G  = int(instance.Generator_Types.extract_values()[None])  # number of generator types

    Scenario_Weight = instance.Scenario_Weight.extract_values()  # probability/weight of each scenario
    Discount_Rate   = instance.Discount_Rate.value                # discount rate used for NPV calcs
    RES_Names       = instance.RES_Names.extract_values()         # renewable source display names
    Generator_Names = instance.Generator_Names.extract_values()   # generator display names
    Fuel_Names      = instance.Fuel_Names.extract_values()        # fuel display names

    StartDate       = pd.to_datetime(instance.StartDate())    # simulation start timestamp
    start_year      = StartDate.year
    start_month     = StartDate.month
    start_day       = StartDate.day
    start_hour      = StartDate.hour
    start_minute    = StartDate.minute
    start_second    = StartDate.second

    "Generating years-steps tuples list"
    steps = [i for i in range(1, ST+1)]                        # step indices [1..ST]
    
    years_steps_list = [1 for i in range(1, ST+1)]              # first year of each investment step
    s_dur = instance.Step_Duration.value                        # duration (years) of each step
    for i in range(1, ST): 
        years_steps_list[i] = years_steps_list[i-1] + s_dur     # cumulative start year of step i
    ys_tuples_list = [[] for i in range(1, Y+1)]                # will hold (year, step) pairs for every year
    for y in range(1, Y+1):  
        if len(years_steps_list) == 1:
            ys_tuples_list[y-1] = (y,1)                         # single-step case: every year maps to step 1
        else:
            for i in range(len(years_steps_list)-1):
                if y >= years_steps_list[i] and y < years_steps_list[i+1]:
                    ys_tuples_list[y-1] = (y, steps[i])         # year falls within step i
                elif y >= years_steps_list[-1]:
                    ys_tuples_list[y-1] = (y, len(steps))       # year falls in the last step
                    
    "Importing energy flows timeseries"  
    RES_Energy_Production       = instance.RES_Energy_Production.get_values()  # RES production per (scenario, year, source, hour)
    BESS_Outflow                = instance.Battery_Outflow.get_values()
    BESS_Inflow                 = instance.Battery_Inflow.get_values()               # battery charging energy per (s,y,t)
    if instance.MILP_Formulation.value == 1 and instance.Generator_Partial_Load.value == 1:
       Generator_Energy_Total      = instance.Generator_Energy_Total.get_values()    # total generator output (full+partial) [MILP + partial load]
       Generator_Energy_Partial    = instance.Generator_Energy_Partial.get_values()  # output while running in partial-load mode
       Generator_Partial           = instance.Generator_Partial.get_values()         # binary/units count running in partial load
       Generator_Full              = instance.Generator_Full.get_values()            # binary/units count running at full load
    elif instance.MILP_Formulation.value == 1 and instance.Generator_Partial_Load.value == 0:
       Generator_Energy_Total = instance.Generator_Energy_Total.get_values()         # total generator output [MILP, no partial load]
    else :
       Generator_Energy_Production = instance.Generator_Energy_Production.get_values()  # generator output [LP formulation]
    Curtailment                 = instance.Energy_Curtailment.get_values()           # curtailed renewable energy per (s,y,t)
    Lost_Load                   = instance.Lost_Load.get_values()                    # unserved demand per (s,y,t)
    Electric_Demand             = instance.Energy_Demand.extract_values()             # input electricity demand per (s,y,t)
    Electricity_From_Grid       = instance.Energy_From_Grid.get_values()              # energy imported from the national grid
    Electricity_To_Grid         = instance.Energy_To_Grid.get_values()                # energy exported/sold to the national grid   
    
    BESS_SOC                    = instance.Battery_SOC.get_values()                  # battery state of charge per (s,y,t)
    LHV                         = instance.Fuel_LHV.extract_values()                 # fuel lower heating value, used to convert energy to fuel volume
    Generator_Efficiency        = instance.Generator_Efficiency.extract_values()      # generator conversion efficiency
    FUEL_emission               = instance.FUEL_emission.get_values()                # CO2 emissions from fuel combustion per (s,y,g,t)
    
    "Creating TimeSeries dictionary and exporting excel"
    TimeSeries = {}                                                                  # nested dict: TimeSeries[scenario][year] -> DataFrame
        
    for s in range(1,S+1):
        TimeSeries[s] = {}
        current_directory = os.path.dirname(os.path.abspath(__file__))
        results_directory = os.path.join(current_directory, '..', 'Results', RUN_ID)
        os.makedirs(results_directory, exist_ok=True)
        results_2_path = os.path.join(results_directory, 'Time_Series_SC_%d.xlsx' % (s))  # one workbook per scenario
        with pd.ExcelWriter(results_2_path) as writer:
         for y in range(1,Y+1):
            
            scenario_header  = []                                                    # top-level MultiIndex column labels being built up below
            flow_header      = []
            component_header = []
            unit_header      = []
            
            TimeSeries[s][y] = pd.DataFrame()                                        # start with an empty frame for this (scenario, year)
            DEM = pd.DataFrame([Electric_Demand[(s,y,t)] for t in range(1,P+1)])      # hourly demand column
            TimeSeries[s][y] = pd.concat([TimeSeries[s][y], DEM], axis=1)             # append demand as first column
            scenario_header  += ['Scenario ' + str(s)]
            flow_header      += ['Electric Demand']
            component_header += ['']
            unit_header      += ['kWh']
            
            for r in range(1,R+1):
                RES = pd.DataFrame([RES_Energy_Production[(s,y,r,t)] for t in range(1,P+1)])  # hourly production of RES source r
                TimeSeries[s][y] = pd.concat([TimeSeries[s][y], RES], axis=1)
                scenario_header  += ['Scenario ' + str(s)]
                flow_header      += ['RES Production']
                component_header += [RES_Names[r]]
                unit_header      += ['kWh']
            
            # Total Energy Production of the generator
            if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:  # only if generators are part of the system
                for g in range(1,G+1):
                    if instance.MILP_Formulation.value:
                       GEN = pd.DataFrame([Generator_Energy_Total[(s,y,g,t)] for t in range(1,P+1)])       # MILP: use total generator output
                    else:
                       GEN = pd.DataFrame([Generator_Energy_Production[(s,y,g,t)] for t in range(1,P+1)])  # LP: use direct production variable
                    TimeSeries[s][y] = pd.concat([TimeSeries[s][y], GEN], axis=1)
                    scenario_header  += ['Scenario ' + str(s)]
                    flow_header      += ['Generator Production']
                    component_header += [Generator_Names[g]]
                    unit_header      += ['kWh']
                    
                " Partial Load Effect"
                # Generator Partial Load Production
                if instance.MILP_Formulation.value == 1 and instance.Generator_Partial_Load.value == 1:
                   for g in range(1,G+1):
                       GEN_p = pd.DataFrame([Generator_Energy_Partial[(s,y,g,t)] for t in range(1,P+1)])   # energy produced while in partial-load mode
                       TimeSeries[s][y] = pd.concat([TimeSeries[s][y], GEN_p], axis=1)
                       scenario_header  += ['Scenario ' + str(s)]
                       flow_header      += ['Generator Partial Load Production']
                       component_header += [Generator_Names[g]]
                       unit_header      += ['kWh']
                
                # Units of Generators in Partial Load (1 o 0)       
                if instance.MILP_Formulation.value == 1 and instance.Generator_Partial_Load.value == 1:
                   for g in range(1,G+1):
                       GEN_pu = pd.DataFrame([Generator_Partial[(s,y,g,t)] for t in range(1,P+1)])         # count/flag of units running in partial load
                       TimeSeries[s][y] = pd.concat([TimeSeries[s][y], GEN_pu], axis=1)
                       scenario_header  += ['Scenario ' + str(s)]
                       flow_header      += ['Units of Generators in Partial Load']
                       component_header += [Generator_Names[g]]
                       unit_header      += ['kWh']
                
                # Units of Generators in Full Load
                if instance.MILP_Formulation.value == 1 and instance.Generator_Partial_Load.value == 1:
                   for g in range(1,G+1):
                       GEN_fu = pd.DataFrame([Generator_Full[(s,y,g,t)] for t in range(1,P+1)])            # count/flag of units running at full load
                       TimeSeries[s][y] = pd.concat([TimeSeries[s][y], GEN_fu], axis=1)
                       scenario_header  += ['Scenario ' + str(s)]
                       flow_header      += ['Units of Generators in Full Load']
                       component_header += [Generator_Names[g]]
                       unit_header      += ['kWh']
                
            BESS_OUT         = pd.DataFrame([BESS_Outflow[(s,y,t)] for t in range(1,P+1)])            # battery discharge column
            BESS_IN          = pd.DataFrame([BESS_Inflow[(s,y,t)] for t in range(1,P+1)])              # battery charge column
            LL               = pd.DataFrame([Lost_Load[(s,y,t)] for t in range(1,P+1)])                # unserved demand column
            CURTAIL          = pd.DataFrame([Curtailment[(s,y,t)] for t in range(1,P+1)])              # curtailed energy column
            EL_FROM_GRID     = pd.DataFrame([Electricity_From_Grid[(s,y,t)] for t in range(1,P+1)])    # grid import column
            EL_TO_GRID       = pd.DataFrame([Electricity_To_Grid[(s,y,t)] for t in range(1,P+1)])      # grid export column
            if instance.Model_Components.value == 0 or instance.Model_Components.value == 1:  # battery is part of the system
                if instance.Grid_Connection.value == 1:
                    TimeSeries[s][y] = pd.concat([TimeSeries[s][y], BESS_OUT, BESS_IN, LL, CURTAIL, EL_FROM_GRID,EL_TO_GRID], axis=1) 
                    scenario_header  += ['Scenario ' + str(s),'Scenario ' + str(s),'Scenario ' + str(s),'Scenario ' + str(s),'Scenario ' + str(s),'Scenario ' + str(s)]
                    flow_header      += ['Battery Discharge','Battery Charge','Lost Load','Curtailment','Electricity from grid','Electricity to grid'] 
                    component_header += ['','','','','','']
                    unit_header      += ['kWh','kWh','kWh','kWh','kWh','kWh']
                if instance.Grid_Connection.value == 0:
                    TimeSeries[s][y] = pd.concat([TimeSeries[s][y], BESS_OUT, BESS_IN, LL, CURTAIL], axis=1) 
                    scenario_header  += ['Scenario ' + str(s),'Scenario ' + str(s),'Scenario ' + str(s),'Scenario ' + str(s)]
                    flow_header      += ['Battery Discharge','Battery Charge','Lost Load','Curtailment'] 
                    component_header += ['','','','']
                    unit_header      += ['kWh','kWh','kWh','kWh']
                
                
                SOC              = pd.DataFrame([BESS_SOC[(s,y,t)] for t in range(1,P+1)])             # battery state-of-charge column
                TimeSeries[s][y] = pd.concat([TimeSeries[s][y], SOC], axis=1)
                scenario_header  += ['Scenario ' + str(s)]
                flow_header      += ['Battery SOC']
                component_header += ['']
                unit_header      += ['kWh']

            if instance.Model_Components.value == 2:  # no battery in the system: only lost load/curtailment/grid flows
                if instance.Grid_Connection.value == 1:
                    TimeSeries[s][y] = pd.concat([TimeSeries[s][y], LL, CURTAIL, EL_FROM_GRID, EL_TO_GRID], axis=1) 
                    scenario_header  += ['Scenario ' + str(s),'Scenario ' + str(s),'Scenario ' + str(s),'Scenario ' + str(s)]
                    flow_header      += ['Lost Load','Curtailment','Electricity from grid','Electricity to grid'] 
                    component_header += ['','','','']
                    unit_header      += ['kWh','kWh','kWh','kWh']
                if instance.Grid_Connection.value == 0:
                    TimeSeries[s][y] = pd.concat([TimeSeries[s][y], LL, CURTAIL], axis=1) 
                    scenario_header  += ['Scenario ' + str(s),'Scenario ' + str(s)]
                    flow_header      += ['Lost Load','Curtailment'] 
                    component_header += ['','']
                    unit_header      += ['kWh','kWh']
                
            if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:  # generators present: add fuel consumption & emissions
                for g in range(1,G+1):
                    if instance.MILP_Formulation.value:
                       FUEL = pd.DataFrame([Generator_Energy_Total[(s,y,g,t)]/LHV[g]/Generator_Efficiency[g] for t in range(1,P+1)])       # fuel volume = energy / (LHV * efficiency)
                    else:
                       FUEL = pd.DataFrame([Generator_Energy_Production[(s,y,g,t)]/LHV[g]/Generator_Efficiency[g] for t in range(1,P+1)]) 
                    TimeSeries[s][y] = pd.concat([TimeSeries[s][y], FUEL], axis=1)
                    scenario_header  += ['Scenario ' + str(s)]
                    flow_header      += ['Fuel Consumption']
                    component_header += [Fuel_Names[g]]
                    unit_header      += ['Lt']
                    
                for g in range(1,G+1):
                    CO2              = pd.DataFrame([FUEL_emission[(s,y,g,t)] for t in range(1,P+1)])   # hourly CO2 emission from fuel combustion
                    TimeSeries[s][y] = pd.concat([TimeSeries[s][y], CO2], axis=1)
                    scenario_header  += ['Scenario ' + str(s)]
                    flow_header      += ['CO2 emission']
                    component_header += [Fuel_Names[g]]
                    unit_header      += ['kg']            
                
            TimeSeries[s][y].columns = pd.MultiIndex.from_arrays([scenario_header, flow_header, component_header, unit_header], names=['','Flow','Component','Unit'])  # assemble 4-level column MultiIndex
            date                     = str(start_year+y-1)+'/'+str(start_month)+'/'+str(start_day)+' '+str(start_hour)+':'+str(start_minute)  # calendar date for the first hour of this simulation year
            TimeSeries[s][y].index   = pd.date_range(start=date, periods=P, freq='h')                   # hourly DatetimeIndex for the row axis
            
            round(TimeSeries[s][y],1).to_excel(writer, sheet_name='Year ' + str(y))                     # write this year's sheet, rounded to 1 decimal
            
    return TimeSeries

    
#%% Economic output        
def EnergySystemCost(instance, Optimization_Goal):
    # Builds the full economic cost/emission breakdown table (investment, fixed O&M,
    # variable O&M, replacement, fuel, grid, salvage, NPC, LCOE and CO2 emissions).
    
    #%% Importing parameters
    S  = int(instance.Scenarios.extract_values()[None])
    P  = int(instance.Periods.extract_values()[None])
    Y  = int(instance.Years.extract_values()[None])
    ST = int(instance.Steps_Number.extract_values()[None])
    R  = int(instance.RES_Sources.extract_values()[None])
    G  = int(instance.Generator_Types.extract_values()[None])

    RES_Names = instance.RES_Names.extract_values()
    Generator_Names = instance.Generator_Names.extract_values()
    Fuel_Names = instance.Fuel_Names.extract_values()
    Discount_Rate = instance.Discount_Rate.value

    upgrade_years_list = [1 for i in range(ST)]                    # first year of each investment step
    s_dur = instance.Step_Duration.value
    for i in range(1, ST): 
        upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    
    yu_tuples_list = [0 for i in range(1,Y+1)]                      # (year, step) for every project year
    if ST == 1:    
        for y in range(1,Y+1):            
            yu_tuples_list[y-1] = (y, 1)    
    else:        
        for y in range(1,Y+1):            
            for i in range(len(upgrade_years_list)-1):
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:
                    yu_tuples_list[y-1] = (y, [st for st in range(ST)][i+1])                
                elif y >= upgrade_years_list[-1]:
                    yu_tuples_list[y-1] = (y, ST)   
    tup_list = [[] for i in range(ST-1)]                            # (year, step) only for the first year of each step after the first
    for i in range(0,ST-1):
        tup_list[i] = yu_tuples_list[s_dur*i + s_dur]      


    #%% Investment cost
    "Total"
    Grid_Investment = (instance.Grid_Connection_Cost.value * instance.Grid_Connection.value * instance.Grid_Distance.value)/(1 + instance.Discount_Rate.value)**(instance.Year_Grid_Connection.value-1)  # discounted cost of connecting to the grid
    Total_Investment_Cost = pd.DataFrame(['Total Investment cost', 'System', '-', 'kUSD', (instance.Investment_Cost.value + Grid_Investment)/1e3]).T.set_index([0,1,2,3])  # objective-function investment cost plus grid connection
    Total_Investment_Cost.columns = ['Total']
    Total_Investment_Cost.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']

    if instance.MILP_Formulation.value:  # unit-based (integer) capacity expansion
     "Renewable sources"
     RES_Units_milp = instance.RES_Units_milp.get_values()                        # number of RES units installed, cumulative per step
     RES_Nominal_Capacity = instance.RES_Nominal_Capacity.extract_values()        # capacity per unit
     RES_capacity = instance.RES_capacity.extract_values()                       # pre-existing (already installed) capacity
     if instance.Greenfield_Investment.value == 1:  # Bug fix 2026-09-04, found via linopy Stage 7 cross-validation: Constraints_Greenfield_Milp's own Investment_Cost/Salvage_Value/emission rules never reference RES_capacity/Generator_capacity/Battery_capacity at all (i.e. they behave as if these were 0), regardless of what the raw .dat says -- this reporting function must match that, not just trust the .dat to have been hand-zeroed (the one prior Pyomo-vs-Pyomo greenfield validation happened to use a .dat where it was, masking this).
        RES_capacity = {r: 0.0 for r in RES_capacity}
     RES_Inv_Specific_Cost = instance.RES_Specific_Investment_Cost.extract_values()  # cost per unit of capacity
     RES_Investment_Cost = pd.DataFrame()
     for r in range(1,R+1):
        r_inv = ((RES_Units_milp[1,r]*RES_Nominal_Capacity[r])-RES_capacity[r])*RES_Inv_Specific_Cost[r]  # step-1 investment: only the newly added capacity beyond what already exists
        res_inv = pd.DataFrame(['Investment cost', RES_Names[r], '-', 'kUSD', r_inv/1e3]).T.set_index([0,1,2,3]) 
        if ST == 1:
            res_inv.columns = ['Total']
        else:
            res_inv.columns = ['Step 1']
        res_inv.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
        RES_Investment_Cost = pd.concat([RES_Investment_Cost, res_inv], axis=1).fillna(0)
        for (y,st) in tup_list:
            r_inv = (RES_Units_milp[st,r]-RES_Units_milp[st-1,r])*RES_Nominal_Capacity[r]*RES_Inv_Specific_Cost[r]/((1+Discount_Rate)**(y-1))  # discounted cost of the capacity added in this later step
            res_inv = pd.DataFrame(['Investment cost', RES_Names[r], '-', 'kUSD', r_inv/1e3]).T.set_index([0,1,2,3]) 
            if ST == 1:
                res_inv.columns = ['Total']
            else:
                res_inv.columns = ['Step '+str(st)]
            res_inv.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
            RES_Investment_Cost = pd.concat([RES_Investment_Cost, res_inv], axis=1).fillna(0)
     RES_Investment_Cost = RES_Investment_Cost.T.groupby(level=[0], sort=False).sum().T   # merge duplicate step columns across sources
     res_inv_tot = RES_Investment_Cost.sum(1).to_frame()                                  # row-wise total across all steps
     res_inv_tot.columns = ['Total']
     if ST != 1:
        RES_Investment_Cost = pd.concat([RES_Investment_Cost, res_inv_tot],axis=1)        # append the Total column when there are multiple steps
        
     "Battery Bank"
     if instance.Model_Components.value == 0 or instance.Model_Components.value == 1:  # battery included in the system
         BESS_Units = instance.Battery_Units.get_values()
         BESS_Nominal_Capacity_milp = instance.Battery_Nominal_Capacity_milp.value
         BESS_capacity = instance.Battery_capacity.value
         if instance.Greenfield_Investment.value == 1:  # see the RES_capacity comment above (Bug fix 2026-09-04)
            BESS_capacity = 0.0
         BESS_Inv_Specific_Cost = instance.Battery_Specific_Investment_Cost.value
         BESS_Investment_Cost = pd.DataFrame()
         b_inv = ((BESS_Units[1]*BESS_Nominal_Capacity_milp)-BESS_capacity)*BESS_Inv_Specific_Cost  # step-1 investment beyond pre-existing capacity
         bess_inv = pd.DataFrame(['Investment cost', 'Battery bank', '-', 'kUSD', b_inv/1e3]).T.set_index([0,1,2,3]) 
         if ST == 1:
            bess_inv.columns = ['Total']
         else:
            bess_inv.columns = ['Step 1']
         bess_inv.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
         BESS_Investment_Cost = pd.concat([BESS_Investment_Cost, bess_inv], axis=1).fillna(0)
         for (y,st) in tup_list:
            b_inv = (BESS_Units[st]-BESS_Units[st-1])*BESS_Nominal_Capacity_milp*BESS_Inv_Specific_Cost/((1+Discount_Rate)**(y-1))  # discounted incremental battery investment for later steps
            bess_inv = pd.DataFrame(['Investment cost', 'Battery bank', '-', 'kUSD', b_inv/1e3]).T.set_index([0,1,2,3]) 
            if ST == 1:
                bess_inv.columns = ['Total']
            else:
                bess_inv.columns = ['Step '+str(st)]
            bess_inv.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
            BESS_Investment_Cost = pd.concat([BESS_Investment_Cost, bess_inv], axis=1).fillna(0)
         BESS_Investment_Cost = BESS_Investment_Cost.T.groupby(level=[0], sort=False).sum().T
         bess_inv_tot = BESS_Investment_Cost.sum(1).to_frame()
         bess_inv_tot.columns = ['Total']
         if ST != 1:
            BESS_Investment_Cost = pd.concat([BESS_Investment_Cost, bess_inv_tot],axis=1)
        
     "Generator"
     if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:  # generators included in the system
         Generator_Units = instance.Generator_Units.get_values()
         Generator_Inv_Specific_Cost = instance.Generator_Specific_Investment_Cost.extract_values()
         Generator_capacity = instance.Generator_capacity.extract_values()
         if instance.Greenfield_Investment.value == 1:  # see the RES_capacity comment above (Bug fix 2026-09-04)
            Generator_capacity = {g: 0.0 for g in Generator_capacity}
         Generator_Nominal_Capacity_milp = instance.Generator_Nominal_Capacity_milp.extract_values()
         Generator_Investment_Cost = pd.DataFrame()
         for g in range(1,G+1):
            g_inv = ((Generator_Units[1,g]*Generator_Nominal_Capacity_milp[g])-Generator_capacity[g])*Generator_Inv_Specific_Cost[g]  # step-1 investment beyond pre-existing capacity
            gen_inv = pd.DataFrame(['Investment cost', Generator_Names[g], '-', 'kUSD', g_inv/1e3]).T.set_index([0,1,2,3]) 
            if ST == 1:
                gen_inv.columns = ['Total']
            else:
                gen_inv.columns = ['Step 1']
            gen_inv.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
            Generator_Investment_Cost = pd.concat([Generator_Investment_Cost, gen_inv], axis=1).fillna(0)
            for (y,st) in tup_list:
                g_inv = ((Generator_Units[st,g]-Generator_Units[st-1,g])*Generator_Nominal_Capacity_milp[g])*Generator_Inv_Specific_Cost[g]/((1+Discount_Rate)**(y-1))  # discounted incremental generator investment for later steps
                gen_inv = pd.DataFrame(['Investment cost', Generator_Names[g], '-', 'kUSD', g_inv/1e3]).T.set_index([0,1,2,3]) 
                if ST == 1:
                    gen_inv.columns = ['Total']
                else:
                    gen_inv.columns = ['Step '+str(st)]
                gen_inv.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
                Generator_Investment_Cost = pd.concat([Generator_Investment_Cost, gen_inv], axis=1).fillna(0)
         Generator_Investment_Cost = Generator_Investment_Cost.T.groupby(level=[0], sort=False).sum().T
         gen_inv_tot = Generator_Investment_Cost.sum(1).to_frame()
         gen_inv_tot.columns = ['Total']
         if ST != 1:
            Generator_Investment_Cost = pd.concat([Generator_Investment_Cost, gen_inv_tot],axis=1)
        
    else:  # continuous (LP) capacity expansion — capacities are continuous variables rather than discrete units
        
     "Renewable sources"
     RES_Units = instance.RES_Units.get_values()                                  # continuous installed capacity multiplier per step
     RES_Nominal_Capacity = instance.RES_Nominal_Capacity.extract_values()
     RES_capacity = instance.RES_capacity.extract_values()
     if instance.Greenfield_Investment.value == 1:  # see the RES_capacity comment in the MILP branch above (Bug fix 2026-09-04)
        RES_capacity = {r: 0.0 for r in RES_capacity}
     RES_Inv_Specific_Cost = instance.RES_Specific_Investment_Cost.extract_values()
     RES_Investment_Cost = pd.DataFrame()
     for r in range(1,R+1):
        r_inv = ((RES_Units[1,r]*RES_Nominal_Capacity[r])-RES_capacity[r])*RES_Inv_Specific_Cost[r]  # step-1 investment beyond pre-existing capacity
        res_inv = pd.DataFrame(['Investment cost', RES_Names[r], '-', 'kUSD', r_inv/1e3]).T.set_index([0,1,2,3]) 
        if ST == 1:
            res_inv.columns = ['Total']
        else:
            res_inv.columns = ['Step 1']
        res_inv.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
        RES_Investment_Cost = pd.concat([RES_Investment_Cost, res_inv], axis=1).fillna(0)
        for (y,st) in tup_list:
            r_inv = (RES_Units[st,r]-RES_Units[st-1,r])*RES_Nominal_Capacity[r]*RES_Inv_Specific_Cost[r]/((1+Discount_Rate)**(y-1))  # discounted incremental investment for later steps
            res_inv = pd.DataFrame(['Investment cost', RES_Names[r], '-', 'kUSD', r_inv/1e3]).T.set_index([0,1,2,3]) 
            if ST == 1:
                res_inv.columns = ['Total']
            else:
                res_inv.columns = ['Step '+str(st)]
            res_inv.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
            RES_Investment_Cost = pd.concat([RES_Investment_Cost, res_inv], axis=1).fillna(0)
     RES_Investment_Cost = RES_Investment_Cost.T.groupby(level=[0], sort=False).sum().T
     res_inv_tot = RES_Investment_Cost.sum(1).to_frame()
     res_inv_tot.columns = ['Total']
     if ST != 1:
        RES_Investment_Cost = pd.concat([RES_Investment_Cost, res_inv_tot],axis=1)
        
     "Battery bank"
     if instance.Model_Components.value == 0 or instance.Model_Components.value == 1:
         BESS_Nominal_Capacity = instance.Battery_Nominal_Capacity.get_values()    # continuous installed battery capacity per step
         BESS_capacity = instance.Battery_capacity.value
         if instance.Greenfield_Investment.value == 1:  # see the RES_capacity comment in the MILP branch above (Bug fix 2026-09-04)
            BESS_capacity = 0.0
         BESS_Inv_Specific_Cost = instance.Battery_Specific_Investment_Cost.value
         BESS_Investment_Cost = pd.DataFrame()
         b_inv = (BESS_Nominal_Capacity[1]-BESS_capacity)*BESS_Inv_Specific_Cost  # step-1 investment beyond pre-existing capacity
         bess_inv = pd.DataFrame(['Investment cost', 'Battery bank', '-', 'kUSD', b_inv/1e3]).T.set_index([0,1,2,3]) 
         if ST == 1:
            bess_inv.columns = ['Total']
         else:
            bess_inv.columns = ['Step 1']
         bess_inv.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
         BESS_Investment_Cost = pd.concat([BESS_Investment_Cost, bess_inv], axis=1).fillna(0)
         for (y,st) in tup_list:
            b_inv = (BESS_Nominal_Capacity[st]-BESS_Nominal_Capacity[st-1])*BESS_Inv_Specific_Cost/((1+Discount_Rate)**(y-1))  # discounted incremental battery investment for later steps
            bess_inv = pd.DataFrame(['Investment cost', 'Battery bank', '-', 'kUSD', b_inv/1e3]).T.set_index([0,1,2,3]) 
            if ST == 1:
                bess_inv.columns = ['Total']
            else:
                bess_inv.columns = ['Step '+str(st)]
            bess_inv.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
            BESS_Investment_Cost = pd.concat([BESS_Investment_Cost, bess_inv], axis=1).fillna(0)
         BESS_Investment_Cost = BESS_Investment_Cost.T.groupby(level=[0], sort=False).sum().T
         bess_inv_tot = BESS_Investment_Cost.sum(1).to_frame()
         bess_inv_tot.columns = ['Total']
         if ST != 1:
            BESS_Investment_Cost = pd.concat([BESS_Investment_Cost, bess_inv_tot],axis=1)
        
     "Generator"
     if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:
         Generator_Capacity = instance.Generator_Nominal_Capacity.get_values()      # continuous installed generator capacity per step
         Generator_Inv_Specific_Cost = instance.Generator_Specific_Investment_Cost.extract_values()
         Generator_capacity = instance.Generator_capacity.extract_values()
         if instance.Greenfield_Investment.value == 1:  # see the RES_capacity comment in the MILP branch above (Bug fix 2026-09-04)
            Generator_capacity = {g: 0.0 for g in Generator_capacity}
         Generator_Investment_Cost = pd.DataFrame()
         for g in range(1,G+1):
            g_inv = (Generator_Capacity[1,g]-Generator_capacity[g])*Generator_Inv_Specific_Cost[g]  # step-1 investment beyond pre-existing capacity
            gen_inv = pd.DataFrame(['Investment cost', Generator_Names[g], '-', 'kUSD', g_inv/1e3]).T.set_index([0,1,2,3]) 
            if ST == 1:
                gen_inv.columns = ['Total']
            else:
                gen_inv.columns = ['Step 1']
            gen_inv.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
            Generator_Investment_Cost = pd.concat([Generator_Investment_Cost, gen_inv], axis=1).fillna(0)
            for (y,st) in tup_list:
                g_inv = (Generator_Capacity[st,g]-Generator_Capacity[st-1,g])*Generator_Inv_Specific_Cost[g]/((1+Discount_Rate)**(y-1))  # discounted incremental investment for later steps
                gen_inv = pd.DataFrame(['Investment cost', Generator_Names[g], '-', 'kUSD', g_inv/1e3]).T.set_index([0,1,2,3]) 
                if ST == 1:
                    gen_inv.columns = ['Total']
                else:
                    gen_inv.columns = ['Step '+str(st)]
                gen_inv.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
                Generator_Investment_Cost = pd.concat([Generator_Investment_Cost, gen_inv], axis=1).fillna(0)
         Generator_Investment_Cost = Generator_Investment_Cost.T.groupby(level=[0], sort=False).sum().T
         gen_inv_tot = Generator_Investment_Cost.sum(1).to_frame()
         gen_inv_tot.columns = ['Total']
         if ST != 1:
            Generator_Investment_Cost = pd.concat([Generator_Investment_Cost, gen_inv_tot],axis=1)

    "National Grid"
    if instance.Grid_Connection.value == 1:  # grid connection cost table, only if the microgrid is grid-connected
        Grid_Connection_Specific_Cost = instance.Grid_Connection_Cost.extract_values()  
        Grid_Distance = instance.Grid_Distance.extract_values()   
        Grid_Investment_Cost = pd.DataFrame()
        gr_inv = Grid_Distance[None]*Grid_Connection_Specific_Cost[None]*instance.Grid_Connection.value  # step-1 (undiscounted) connection cost = distance * unit cost
        grid_inv = pd.DataFrame(['Investment cost', 'National Grid', '-', 'kUSD', gr_inv/1e3]).T.set_index([0,1,2,3]) 
        if ST == 1:
            grid_inv.columns = ['Total']
        else:
            grid_inv.columns = ['Step 1']
        grid_inv.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
        Grid_Investment_Cost = pd.concat([Grid_Investment_Cost, grid_inv], axis=1).fillna(0)
        for (y,st) in tup_list:
            gr_inv = (Grid_Distance[None]*Grid_Connection_Specific_Cost[None]*instance.Grid_Connection.value)/((1+Discount_Rate)**(y-1))  # discounted grid connection cost for later steps
            grid_inv = pd.DataFrame(['Investment cost', 'National Grid', '-', 'kUSD', gr_inv/1e3]).T.set_index([0,1,2,3]) 
            if ST == 1:
                grid_inv.columns = ['Total']
            else:
                grid_inv.columns = ['Step '+str(st)]
            grid_inv.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
            Grid_Investment_Cost = pd.concat([Grid_Investment_Cost, grid_inv], axis=1).fillna(0)
        Grid_Investment_Cost = Grid_Investment_Cost.T.groupby(level=[0], sort=False).sum().T
        grid_inv_tot = Grid_Investment_Cost.sum(1).to_frame()
        grid_inv_tot.columns = ['Total']
        if ST != 1:
            Grid_Investment_Cost = pd.concat([Grid_Investment_Cost, grid_inv_tot],axis=1)
 

    #%% Fixed costs  
    
    "Renewable sources"    
    RES_OM_Specific_Cost = instance.RES_Specific_OM_Cost.extract_values()          # O&M cost as a fraction of investment cost, per year
    RES_Fixed_Cost = pd.DataFrame()
    for r in range(1,R+1):
        r_fc = 0
        for (y,st) in yu_tuples_list:
            if instance.MILP_Formulation.value:
               r_fc += RES_Units_milp[st,r]*RES_Nominal_Capacity[r]*RES_Inv_Specific_Cost[r]*RES_OM_Specific_Cost[r]/((1+Discount_Rate)**(y))  # discounted O&M cost on installed capacity for year y (MILP)
            else:
               r_fc += RES_Units[st,r]*RES_Nominal_Capacity[r]*RES_Inv_Specific_Cost[r]*RES_OM_Specific_Cost[r]/((1+Discount_Rate)**(y))       # discounted O&M cost on installed capacity for year y (LP)
        res_fc = pd.DataFrame(['Fixed cost', RES_Names[r], '-', 'kUSD', r_fc/1e3]).T.set_index([0,1,2,3]) 
        res_fc.columns = ['Total']
        res_fc.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
        RES_Fixed_Cost = pd.concat([RES_Fixed_Cost, res_fc], axis=1).fillna(0)
    RES_Fixed_Cost = RES_Fixed_Cost.T.groupby(level=[0], sort=False).sum().T
    "Battery bank"
    if instance.Model_Components.value == 0 or instance.Model_Components.value == 1:
        BESS_OM_Specific_Cost = instance.Battery_Specific_OM_Cost.value    
        BESS_Fixed_Cost = pd.DataFrame()
        b_fc = 0
        for (y,st) in yu_tuples_list:
            if instance.MILP_Formulation.value:
               b_fc += BESS_Units[st]*BESS_Nominal_Capacity_milp*BESS_Inv_Specific_Cost*BESS_OM_Specific_Cost/((1+Discount_Rate)**(y))  # discounted battery O&M cost for year y (MILP)
            else:
               b_fc += BESS_Nominal_Capacity[st]*BESS_Inv_Specific_Cost*BESS_OM_Specific_Cost/((1+Discount_Rate)**(y))                  # discounted battery O&M cost for year y (LP)
        bess_fc = pd.DataFrame(['Fixed cost', 'Battery bank', '-', 'kUSD', b_fc/1e3]).T.set_index([0,1,2,3]) 
        bess_fc.columns = ['Total']
        bess_fc.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
        BESS_Fixed_Cost = pd.concat([BESS_Fixed_Cost, bess_fc], axis=1).fillna(0) 
        BESS_Fixed_Cost = BESS_Fixed_Cost.T.groupby(level=[0], sort=False).sum().T
    "Generators"
    if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:
        Generator_OM_Specific_Cost = instance.Generator_Specific_OM_Cost.extract_values()     
        Generator_Fixed_Cost = pd.DataFrame()
        for g in range(1,G+1):
            g_fc = 0
            for (y,st) in yu_tuples_list:
                if instance.MILP_Formulation.value:
                   g_fc += (Generator_Units[st,g]*Generator_Nominal_Capacity_milp[g])*Generator_Inv_Specific_Cost[g]*Generator_OM_Specific_Cost[g]/((1+Discount_Rate)**(y))  # discounted generator O&M cost for year y (MILP)
                else:
                    g_fc += Generator_Capacity[st,g]*Generator_Inv_Specific_Cost[g]*Generator_OM_Specific_Cost[g]/((1+Discount_Rate)**(y))                                  # discounted generator O&M cost for year y (LP)
            gen_fc = pd.DataFrame(['Fixed cost', Generator_Names[g], '-', 'kUSD', g_fc/1e3]).T.set_index([0,1,2,3]) 
            gen_fc.columns = ['Total']
            gen_fc.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
            Generator_Fixed_Cost = pd.concat([Generator_Fixed_Cost, gen_fc], axis=1).fillna(0)
        Generator_Fixed_Cost = Generator_Fixed_Cost.T.groupby(level=[0], sort=False).sum().T
    "National Grid"
    if instance.Grid_Connection.value == 1:
        Grid_OM_Cost = (instance.Grid_Connection_Cost.value * instance.Grid_Connection.value * instance.Grid_Distance.value)* instance.Grid_Maintenance_Cost.value   # annual maintenance cost = connection cost * maintenance rate
        Grid_Fixed_Cost = pd.DataFrame()
        g_fc = 0
        for (y,st) in yu_tuples_list:
            if y < instance.Year_Grid_Connection.extract_values()[None]:
                g_fc += (0)/((1+Discount_Rate)**(y))                     # no maintenance cost before the grid connection year
            else:
                g_fc += (Grid_OM_Cost)/((1+Discount_Rate)**(y))          # discounted maintenance cost once connected
        grid_fc = pd.DataFrame(['Fixed cost', 'National Grid', '-', 'kUSD', g_fc/1e3]).T.set_index([0,1,2,3]) 
        grid_fc.columns = ['Total']
        grid_fc.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
        Grid_Fixed_Cost = pd.concat([Grid_Fixed_Cost, grid_fc], axis=1).fillna(0)
        Grid_Fixed_Cost = Grid_Fixed_Cost.T.groupby(level=[0], sort=False).sum().T
    "Total"
    if instance.Grid_Connection.value == 1:
        Fixed_Costs_Act = pd.DataFrame(['Total fixed O&M cost', 'System', '-', 'kUSD', (instance.Operation_Maintenance_Cost_Act.value + Grid_Fixed_Cost.iloc[0]['Total']*1000)/1e3]).T.set_index([0,1,2,3])  # model's O&M cost plus grid maintenance
    if instance.Grid_Connection.value == 0:
        Fixed_Costs_Act = pd.DataFrame(['Total fixed O&M cost', 'System', '-', 'kUSD', (instance.Operation_Maintenance_Cost_Act.value)/1e3]).T.set_index([0,1,2,3])
    Fixed_Costs_Act.columns = ['Total']
    Fixed_Costs_Act.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
    #%% Variable costs

    "Grid electricity cost and revenue"  
    if instance.Grid_Connection.value == 1:
        Grid_El_Cost = pd.DataFrame()
        Grid_El_Rev = pd.DataFrame()
        for s in range(1,S+1):
            Grid_Cost = pd.DataFrame(['Grid electricity cost', "National Grid", s , 'kUSD', instance.Total_Electricity_Cost_Act.get_values()[s]/1e3]).T   # discounted cost of electricity purchased from the grid, per scenario
            Grid_Rev = pd.DataFrame(['Grid electricity revenue', "National Grid", s , 'kUSD', instance.Total_Revenues_Act.get_values()[s]/1e3]).T          # discounted revenue from electricity sold to the grid, per scenario
            Grid_El_Cost = pd.concat([Grid_El_Cost, Grid_Cost], axis=0)
            Grid_El_Rev = pd.concat([Grid_El_Rev, Grid_Rev], axis=0)
        Grid_El_Cost = Grid_El_Cost.set_index([0,1,2,3])
        Grid_El_Rev = Grid_El_Rev.set_index([0,1,2,3])
        # Bug fix 2026-09-04, found via linopy Stage 5 cross-validation: every other cost-table
        # block in this function names its index levels right after set_index([0,1,2,3]) (see
        # RES_Investment_Cost etc. above) -- this was the one block that didn't. Once concatenated
        # into the overall Costs table below, the missing names collapsed the whole MultiIndex's
        # level names for any grid-connected design, breaking PrintResults()'s later
        # `Costs.xs(..., level='Cost item')` lookup with `KeyError: 'Level Cost item not found'`.
        Grid_El_Cost.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
        Grid_El_Rev.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
        Grid_El_Cost.columns = ['Total']
        Grid_El_Rev.columns = ['Total']
    
    "Total"
    Variable_Costs_Act = pd.DataFrame()
    for s in range(1,S+1):
        if instance.Grid_Connection.value == 1:
            Variable_Cost = pd.DataFrame(['Total variable O&M cost', 'System', s, 'kUSD', (instance.Total_Scenario_Variable_Cost_Act.get_values()[s] - instance.Operation_Maintenance_Cost_Act.value +
                                          Grid_El_Rev.iloc[0]['Total']*1000)/1e3]).T   # scenario variable cost minus fixed O&M (already counted separately) plus grid revenue add-back
        if instance.Grid_Connection.value == 0:
            Variable_Cost = pd.DataFrame(['Total variable O&M cost', 'System', s, 'kUSD', (instance.Total_Scenario_Variable_Cost_Act.get_values()[s] - instance.Operation_Maintenance_Cost_Act.value)/1e3]).T
        Variable_Costs_Act = pd.concat([Variable_Costs_Act, Variable_Cost], axis=0)
    Variable_Costs_Act = Variable_Costs_Act.set_index([0,1,2,3])
    Variable_Costs_Act.columns = ['Total']
    Variable_Costs_Act.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']

    "Replacement cost"
    if instance.Model_Components.value == 0 or instance.Model_Components.value == 1:
        BESS_Replacement_Cost = pd.DataFrame()
        for s in range(1,S+1):
            BESS_Rep = pd.DataFrame(['Replacement cost', 'Battery bank', s, 'kUSD', instance.Battery_Replacement_Cost_Act.get_values()[s]/1e3]).T   # discounted battery replacement cost per scenario
            BESS_Replacement_Cost = pd.concat([BESS_Replacement_Cost, BESS_Rep], axis=0)
        BESS_Replacement_Cost = BESS_Replacement_Cost.set_index([0,1,2,3])
        BESS_Replacement_Cost.columns = ['Total']
        BESS_Replacement_Cost.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']

    "Fuel cost" 
    if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:
        Fuel_Cost = pd.DataFrame()
        for g in range(1,G+1):   
            for s in range(1,S+1):
                fc = pd.DataFrame(['Fuel cost', Fuel_Names[g], s, 'kUSD', instance.Total_Fuel_Cost_Act.get_values()[s,g]/1e3]).T   # discounted fuel cost per generator/scenario
                Fuel_Cost = pd.concat([Fuel_Cost, fc], axis=0)
        Fuel_Cost = Fuel_Cost.set_index([0,1,2,3])
        Fuel_Cost.columns = ['Total']
        Fuel_Cost.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
    
    "Lost load cost"
    LostLoad_Cost = pd.DataFrame()        
    for s in range(1,S+1):
        LostLoad = pd.DataFrame(['Lost load cost', 'System', s, 'kUSD', instance.Scenario_Lost_Load_Cost_Act.get_values()[s]/1e3]).T   # discounted cost of unserved demand per scenario
        LostLoad_Cost = pd.concat([LostLoad_Cost, LostLoad], axis=0)
    LostLoad_Cost = LostLoad_Cost.set_index([0,1,2,3])
    LostLoad_Cost.columns = ['Total']
    LostLoad_Cost.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']


    #%% Salvage value
    Salvage_Value = pd.DataFrame(['Salvage value', 'System', '-', 'kUSD', -instance.Salvage_Value.value/1e3]).T.set_index([0,1,2,3])   # residual value of assets at end of project life (stored as a negative cost, i.e. a credit)
    Salvage_Value.columns = ['Total']
    Salvage_Value.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
    
    #%% Emission
    CO2_out = pd.DataFrame()  
    for s in range(1,S+1):
        CO = pd.DataFrame(['TOTAL CO2 emission', 'System', s, 'ton', instance.Scenario_CO2_emission.get_values()[s]/1e3]).T   # total CO2 emissions per scenario
        CO2_out = pd.concat([CO2_out, CO], axis=0)
    CO2_out = CO2_out.set_index([0,1,2,3])
    CO2_out.columns = ['Total']
    CO2_out.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
    
    if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:
        CO2_fuel = pd.DataFrame()  
        for s in range(1,S+1):
            CO = pd.DataFrame(['Fuel CO2 emission', 'System', s, 'ton', instance.Scenario_FUEL_emission.get_values()[s]/1e3]).T   # CO2 emissions from fuel combustion per scenario
            CO2_fuel = pd.concat([CO2_fuel, CO], axis=0)
        CO2_fuel = CO2_fuel.set_index([0,1,2,3])
        CO2_fuel.columns = ['Total']
        CO2_fuel.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
    
    RES_emission = pd.DataFrame(['RES CO2 emission', 'RES', '-', 'ton', instance.RES_emission.value/1e3]).T.set_index([0,1,2,3])   # embodied/lifecycle emissions from RES manufacturing
    RES_emission.columns = ['Total']
    RES_emission.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
    
    if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:
        GEN_emission = pd.DataFrame(['GEN CO2 emission', 'GEN', '-', 'ton', instance.GEN_emission.value/1e3]).T.set_index([0,1,2,3])   # embodied emissions from generator manufacturing
        GEN_emission.columns = ['Total']
        GEN_emission.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
   
    if instance.Model_Components.value == 0 or instance.Model_Components.value == 1:
        BESS_emission = pd.DataFrame(['BESS CO2 emission', 'BESS', '-', 'ton', instance.BESS_emission.value/1e3]).T.set_index([0,1,2,3])   # embodied emissions from battery manufacturing
        BESS_emission.columns = ['Total']
        BESS_emission.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
    
    if instance.Grid_Connection.value == 1:
        CO2_grid = pd.DataFrame()  
        for s in range(1,S+1):
            CO_grid = pd.DataFrame(['Grid CO2 emission', 'System', s, 'ton', instance.Scenario_GRID_emission.get_values()[s]/1e3]).T   # emissions attributable to imported grid electricity
            CO2_grid = pd.concat([CO2_grid, CO_grid], axis=0)
        CO2_grid = CO2_grid.set_index([0,1,2,3])
        # Bug fix 2026-09-04, found via linopy Stage 5 cross-validation, same root cause as the
        # Grid_El_Cost/Grid_El_Rev fix above: this block never named its index levels, unlike
        # every sibling emission block (CO2_fuel, RES_emission, GEN_emission, BESS_emission).
        CO2_grid.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
        CO2_grid.columns = ['Total']
     #%% Net present cost
    if Optimization_Goal == 1:  # single-objective cost minimization            
        if instance.Grid_Connection.value == 1:
            Net_Present_Cost = pd.DataFrame(['Weighted Net present cost', 'System', '-', 'kUSD', (instance.ObjectiveFuntion.expr() + Grid_Investment + Grid_Fixed_Cost.iloc[0]['Total']*1000)/1e3]).T.set_index([0,1,2,3])  # objective value plus grid investment & fixed cost (excluded from the objective)
        if instance.Grid_Connection.value == 0:
            Net_Present_Cost = pd.DataFrame(['Weighted Net present cost', 'System', '-', 'kUSD', (instance.ObjectiveFuntion.expr())/1e3]).T.set_index([0,1,2,3])
        Net_Present_Cost.columns = ['Total']
        Net_Present_Cost.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
                                    
    elif Optimization_Goal == 2:  # multi-objective (cost + emissions)
        if instance.Grid_Connection.value == 1:
            Net_Present_Cost = pd.DataFrame(['Weighted Net present cost', 'System', '-', 'kUSD', (instance.ObjectiveFuntion.expr() + Grid_Investment + Grid_Fixed_Cost.iloc[0]['Total']*1000)/1e3]).T.set_index([0,1,2,3])
        if instance.Grid_Connection.value == 0:
            Net_Present_Cost = pd.DataFrame(['Weighted Net present cost', 'System', '-', 'kUSD', (instance.ObjectiveFuntion.expr())/1e3]).T.set_index([0,1,2,3])
        Net_Present_Cost.columns = ['Total']
        Net_Present_Cost.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
    
    NPC = pd.DataFrame()
    for s in range (1,S+1):
        if instance.Grid_Connection.value == 1:
            Net_Present_Cost_sc = pd.DataFrame(['Net present cost ', 'System',s, 'kUSD', (instance.Scenario_Net_Present_Cost.get_values()[s] + Grid_Investment + Grid_Fixed_Cost.iloc[0]['Total']*1000)/1e3]).T   # per-scenario NPC including grid cost
        if instance.Grid_Connection.value == 0:
            Net_Present_Cost_sc = pd.DataFrame(['Net present cost ', 'System',s, 'kUSD', (instance.Scenario_Net_Present_Cost.get_values()[s])/1e3]).T
        NPC = pd.concat([NPC,Net_Present_Cost_sc], axis=0)
    NPC = NPC.set_index([0,1,2,3])
    NPC.columns = ['Total']
    NPC.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
    #%%LCOE
    Electric_Demand = pd.DataFrame.from_dict(instance.Energy_Demand.extract_values(), orient='index') #[Wh]
    Electric_Demand.index = pd.MultiIndex.from_tuples(list(Electric_Demand.index))
    Scenario_Weight_map = instance.Scenario_Weight.extract_values()
    # Weight each (scenario,year,period) demand value by its scenario weight before collapsing
    # scenarios in the groupby below, so the result is a weighted-average demand per year
    # (matching how the weighted NPC/objective already weights each scenario) rather than a
    # raw sum across all scenarios, which previously inflated this denominator ~S-fold and
    # made the aggregate LCOE artificially small for any multi-scenario (S>1) run.
    Electric_Demand.iloc[:,0] = Electric_Demand.iloc[:,0] * Electric_Demand.index.get_level_values(0).map(Scenario_Weight_map)  # apply scenario weight to each demand value
    Electric_Demand = Electric_Demand.groupby(level=[1], sort=False).sum()   # collapse scenario level, summing the weighted demand into a per-year series
    Energy_Demand = instance.Energy_Demand.extract_values()
    
    LCOE_scenarios = pd.DataFrame()
    for s in range(1,S+1):
        LC = pd.DataFrame(['Levelized Cost of Energy scenarios','System',s,'USD/kWh',(instance.Scenario_Net_Present_Cost.get_values()[s]/sum(sum(Energy_Demand[s,i,t] for t in range(1,P+1))/(1+Discount_Rate)**i for i in range(1,(Y+1)))) ]).T   # scenario NPC / discounted scenario demand
        LCOE_scenarios = pd.concat([LCOE_scenarios, LC], axis=0)
    LCOE_scenarios = LCOE_scenarios.set_index([0,1,2,3])
    LCOE_scenarios.columns = ['Total']
    LCOE_scenarios.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
    
    Net_Present_Demand = sum(Electric_Demand.iloc[i-1,0]/(1+Discount_Rate)**i for i in range(1,(Y+1)))    #[Wh]  # discounted weighted-average demand over the project life
    if instance.Grid_Connection.value == 1:
        LCOE = pd.DataFrame([(Net_Present_Cost.iloc[0,0] + Grid_El_Rev.iloc[0]['Total']*1000/1e3)/Net_Present_Demand])*1e3    #[USD/KWh]   # weighted NPC (plus revenue add-back) divided by discounted demand
    if instance.Grid_Connection.value == 0:
        LCOE = pd.DataFrame([(Net_Present_Cost.iloc[0,0])/Net_Present_Demand])*1e3    #[USD/KWh]
    LCOE.index = pd.MultiIndex.from_arrays([['Levelized Cost of Energy '],['System'],['-'],['USD/kWh']])
    LCOE.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']
    LCOE.columns = ['Total']
    #%% Concatenating
    # Stack every cost/emission sub-table into one long DataFrame, indexed by ('Cost item',
    # 'Component', 'Scenario', 'Unit'). Which tables are included depends on which components
    # (RES/BESS/Generator) and grid connection are active in this run.
    if instance.Model_Components.value == 0:  # RES + Battery + Generator
        if instance.Grid_Connection.value == 1:
            SystemCost = pd.concat([round(Net_Present_Cost.astype(float),3),
                                    round(NPC.astype(float),3),
                                    round(Total_Investment_Cost.astype(float),3),
                                    round(Fixed_Costs_Act.astype(float),3),
                                    round(Variable_Costs_Act.astype(float),3),
                                    round(Salvage_Value.astype(float),3),                           
                                    round(LCOE.astype(float),4),
                                    round(LCOE_scenarios.astype(float),4),
                                    round(RES_Investment_Cost.astype(float),3),
                                    round(BESS_Investment_Cost.astype(float),3),
                                    round(Generator_Investment_Cost.astype(float),3),
                                    round(Grid_Investment_Cost.astype(float),3),
                                    round(RES_Fixed_Cost.astype(float),3),
                                    round(BESS_Fixed_Cost.astype(float),3),
                                    round(Generator_Fixed_Cost.astype(float),3),
                                    round(Grid_Fixed_Cost.astype(float),3),
                                    round(LostLoad_Cost.astype(float),3),
                                    round(BESS_Replacement_Cost.astype(float),3),
                                    round(Fuel_Cost.astype(float),3),
                                    round(Grid_El_Cost.astype(float),3),
                                    round(Grid_El_Rev.astype(float),3),
                                    round(CO2_fuel.astype(float),3),
                                    round(CO2_grid.astype(float),3),
                                    round(RES_emission.astype(float),3),
                                    round(GEN_emission.astype(float),3),
                                    round(BESS_emission.astype(float),3),
                                    round(CO2_out.astype(float),3)], axis=0).fillna('-')
        if instance.Grid_Connection.value == 0:
            SystemCost = pd.concat([round(Net_Present_Cost.astype(float),3),
                                    round(NPC.astype(float),3),
                                    round(Total_Investment_Cost.astype(float),3),
                                    round(Fixed_Costs_Act.astype(float),3),
                                    round(Variable_Costs_Act.astype(float),3),
                                    round(Salvage_Value.astype(float),3),                           
                                    round(LCOE.astype(float),4),
                                    round(LCOE_scenarios.astype(float),4),
                                    round(RES_Investment_Cost.astype(float),3),
                                    round(BESS_Investment_Cost.astype(float),3),
                                    round(Generator_Investment_Cost.astype(float),3),
                                    round(RES_Fixed_Cost.astype(float),3),
                                    round(BESS_Fixed_Cost.astype(float),3),
                                    round(Generator_Fixed_Cost.astype(float),3),
                                    round(LostLoad_Cost.astype(float),3),
                                    round(BESS_Replacement_Cost.astype(float),3),
                                    round(Fuel_Cost.astype(float),3),
                                    round(CO2_fuel.astype(float),3),
                                    round(RES_emission.astype(float),3),
                                    round(GEN_emission.astype(float),3),
                                    round(BESS_emission.astype(float),3),
                                    round(CO2_out.astype(float),3)], axis=0).fillna('-')
    
    if instance.Model_Components.value == 1:
        if instance.Grid_Connection.value == 1:
            SystemCost = pd.concat([round(Net_Present_Cost.astype(float),3),
                                    round(NPC.astype(float),3),
                                    round(Total_Investment_Cost.astype(float),3),
                                    round(Fixed_Costs_Act.astype(float),3),
                                    round(Variable_Costs_Act.astype(float),3),
                                    round(Salvage_Value.astype(float),3),                           
                                    round(LCOE.astype(float),4),
                                    round(LCOE_scenarios.astype(float),4),
                                    round(RES_Investment_Cost.astype(float),3),
                                    round(BESS_Investment_Cost.astype(float),3),                      
                                    round(Grid_Investment_Cost.astype(float),3),
                                    round(RES_Fixed_Cost.astype(float),3),
                                    round(BESS_Fixed_Cost.astype(float),3),
                                    round(Grid_Fixed_Cost.astype(float),3),
                                    round(LostLoad_Cost.astype(float),3),
                                    round(BESS_Replacement_Cost.astype(float),3),
                                    round(Grid_El_Cost.astype(float),3),
                                    round(Grid_El_Rev.astype(float),3),
                                    round(CO2_grid.astype(float),3),
                                    round(RES_emission.astype(float),3),
                                    round(BESS_emission.astype(float),3),
                                    round(CO2_out.astype(float),3)], axis=0).fillna('-')
        if instance.Grid_Connection.value == 0:
            SystemCost = pd.concat([round(Net_Present_Cost.astype(float),3),
                                    round(NPC.astype(float),3),
                                    round(Total_Investment_Cost.astype(float),3),
                                    round(Fixed_Costs_Act.astype(float),3),
                                    round(Variable_Costs_Act.astype(float),3),
                                    round(Salvage_Value.astype(float),3),                           
                                    round(LCOE.astype(float),4),
                                    round(LCOE_scenarios.astype(float),4),
                                    round(RES_Investment_Cost.astype(float),3),
                                    round(BESS_Investment_Cost.astype(float),3),                                                        
                                    round(RES_Fixed_Cost.astype(float),3),
                                    round(BESS_Fixed_Cost.astype(float),3),
                                    round(LostLoad_Cost.astype(float),3),
                                    round(BESS_Replacement_Cost.astype(float),3),
                                    round(RES_emission.astype(float),3),
                                    round(BESS_emission.astype(float),3),
                                    round(CO2_out.astype(float),3)], axis=0).fillna('-')
    
    if instance.Model_Components.value == 2:
        if instance.Grid_Connection.value == 1:
            SystemCost = pd.concat([round(Net_Present_Cost.astype(float),3),
                                    round(NPC.astype(float),3),
                                    round(Total_Investment_Cost.astype(float),3),
                                    round(Fixed_Costs_Act.astype(float),3),
                                    round(Variable_Costs_Act.astype(float),3),
                                    round(Salvage_Value.astype(float),3),                           
                                    round(LCOE.astype(float),4),
                                    round(LCOE_scenarios.astype(float),4),
                                    round(RES_Investment_Cost.astype(float),3),
                                    round(Generator_Investment_Cost.astype(float),3),
                                    round(Grid_Investment_Cost.astype(float),3),
                                    round(RES_Fixed_Cost.astype(float),3),
                                    round(Generator_Fixed_Cost.astype(float),3),
                                    round(Grid_Fixed_Cost.astype(float),3),
                                    round(LostLoad_Cost.astype(float),3),
                                    round(Fuel_Cost.astype(float),3),
                                    round(Grid_El_Cost.astype(float),3),
                                    round(Grid_El_Rev.astype(float),3),
                                    round(CO2_fuel.astype(float),3),
                                    round(CO2_grid.astype(float),3),
                                    round(RES_emission.astype(float),3),
                                    round(GEN_emission.astype(float),3),
                                    round(CO2_out.astype(float),3)], axis=0).fillna('-')
        if instance.Grid_Connection.value == 0:
            SystemCost = pd.concat([round(Net_Present_Cost.astype(float),3),
                                    round(NPC.astype(float),3),
                                    round(Total_Investment_Cost.astype(float),3),
                                    round(Fixed_Costs_Act.astype(float),3),
                                    round(Variable_Costs_Act.astype(float),3),
                                    round(Salvage_Value.astype(float),3),                           
                                    round(LCOE.astype(float),4),
                                    round(LCOE_scenarios.astype(float),4),
                                    round(RES_Investment_Cost.astype(float),3),
                                    round(Generator_Investment_Cost.astype(float),3),
                                    round(RES_Fixed_Cost.astype(float),3),
                                    round(Generator_Fixed_Cost.astype(float),3),
                                    round(LostLoad_Cost.astype(float),3),
                                    round(Fuel_Cost.astype(float),3),
                                    round(CO2_fuel.astype(float),3),
                                    round(RES_emission.astype(float),3),
                                    round(GEN_emission.astype(float),3),
                                    round(CO2_out.astype(float),3)], axis=0).fillna('-')
            
    return  SystemCost   # long-format DataFrame with all cost/emission rows for this run


#%% Plant size output
def EnergySystemSize(instance):
    # Builds the installed-capacity table (final size per component, per investment step).

    #%% Importing parameters
    S  = int(instance.Scenarios.extract_values()[None])
    P  = int(instance.Periods.extract_values()[None])
    Y  = int(instance.Years.extract_values()[None])
    ST = int(instance.Steps_Number.extract_values()[None])
    R  = int(instance.RES_Sources.extract_values()[None])
    G  = int(instance.Generator_Types.extract_values()[None])

    RES_Names = instance.RES_Names.extract_values()
    Generator_Names = instance.Generator_Names.extract_values()
    Fuel_Names = instance.Fuel_Names.extract_values()
    
    upgrade_years_list = [1 for i in range(ST)]                # first year of each investment step
    s_dur = instance.Step_Duration.value
    for i in range(1, ST): 
        upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    
    yu_tuples_list = [0 for i in range(1,Y+1)]                  # (year, step) for every project year
    if ST == 1:    
        for y in range(1,Y+1):            
            yu_tuples_list[y-1] = (y, 1)    
    else:        
        for y in range(1,Y+1):            
            for i in range(len(upgrade_years_list)-1):
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:
                    yu_tuples_list[y-1] = (y, [st for st in range(ST)][i+1])                
                elif y >= upgrade_years_list[-1]:
                    yu_tuples_list[y-1] = (y, ST)   
    tup_list = [[] for i in range(ST-1)]                        # (year, step) only for the first year of each step after the first
    for i in range(0,ST-1):
        tup_list[i] = yu_tuples_list[s_dur*i + s_dur]
        
    #%%
    if instance.MILP_Formulation.value:  # unit-based (integer) capacity expansion
     "Renewable sources"
     RES_Units_milp = instance.RES_Units_milp.get_values()
     RES_Nominal_Capacity = instance.RES_Nominal_Capacity.extract_values()
     RES_Size = pd.DataFrame()
     for r in range(1,R+1):
        r_size = RES_Units_milp[1,r]*RES_Nominal_Capacity[r]       # step-1 installed capacity in kW
        res_size = pd.DataFrame([RES_Names[r], 'kW', r_size]).T.set_index([0,1])
        if ST == 1 :
            res_size.columns = ['Total']
        else:
            res_size.columns = ['Step 1']
        res_size.index.names = ['Component', 'Unit']
        RES_Size = pd.concat([RES_Size,res_size], axis=1).fillna(0)
        for (y,st) in tup_list:
            res_size = pd.DataFrame([RES_Names[r], 'kW', ((RES_Units_milp[st,r]-RES_Units_milp[st-1,r])*RES_Nominal_Capacity[r])]).T.set_index([0,1])  # incremental capacity added at step st
            if ST == 1:
                res_size.columns = ['Total']
            else:
                res_size.columns = ['Step '+str(st)]
            res_size.index.names = ['Component', 'Unit']
            RES_Size = pd.concat([RES_Size,res_size], axis=1).fillna(0)
     RES_Size = RES_Size.T.groupby(level=[0], sort=False).sum().T
     res_size_tot = RES_Size.sum(1).to_frame()                          # cumulative final size across all steps
     res_size_tot.columns = ['Total']
     if ST != 1:
        RES_Size = pd.concat([RES_Size, res_size_tot],axis=1)
        
     "Battery Bank"
     if instance.Model_Components.value == 0 or instance.Model_Components.value == 1:
         BESS_Nominal_Capacity_milp = instance.Battery_Nominal_Capacity_milp.value
         BESS_Units = instance.Battery_Units.get_values()
         BESS_Size = pd.DataFrame()
         bess_size = pd.DataFrame(['Battery bank', 'kWh', (BESS_Units[1]*BESS_Nominal_Capacity_milp)]).T.set_index([0,1])   # step-1 installed battery capacity in kWh
         
         if ST == 1:
            bess_size.columns = ['Total']
         else:
            bess_size.columns = ['Step 1']
         bess_size.index.names = ['Component', 'Unit']
         BESS_Size = pd.concat([BESS_Size, bess_size], axis=1).fillna(0)
         for (y,st) in tup_list:
            bess_size = pd.DataFrame(['Battery bank', 'kWh', ((BESS_Units[st]-BESS_Units[st-1])*BESS_Nominal_Capacity_milp)]).T.set_index([0,1])  # incremental battery capacity added at step st
            
            if ST == 1:
                bess_size.columns = ['Total']
            else:
                bess_size.columns = ['Step '+str(st)]
            bess_size.index.names = ['Component', 'Unit']
            BESS_Size = pd.concat([BESS_Size, bess_size], axis=1).fillna(0)     
         BESS_Size = BESS_Size.T.groupby(level=[0], sort=False).sum().T
         bess_size_tot = BESS_Size.sum(1).to_frame()
         bess_size_tot.columns = ['Total']
         if ST != 1:
            BESS_Size = pd.concat([BESS_Size, bess_size_tot],axis=1)
        
     "Generator"
     if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:
         Generator_Units = instance.Generator_Units.get_values()
         Generator_Nominal_Capacity_milp = instance.Generator_Nominal_Capacity_milp.extract_values() 
         Generator_Size = pd.DataFrame()
         for g in range(1,G+1):
            gen_size = pd.DataFrame([Generator_Names[g], 'kW', (Generator_Units[1,g]*Generator_Nominal_Capacity_milp[g])]).T.set_index([0,1])   # step-1 installed generator capacity in kW
            
            if ST == 1:
                gen_size.columns = ['Total']
            else:
                gen_size.columns = ['Step 1']
            gen_size.index.names = ['Component', 'Unit']
            Generator_Size = pd.concat([Generator_Size, gen_size], axis=1).fillna(0)
            for (y,st) in tup_list:
                gen_size = pd.DataFrame([Generator_Names[g], 'kW', ((Generator_Units[st,g]-Generator_Units[st-1,g])*Generator_Nominal_Capacity_milp[g])]).T.set_index([0,1])  # incremental generator capacity added at step st
                
                if ST == 1:
                    gen_size.columns = ['Total']
                else:
                    gen_size.columns = ['Step '+str(st)]
                gen_size.index.names = ['Component', 'Unit']
                Generator_Size = pd.concat([Generator_Size, gen_size], axis=1).fillna(0)
         Generator_Size = Generator_Size.T.groupby(level=[0], sort=False).sum().T
         gen_size_tot = Generator_Size.sum(1).to_frame()
         gen_size_tot.columns = ['Total']
         if ST != 1:
            Generator_Size = pd.concat([Generator_Size, gen_size_tot],axis=1)
         
    else:  # continuous (LP) capacity expansion
        
     "Renewable Sources"   
     RES_Units = instance.RES_Units.get_values()
     RES_Nominal_Capacity = instance.RES_Nominal_Capacity.extract_values()    
     RES_Size = pd.DataFrame()
     for r in range(1,R+1):
        r_size = RES_Units[1,r]*RES_Nominal_Capacity[r]          # step-1 installed capacity in kW
        res_size = pd.DataFrame([RES_Names[r], 'kW', r_size]).T.set_index([0,1])
        if ST == 1 :
            res_size.columns = ['Total']
        else:
            res_size.columns = ['Step 1']
        res_size.index.names = ['Component', 'Unit']
        RES_Size = pd.concat([RES_Size,res_size], axis=1).fillna(0)
        for (y,st) in tup_list:
            res_size = pd.DataFrame([RES_Names[r], 'kW', ((RES_Units[st,r]-RES_Units[st-1,r])*RES_Nominal_Capacity[r])]).T.set_index([0,1])   # incremental capacity added at step st
            if ST == 1:
                res_size.columns = ['Total']
            else:
                res_size.columns = ['Step '+str(st)]
            res_size.index.names = ['Component', 'Unit']
            RES_Size = pd.concat([RES_Size,res_size], axis=1).fillna(0)
     RES_Size = RES_Size.T.groupby(level=[0], sort=False).sum().T
     res_size_tot = RES_Size.sum(1).to_frame()
     res_size_tot.columns = ['Total']
     if ST != 1:
        RES_Size = pd.concat([RES_Size, res_size_tot],axis=1)

     "Battery bank"
     if instance.Model_Components.value == 0 or instance.Model_Components.value == 1:
         BESS_Nominal_Capacity = instance.Battery_Nominal_Capacity.get_values()
         BESS_Size = pd.DataFrame()
         bess_size = pd.DataFrame(['Battery bank', 'kWh', BESS_Nominal_Capacity[1]]).T.set_index([0,1])   # step-1 installed battery capacity in kWh
         if ST == 1:
            bess_size.columns = ['Total']
         else:
            bess_size.columns = ['Step 1']
         bess_size.index.names = ['Component', 'Unit']
         BESS_Size = pd.concat([BESS_Size, bess_size], axis=1).fillna(0)
         for (y,st) in tup_list:
            bess_size = pd.DataFrame(['Battery bank', 'kWh', (BESS_Nominal_Capacity[st]-BESS_Nominal_Capacity[st-1])]).T.set_index([0,1])   # incremental battery capacity added at step st
            if ST == 1:
                bess_size.columns = ['Total']
            else:
                bess_size.columns = ['Step '+str(st)]
            bess_size.index.names = ['Component', 'Unit']
            BESS_Size = pd.concat([BESS_Size, bess_size], axis=1).fillna(0)     
         BESS_Size = BESS_Size.T.groupby(level=[0], sort=False).sum().T
         bess_size_tot = BESS_Size.sum(1).to_frame()
         bess_size_tot.columns = ['Total']
         if ST != 1:
            BESS_Size = pd.concat([BESS_Size, bess_size_tot],axis=1)

     "Generators"
     if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:
         Generator_Capacity = instance.Generator_Nominal_Capacity.get_values()   
         Generator_Size = pd.DataFrame()
         for g in range(1,G+1):
            gen_size = pd.DataFrame([Generator_Names[g], 'kW', Generator_Capacity[1,g]]).T.set_index([0,1])   # step-1 installed generator capacity in kW
            if ST == 1:
                gen_size.columns = ['Total']
            else:
                gen_size.columns = ['Step 1']
            gen_size.index.names = ['Component', 'Unit']
            Generator_Size = pd.concat([Generator_Size, gen_size], axis=1).fillna(0)
            for (y,st) in tup_list:
                gen_size = pd.DataFrame([Generator_Names[g], 'kW', (Generator_Capacity[st,g]-Generator_Capacity[st-1,g])]).T.set_index([0,1])   # incremental capacity added at step st
                if ST == 1:
                    gen_size.columns = ['Total']
                else:
                    gen_size.columns = ['Step '+str(st)]
                gen_size.index.names = ['Component', 'Unit']
                Generator_Size = pd.concat([Generator_Size, gen_size], axis=1).fillna(0)
         Generator_Size = Generator_Size.T.groupby(level=[0], sort=False).sum().T
         gen_size_tot = Generator_Size.sum(1).to_frame()
         gen_size_tot.columns = ['Total']
         if ST != 1:
            Generator_Size = pd.concat([Generator_Size, gen_size_tot],axis=1)
               
    #%% Concatenating
    if instance.Model_Components.value == 0:  # RES + Battery + Generator
        SystemSize = pd.concat([round(RES_Size.astype(float),2),
                                round(BESS_Size.astype(float),2),
                                round(Generator_Size.astype(float),2)], axis=0).fillna('-')
    if instance.Model_Components.value == 1:  # RES + Battery only
        SystemSize = pd.concat([round(RES_Size.astype(float),2),
                                round(BESS_Size.astype(float),2)], axis=0).fillna('-')
    if instance.Model_Components.value == 2:  # RES + Generator only
        SystemSize = pd.concat([round(RES_Size.astype(float),2),
                                round(Generator_Size.astype(float),2)], axis=0).fillna('-')
        
        
    if instance.Multiobjective_Optimization == 1:  # report which point of the Pareto front this run corresponds to
        if instance.Pareto_solution == 1: 
            print("\nMULTI-OBJECTIVE OPTMIZATION: Solution for MINIMUM CO2 EMISSIONS and MAXIMUM COSTS")
        elif instance.Pareto_solution == instance.Pareto_points:
            print("\nMULTI-OBJECTIVE OPTMIZATION: Solution for MINIMUM COSTS and MAXIMUM CO2 EMISSIONS")
        else:
            print("\nMULTI-OBJECTIVE OPTMIZATION: Intermediate Solution along Pareto Curve Optimal Front")
    else: print("\nSINGLE-OBJECTIVE OPTMIZATION: Solution for MINIMUM COSTS")

    print("\n------------------------------------------------------------------------------------")
    print(SystemSize)
    print("\n------------------------------------------------------------------------------------")
    return SystemSize
#%% Yearly costs
def YearlyCosts(instance):
    # Builds a year-by-year cash-flow table (fixed costs, variable costs, replacement, fuel,
    # grid cost/revenue and lost-load cost), one row per year, one column-group per cost item.

    "Importing parameters"
    S  = int(instance.Scenarios.extract_values()[None])
    P  = int(instance.Periods.extract_values()[None])
    Y  = int(instance.Years.extract_values()[None])
    ST = int(instance.Steps_Number.extract_values()[None])
    R  = int(instance.RES_Sources.extract_values()[None])
    G  = int(instance.Generator_Types.extract_values()[None])

    RES_Names = instance.RES_Names.extract_values()
    Generator_Names = instance.Generator_Names.extract_values()
    Fuel_Names = instance.Fuel_Names.extract_values()

    "Generating years-steps tuples list"
    steps = [i for i in range(1, ST+1)]
    
    years_steps_list = [1 for i in range(1, ST+1)]
    s_dur = instance.Step_Duration.value
    for i in range(1, ST): 
        years_steps_list[i] = years_steps_list[i-1] + s_dur
    ys_tuples_list = [[] for i in range(1, Y+1)]
    for y in range(1, Y+1):  
        if len(years_steps_list) == 1:
            ys_tuples_list[y-1] = (y,1)
        else:
            for i in range(len(years_steps_list)-1):
                if y >= years_steps_list[i] and y < years_steps_list[i+1]:
                    ys_tuples_list[y-1] = (y, steps[i])       
                elif y >= years_steps_list[-1]:
                    ys_tuples_list[y-1] = (y, len(steps)) 

    #%% Fixed costs
    
    if instance.MILP_Formulation.value:  # unit-based (integer) capacity expansion
     "Renewable sources"
     RES_Units_milp = instance.RES_Units_milp.get_values()
     RES_Sources = instance.RES_Sources.value
     RES_Nominal_Capacity = instance.RES_Nominal_Capacity.extract_values()    
     RES_Inv_Specific_Cost = instance.RES_Specific_Investment_Cost.extract_values()
     RES_OM_Specific_Cost = instance.RES_Specific_OM_Cost.extract_values()
     RES_Yearly_Cost = pd.DataFrame()
     for r in range(1,RES_Sources+1):
        res_yc_source =  pd.DataFrame()
        for (y,st) in ys_tuples_list:
            res_yc = pd.DataFrame(['Year '+str(y), RES_Units_milp[st,r]*RES_Nominal_Capacity[r]*RES_Inv_Specific_Cost[r]*RES_OM_Specific_Cost[r]/1e3]).T.set_index([0])   # O&M cost on installed capacity in year y (undiscounted)
            res_yc.columns = pd.MultiIndex.from_arrays([['Fixed costs'],[RES_Names[r]],['-'],['kUSD']], names=['','Component','Scenario','Unit'])
            res_yc_source = pd.concat([res_yc_source,res_yc], axis=0)
        RES_Yearly_Cost = pd.concat([RES_Yearly_Cost,res_yc_source], axis=1)
        
     "Battery bank"
     if instance.Model_Components.value == 0 or instance.Model_Components.value == 1:
         BESS_Nominal_Capacity_milp = instance.Battery_Nominal_Capacity_milp.value
         BESS_Units = instance.Battery_Units.get_values() 
         BESS_Inv_Specific_Cost = instance.Battery_Specific_Investment_Cost.value
         BESS_OM_Specific_Cost = instance.Battery_Specific_OM_Cost.value
         BESS_Yearly_Cost = pd.DataFrame()
         for (y,st) in ys_tuples_list:
            bess_yc = pd.DataFrame(['Year '+str(y), BESS_Units[st]*BESS_Nominal_Capacity_milp*BESS_Inv_Specific_Cost*BESS_OM_Specific_Cost/1e3]).T.set_index([0])   # battery O&M cost in year y (undiscounted)
            bess_yc.columns = pd.MultiIndex.from_arrays([['Fixed costs'],['Battery bank'],['-'],['kUSD']], names=['','Component','Scenario','Unit'])
            BESS_Yearly_Cost = pd.concat([BESS_Yearly_Cost,bess_yc], axis=0)
        
     "Generator"
     if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:
         Generator_Types = instance.Generator_Types.value
         Generator_Units = instance.Generator_Units.get_values()
         Generator_Nominal_Capacity_milp = instance.Generator_Nominal_Capacity_milp.extract_values()  
         Generator_Inv_Specific_Cost = instance.Generator_Specific_Investment_Cost.extract_values()
         Generator_OM_Specific_Cost = instance.Generator_Specific_OM_Cost.extract_values()
         Generator_Yearly_Cost = pd.DataFrame()
         for g in range(1,Generator_Types+1):
            gen_yc_types =  pd.DataFrame()
            for (y,st) in ys_tuples_list:
                gen_yc = pd.DataFrame(['Year '+str(y), (Generator_Units[st,g]*Generator_Nominal_Capacity_milp[g])*Generator_Inv_Specific_Cost[g]*Generator_OM_Specific_Cost[g]/1e3]).T.set_index([0])   # generator O&M cost in year y (undiscounted)
                gen_yc.columns = pd.MultiIndex.from_arrays([['Fixed costs'],[Generator_Names[g]],['-'],['kUSD']], names=['','Component','Scenario','Unit'])
                gen_yc_types = pd.concat([gen_yc_types,gen_yc], axis=0)
            Generator_Yearly_Cost = pd.concat([Generator_Yearly_Cost,gen_yc_types], axis=1)
    else:  # continuous (LP) capacity expansion
     "Renewable Sources"
     RES_Units = instance.RES_Units.get_values()
     RES_Sources = instance.RES_Sources.value
     RES_Nominal_Capacity = instance.RES_Nominal_Capacity.extract_values()    
     RES_Inv_Specific_Cost = instance.RES_Specific_Investment_Cost.extract_values()
     RES_OM_Specific_Cost = instance.RES_Specific_OM_Cost.extract_values()
     RES_Yearly_Cost = pd.DataFrame()
     for r in range(1,RES_Sources+1):
        res_yc_source =  pd.DataFrame()
        for (y,st) in ys_tuples_list:
            res_yc = pd.DataFrame(['Year '+str(y), RES_Units[st,r]*RES_Nominal_Capacity[r]*RES_Inv_Specific_Cost[r]*RES_OM_Specific_Cost[r]/1e3]).T.set_index([0])    # O&M cost on installed capacity in year y (undiscounted)
            res_yc.columns = pd.MultiIndex.from_arrays([['Fixed costs'],[RES_Names[r]],['-'],['kUSD']], names=['','Component','Scenario','Unit'])
            res_yc_source = pd.concat([res_yc_source,res_yc], axis=0)
        RES_Yearly_Cost = pd.concat([RES_Yearly_Cost,res_yc_source], axis=1)
        
     "Battery bank"
     if instance.Model_Components.value == 0 or instance.Model_Components.value == 1:
         BESS_Nominal_Capacity = instance.Battery_Nominal_Capacity.extract_values()    
         BESS_Inv_Specific_Cost = instance.Battery_Specific_Investment_Cost.value
         BESS_OM_Specific_Cost = instance.Battery_Specific_OM_Cost.value
         BESS_Yearly_Cost = pd.DataFrame()
         for (y,st) in ys_tuples_list:
            bess_yc = pd.DataFrame(['Year '+str(y), BESS_Nominal_Capacity[st]*BESS_Inv_Specific_Cost*BESS_OM_Specific_Cost/1e3]).T.set_index([0])   # battery O&M cost in year y (undiscounted)
            bess_yc.columns = pd.MultiIndex.from_arrays([['Fixed costs'],['Battery bank'],['-'],['kUSD']], names=['','Component','Scenario','Unit'])
            BESS_Yearly_Cost = pd.concat([BESS_Yearly_Cost,bess_yc], axis=0)

     "Generator"
     if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:
         Generator_Types = instance.Generator_Types.value
         Generator_Nominal_Capacity = instance.Generator_Nominal_Capacity.get_values()    
         Generator_Inv_Specific_Cost = instance.Generator_Specific_Investment_Cost.extract_values()
         Generator_OM_Specific_Cost = instance.Generator_Specific_OM_Cost.extract_values()
         Generator_Yearly_Cost = pd.DataFrame()
         for g in range(1,Generator_Types+1):
            gen_yc_types =  pd.DataFrame()
            for (y,st) in ys_tuples_list:
                gen_yc = pd.DataFrame(['Year '+str(y), Generator_Nominal_Capacity[st,g]*Generator_Inv_Specific_Cost[g]*Generator_OM_Specific_Cost[g]/1e3]).T.set_index([0])   # generator O&M cost in year y (undiscounted)
                gen_yc.columns = pd.MultiIndex.from_arrays([['Fixed costs'],[Generator_Names[g]],['-'],['kUSD']], names=['','Component','Scenario','Unit'])
                gen_yc_types = pd.concat([gen_yc_types,gen_yc], axis=0)
            Generator_Yearly_Cost = pd.concat([Generator_Yearly_Cost,gen_yc_types], axis=1)

    "National Grid"     
    if instance.Grid_Connection.value == 1:
        Grid_Connection_Specific_Cost = instance.Grid_Connection_Cost.extract_values()  
        Grid_Distance = instance.Grid_Distance.extract_values() 
        Grid_OM_Specific_Cost = instance.Grid_Maintenance_Cost.extract_values()
        Grid_Connection = instance.Grid_Connection.extract_values()  
        Grid_Yearly_Fixed_Cost = pd.DataFrame()
        for (y,st) in ys_tuples_list:
            if y < instance.Year_Grid_Connection.extract_values()[None]:
                grid_yc = pd.DataFrame(['Year '+str(y), 0]).T.set_index([0])                                                                        # no cost before connection year
            else:
                grid_yc = pd.DataFrame(['Year '+str(y), Grid_Distance[None]*Grid_Connection_Specific_Cost[None]*Grid_OM_Specific_Cost[None]*Grid_Connection[None]/1e3]).T.set_index([0])   # annual grid maintenance cost
            grid_yc.columns = pd.MultiIndex.from_arrays([['Fixed costs'],['Grid'],['-'],['kUSD']], names=['','Component','Scenario','Unit'])
            Grid_Yearly_Fixed_Cost = pd.concat([Grid_Yearly_Fixed_Cost,grid_yc], axis=0)
        "Grid costs and revenues"  
        Energy_From_Grid = instance.Energy_From_Grid.get_values()    
        El_Purchased_Price = instance.Grid_Purchased_El_Price
        Grid_Yearly_Cost = pd.DataFrame()    
        for s in range(1, S+1):
            grid_s = pd.DataFrame()
            for (y, st) in ys_tuples_list:
                grid_yc = pd.DataFrame(['Year ' + str(y), sum(Energy_From_Grid[(s, y, t)] for t in range(1, P+1)) * El_Purchased_Price /1e3]).T.set_index([0])   # cost of energy imported from the grid in year y, scenario s
                grid_yc.columns = pd.MultiIndex.from_arrays([['Grid cost'], ['Grid'], [s], ['kUSD']], names=['', 'Component', 'Scenario', 'Unit'])
                grid_s = pd.concat([grid_s, grid_yc], axis=0)
            Grid_Yearly_Cost = pd.concat([Grid_Yearly_Cost, grid_s], axis=1)
       
        if instance.Grid_Connection_Type.value == 0:  # bidirectional connection: energy can also be sold back to the grid
            Energy_To_Grid = instance.Energy_To_Grid.get_values()    
            El_Sold_Price = instance.Grid_Sold_El_Price
            Grid_Yearly_Rev = pd.DataFrame()    
            for s in range(1, S+1):
                grid_s = pd.DataFrame()
                for (y, st) in ys_tuples_list:
                    grid_yc = pd.DataFrame(['Year ' + str(y), sum(Energy_To_Grid[(s, y, t)] for t in range(1, P+1)) * El_Sold_Price /1e3]).T.set_index([0])    # revenue from energy exported to the grid in year y, scenario s
                    grid_yc.columns = pd.MultiIndex.from_arrays([['Grid revenue'], ['Grid'], [s], ['kUSD']], names=['', 'Component', 'Scenario', 'Unit'])
                    grid_s = pd.concat([grid_s, grid_yc], axis=0)
                Grid_Yearly_Rev = pd.concat([Grid_Yearly_Rev, grid_s], axis=0)
   

    #%% Variable costs
    
    "Lost Load"    
    Lost_Load = instance.Lost_Load.get_values()    
    Lost_Load_Specific_Cost = instance.Lost_Load_Specific_Cost.value
    Lost_Load_Yearly_Cost = pd.DataFrame()
    for s in range(1,S+1):
        lost_load_s = pd.DataFrame()
        for (y,st) in ys_tuples_list:
            lost_load_yc = pd.DataFrame(['Year '+str(y), sum(Lost_Load[(s,y,t)] for t in range(1,P+1))*Lost_Load_Specific_Cost/1e3]).T.set_index([0])   # cost of unserved demand in year y, scenario s
            lost_load_yc.columns = pd.MultiIndex.from_arrays([['Lost load cost'],['System'],[s],['kUSD']], names=['','Component','Scenario','Unit'])
            lost_load_s = pd.concat([lost_load_s,lost_load_yc], axis=0)
        Lost_Load_Yearly_Cost = pd.concat([Lost_Load_Yearly_Cost,lost_load_s], axis=1)

    "BESS Replacement Cost"
    if instance.Model_Components.value == 0 or instance.Model_Components.value == 1:
        BESS_Inflow = instance.Battery_Inflow.get_values()    
        BESS_Outflow = instance.Battery_Outflow.get_values()    
        BESS_Unit_Repl_Cost = instance.Unitary_Battery_Replacement_Cost.value
        BESS_Replacement_Yearly_Cost = pd.DataFrame()    
        for s in range(1,S+1):
            Battery_cost_in = [0 for y in range(1,Y+1)]
            Battery_cost_out = [0 for y in range(1,Y+1)]
            Battery_Yearly_cost = [0 for y in range(1,Y+1)]    
            for y in range(1,Y+1):    
                Battery_cost_in[y-1] = sum(BESS_Inflow[s,y,t]*BESS_Unit_Repl_Cost for t in range(1,P+1))    # cost attributed to battery charging throughput (wear from cycling)
                Battery_cost_out[y-1] = sum(BESS_Outflow[s,y,t]*BESS_Unit_Repl_Cost for t in range(1,P+1))   # cost attributed to battery discharging throughput
                Battery_Yearly_cost[y-1] = Battery_cost_in[y-1] + Battery_cost_out[y-1]                      # total replacement cost for year y
            Battery_Yearly_cost = pd.DataFrame(Battery_Yearly_cost)/1e3
            Battery_Yearly_cost.index = Lost_Load_Yearly_Cost.index
            Battery_Yearly_cost.columns = pd.MultiIndex.from_arrays([['Replacement cost'],['Battery bank'],[s],['kUSD']], names=['','Component','Scenario','Unit']) 
            BESS_Replacement_Yearly_Cost = pd.concat([BESS_Replacement_Yearly_Cost, Battery_Yearly_cost], axis=1)
    
    "Fuel cost"
    if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:
        if instance.MILP_Formulation.value:  # MILP: account separately for full-load, partial-load and start-up costs
         Generator_Energy_Partial = instance.Generator_Energy_Partial.get_values()
         Generator_Energy_Total = instance.Generator_Energy_Total.get_values()
         Generator_Full = instance.Generator_Full.get_values()
         Generator_Partial = instance.Generator_Partial.get_values()
         Generator_Marginal_Cost = instance.Generator_Marginal_Cost.extract_values()
         Generator_Marginal_Cost_1 = instance.Generator_Marginal_Cost_1.extract_values()
         Generator_Marginal_Cost_milp = instance.Generator_Marginal_Cost_milp.extract_values()
         Generator_Marginal_Cost_milp_1 = instance.Generator_Marginal_Cost_milp_1.extract_values()
         Generator_Nominal_Capacity_milp = instance.Generator_Nominal_Capacity_milp.extract_values()
         Generator_Start_Cost = instance.Generator_Start_Cost.extract_values()
         Generator_Start_Cost_1 = instance.Generator_Start_Cost_1.extract_values()
         Fuel_Cost_Yearly_Cost = pd.DataFrame()
         for s in range(1,S+1):
            fuel_s = pd.DataFrame()
            for g in range(1,Generator_Types+1):
                fuel_yc_types = pd.DataFrame()
                for (y,st) in ys_tuples_list:
                    if instance.Generator_Partial_Load.value == 1 and instance.Fuel_Specific_Cost_Calculation.value == 1:
                       fuel_yc = pd.DataFrame(['Year '+str(y), (sum(Generator_Full[(s,y,g,t)] for t in range(1,P+1))*Generator_Marginal_Cost[g,y]*Generator_Nominal_Capacity_milp[g]/1e3) + (sum(Generator_Energy_Partial[(s,y,g,t)] for t in range(1,P+1))*Generator_Marginal_Cost_milp[g,y]/1e3) + (sum(Generator_Partial[(s,y,g,t)] for t in range(1,P+1))*Generator_Start_Cost[g,y])/1e3]).T.set_index([0])  # full-load + partial-load + start-up cost, year-specific marginal cost
                    elif instance.Generator_Partial_Load.value == 1 and instance.Fuel_Specific_Cost_Calculation.value == 0:
                        fuel_yc = pd.DataFrame(['Year '+str(y), (sum(Generator_Full[(s,y,g,t)] for t in range(1,P+1))*Generator_Marginal_Cost_1[g]*Generator_Nominal_Capacity_milp[g]/1e3) + (sum(Generator_Energy_Partial[(s,y,g,t)] for t in range(1,P+1))*Generator_Marginal_Cost_milp_1[g]/1e3) + (sum(Generator_Partial[(s,y,g,t)] for t in range(1,P+1))*Generator_Start_Cost_1[g])/1e3]).T.set_index([0])  # same but with a fixed (non year-specific) marginal cost
                    elif instance.Generator_Partial_Load.value == 0 and instance.Fuel_Specific_Cost_Calculation.value == 1:
                       fuel_yc = pd.DataFrame(['Year '+str(y), sum(Generator_Energy_Total[(s,y,g,t)] for t in range(1,P+1))*Generator_Marginal_Cost[g,y]/1e3]).T.set_index([0])   # no partial load: single marginal cost, year-specific
                    else:
                        fuel_yc = pd.DataFrame(['Year '+str(y), sum(Generator_Energy_Total[(s,y,g,t)] for t in range(1,P+1))*Generator_Marginal_Cost_1[g]/1e3]).T.set_index([0])   # no partial load: single fixed marginal cost
                    fuel_yc.columns = pd.MultiIndex.from_arrays([['Fuel cost'],[Fuel_Names[g]],[s],['kUSD']], names=['','Component','Scenario','Unit'])
                    fuel_yc_types = pd.concat([fuel_yc_types,fuel_yc], axis=0)
                fuel_s = pd.concat([fuel_s,fuel_yc_types], axis=0)            
            Fuel_Cost_Yearly_Cost = pd.concat([Fuel_Cost_Yearly_Cost,fuel_s], axis=1).fillna(0)
         Fuel_Cost_Yearly_Cost = Fuel_Cost_Yearly_Cost.groupby(level=[0], sort=False).sum()
        else:  # LP: single production variable, no partial-load distinction
         Generator_Energy_Production = instance.Generator_Energy_Production.get_values()
         Generator_Marginal_Cost = instance.Generator_Marginal_Cost.extract_values()
         Generator_Marginal_Cost_1 = instance.Generator_Marginal_Cost_1.extract_values()
         Fuel_Cost_Yearly_Cost = pd.DataFrame()
         for s in range(1,S+1):
            fuel_s = pd.DataFrame()
            for g in range(1,Generator_Types+1):
                fuel_yc_types = pd.DataFrame()
                for (y,st) in ys_tuples_list:
                    if instance.Fuel_Specific_Cost_Calculation.value == 1:
                     fuel_yc = pd.DataFrame(['Year '+str(y), sum(Generator_Energy_Production[(s,y,g,t)] for t in range(1,P+1))*Generator_Marginal_Cost[g,y]/1e3]).T.set_index([0])   # year-specific marginal fuel cost
                    else: fuel_yc = pd.DataFrame(['Year '+str(y), sum(Generator_Energy_Production[(s,y,g,t)] for t in range(1,P+1))*Generator_Marginal_Cost_1[g]/1e3]).T.set_index([0])   # fixed marginal fuel cost
                    fuel_yc.columns = pd.MultiIndex.from_arrays([['Fuel cost'],[Fuel_Names[g]],[s],['kUSD']], names=['','Component','Scenario','Unit'])
                    fuel_yc_types = pd.concat([fuel_yc_types,fuel_yc], axis=0)
                fuel_s = pd.concat([fuel_s,fuel_yc_types], axis=0)            
            Fuel_Cost_Yearly_Cost = pd.concat([Fuel_Cost_Yearly_Cost,fuel_s], axis=1).fillna(0)
         Fuel_Cost_Yearly_Cost = Fuel_Cost_Yearly_Cost.groupby(level=[0], sort=False).sum()
    #%% Concatenating
    if instance.Model_Components.value == 0:
        if instance.Grid_Connection.value == 1 and instance.Grid_Connection_Type == 0:
            YearlyCost = pd.concat([round(RES_Yearly_Cost.astype(float),2),
                                    round(BESS_Yearly_Cost.astype(float),2),
                                    round(Generator_Yearly_Cost.astype(float),2),
                                    round(Grid_Yearly_Fixed_Cost.astype(float),2), 
                                    round(Lost_Load_Yearly_Cost.astype(float),2),
                                    round(BESS_Replacement_Yearly_Cost.astype(float),2),
                                    round(Fuel_Cost_Yearly_Cost.astype(float),2),
                                    round(Grid_Yearly_Cost.astype(float),2),
                                    round(Grid_Yearly_Rev.astype(float),2)], axis=1) 
        if instance.Grid_Connection.value == 1 and instance.Grid_Connection_Type == 1:
            YearlyCost = pd.concat([round(RES_Yearly_Cost.astype(float),2),
                                    round(BESS_Yearly_Cost.astype(float),2),
                                    round(Generator_Yearly_Cost.astype(float),2),
                                    round(Grid_Yearly_Fixed_Cost.astype(float),2), 
                                    round(Lost_Load_Yearly_Cost.astype(float),2),
                                    round(BESS_Replacement_Yearly_Cost.astype(float),2),
                                    round(Fuel_Cost_Yearly_Cost.astype(float),2),
                                    round(Grid_Yearly_Cost.astype(float),2)], axis=1)
        if instance.Grid_Connection.value == 0:
            YearlyCost = pd.concat([round(RES_Yearly_Cost.astype(float),2),
                                    round(BESS_Yearly_Cost.astype(float),2),
                                    round(Generator_Yearly_Cost.astype(float),2),
                                    round(Lost_Load_Yearly_Cost.astype(float),2),
                                    round(BESS_Replacement_Yearly_Cost.astype(float),2),
                                    round(Fuel_Cost_Yearly_Cost.astype(float),2)], axis=1) 
            
    if instance.Model_Components.value == 1:
        if instance.Grid_Connection.value == 1 and instance.Grid_Connection_Type == 0:
            YearlyCost = pd.concat([round(RES_Yearly_Cost.astype(float),2),
                                    round(BESS_Yearly_Cost.astype(float),2),
                                    round(Grid_Yearly_Fixed_Cost.astype(float),2), 
                                    round(Lost_Load_Yearly_Cost.astype(float),2),
                                    round(BESS_Replacement_Yearly_Cost.astype(float),2),
                                    round(Grid_Yearly_Cost.astype(float),2), 
                                    round(Grid_Yearly_Rev.astype(float),2)], axis=1) 
        if instance.Grid_Connection.value == 1 and instance.Grid_Connection_Type == 1:
            YearlyCost = pd.concat([round(RES_Yearly_Cost.astype(float),2),
                                    round(BESS_Yearly_Cost.astype(float),2),
                                    round(Grid_Yearly_Fixed_Cost.astype(float),2), 
                                    round(Lost_Load_Yearly_Cost.astype(float),2),
                                    round(BESS_Replacement_Yearly_Cost.astype(float),2),
                                    round(Fuel_Cost_Yearly_Cost.astype(float),2),
                                    round(Grid_Yearly_Cost.astype(float),2)], axis=1)
        if instance.Grid_Connection.value == 0:
            YearlyCost = pd.concat([round(RES_Yearly_Cost.astype(float),2),
                                    round(BESS_Yearly_Cost.astype(float),2),
                                    round(Lost_Load_Yearly_Cost.astype(float),2),
                                    round(BESS_Replacement_Yearly_Cost.astype(float),2)], axis=1)
            
    if instance.Model_Components.value == 2:
        if instance.Grid_Connection.value == 1 and instance.Grid_Connection_Type == 0:
            YearlyCost = pd.concat([round(RES_Yearly_Cost.astype(float),2),
                                    round(Generator_Yearly_Cost.astype(float),2),
                                    round(Grid_Yearly_Fixed_Cost.astype(float),2), 
                                    round(Lost_Load_Yearly_Cost.astype(float),2),
                                    round(Fuel_Cost_Yearly_Cost.astype(float),2),
                                    round(Grid_Yearly_Cost.astype(float),2), 
                                    round(Grid_Yearly_Rev.astype(float),2)], axis=1) 
        if instance.Grid_Connection.value == 1 and instance.Grid_Connection_Type == 1:
            YearlyCost = pd.concat([round(RES_Yearly_Cost.astype(float),2),
                                    round(Generator_Yearly_Cost.astype(float),2),
                                    round(Grid_Yearly_Fixed_Cost.astype(float),2), 
                                    round(Lost_Load_Yearly_Cost.astype(float),2),
                                    round(Fuel_Cost_Yearly_Cost.astype(float),2),
                                    round(Grid_Yearly_Cost.astype(float),2)], axis=1)
        if instance.Grid_Connection.value == 0:
            YearlyCost = pd.concat([round(RES_Yearly_Cost.astype(float),2),
                                    round(Generator_Yearly_Cost.astype(float),2),
                                    round(Lost_Load_Yearly_Cost.astype(float),2),
                                    round(Fuel_Cost_Yearly_Cost.astype(float),2)], axis=1)  

    return YearlyCost

#%% Yearly energy parameters
def YearlyEnergyParams(instance, TimeSeries):
    # Aggregates the hourly TimeSeries into yearly percentage shares (generator share, RES load,
    # curtailment share, renewable penetration, battery usage, grid usage), weighted across scenarios.
    
    "Importing parameters"
    S  = int(instance.Scenarios.extract_values()[None])
    P  = int(instance.Periods.extract_values()[None])
    Y  = int(instance.Years.extract_values()[None])
    ST = int(instance.Steps_Number.extract_values()[None])
    R  = int(instance.RES_Sources.extract_values()[None])
    G  = int(instance.Generator_Types.extract_values()[None])

    RES_Names = instance.RES_Names.extract_values()
    Generator_Names = instance.Generator_Names.extract_values()
    Fuel_Names = instance.Fuel_Names.extract_values()
    idx = pd.IndexSlice                                                # convenience alias for building MultiIndex column slices below
    
    #%% Data preparation
    gen_load  = pd.DataFrame()
    res_load  = pd.DataFrame()
    curt_load = pd.DataFrame()
    res_pen   = pd.DataFrame()
    battery_usage = pd.DataFrame()
    grid_usage = pd.DataFrame()   
    for y in range(1,Y+1):
        demand = 0
        curtailment = 0
        renewables  = 0
        generators  = 0
        battery_out = 0
        grid_in = 0   
        grid_out = 0
        for s in range(1,S+1):                                        # weight each scenario's contribution by its probability before summing across scenarios
            demand += TimeSeries[s][y].loc[:,idx['Scenario '+str(s),'Electric Demand',:,:]].sum().sum()*instance.Scenario_Weight.extract_values()[s]
            curtailment += TimeSeries[s][y].loc[:,idx['Scenario '+str(s),'Curtailment',:,:]].sum().sum()*instance.Scenario_Weight.extract_values()[s]
            renewables  += TimeSeries[s][y].loc[:,idx['Scenario '+str(s),'RES Production',:,:]].sum().sum()*instance.Scenario_Weight.extract_values()[s]
            if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:
                generators  += TimeSeries[s][y].loc[:,idx['Scenario '+str(s),'Generator Production',:,:]].sum().sum()*instance.Scenario_Weight.extract_values()[s]
            if instance.Model_Components.value == 0 or instance.Model_Components.value == 1:
                battery_out += TimeSeries[s][y].loc[:,idx['Scenario '+str(s),'Battery Discharge',:,:]].sum().sum()*instance.Scenario_Weight.extract_values()[s]
            if instance.Grid_Connection.value == 1:
                grid_in += TimeSeries[s][y].loc[:,idx['Scenario '+str(s),'Electricity from grid',:,:]].sum().sum()*instance.Scenario_Weight.extract_values()[s]   
                if instance.Grid_Connection_Type.value == 0:
                    grid_out += TimeSeries[s][y].loc[:,idx['Scenario '+str(s),'Electricity to grid',:,:]].sum().sum()*instance.Scenario_Weight.extract_values()[s]
                
        if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:
            gen_load  = pd.concat([gen_load, pd.DataFrame(['Year '+str(y), generators/demand]).T.set_index([0])], axis=0)              # generator share of demand
        res_load  = pd.concat([res_load, pd.DataFrame(['Year '+str(y), (renewables-curtailment)/demand]).T.set_index([0])], axis=0)     # net (used) RES share of demand
        curt_load = pd.concat([curt_load, pd.DataFrame(['Year '+str(y), curtailment/(generators+renewables)]).T.set_index([0])], axis=0) # curtailment as a share of total production
        if instance.Grid_Connection.value == 1:
            grid_usage = pd.concat([grid_usage, pd.DataFrame(['Year '+str(y), grid_in/demand]).T.set_index([0])], axis=0)    # grid import share of demand
        res_pen   = pd.concat([res_pen, pd.DataFrame(['Year '+str(y), renewables/(renewables+generators+grid_in)]).T.set_index([0])], axis=0)  # renewable penetration among all supply sources
        if instance.Model_Components.value == 0 or instance.Model_Components.value == 1:
            battery_usage = pd.concat([battery_usage, pd.DataFrame(['Year '+str(y), battery_out/demand]).T.set_index([0])], axis=0)  # battery discharge share of demand
    gen_load  = round(gen_load.astype(float)*100,2)      # convert fractions to percentages
    res_load  = round(res_load.astype(float)*100,2)
    res_pen   = round(res_pen.astype(float)*100,2)
    curt_load = round(curt_load.astype(float)*100,2)
    battery_usage = round(battery_usage.astype(float)*100,2)
    grid_usage = round(grid_usage.astype(float)*100,2)   

    if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:
        gen_load.columns  = pd.MultiIndex.from_arrays([['Generators share'],['%']], names=['',' '])
    curt_load.columns = pd.MultiIndex.from_arrays([['Curtailment share'],['%']], names=['',' '])
    res_pen.columns   = pd.MultiIndex.from_arrays([['Renewables penetration'],['%']], names=['',' '])
    if instance.Model_Components.value == 0 or instance.Model_Components.value == 1:
        battery_usage.columns = pd.MultiIndex.from_arrays([['Battery usage'],['%']], names=['',' '])
    if instance.Grid_Connection.value == 1:
        grid_usage.columns = pd.MultiIndex.from_arrays([['Grid usage'],['%']], names=['',' ']) 
    
    #%% Concatenating
    if instance.Model_Components.value == 0:  # RES + Battery + Generator
        if instance.Grid_Connection.value == 1:
            YearlyEnergyParams = pd.concat([gen_load,
                                            res_pen,
                                            curt_load,
                                            battery_usage,
                                            grid_usage], axis=1)
        if instance.Grid_Connection.value == 0:
            YearlyEnergyParams = pd.concat([gen_load,
                                            res_pen,
                                            curt_load,
                                            battery_usage], axis=1)          
    if instance.Model_Components.value == 1:
        if instance.Grid_Connection.value == 1:
            YearlyEnergyParams = pd.concat([res_pen,
                                            curt_load,
                                            battery_usage,
                                            grid_usage], axis=1)
        if instance.Grid_Connection.value == 0:
            YearlyEnergyParams = pd.concat([res_pen,
                                            curt_load,
                                            battery_usage], axis=1)
    if instance.Model_Components.value == 2:
        if instance.Grid_Connection.value == 1:
            YearlyEnergyParams = pd.concat([gen_load,
                                            res_pen,
                                            curt_load,
                                            grid_usage], axis=1)
        if instance.Grid_Connection.value == 0:
            YearlyEnergyParams = pd.concat([gen_load,
                                            res_pen,
                                            curt_load], axis=1)
            
    return YearlyEnergyParams, res_pen

#%% Yearly energy parameters
def YearlyEnergyParamsSC(instance, TimeSeries):
    # Same energy-share metrics as YearlyEnergyParams, but kept per-scenario (not weighted/aggregated).
    
    "Importing parameters"
    S  = int(instance.Scenarios.extract_values()[None])
    P  = int(instance.Periods.extract_values()[None])
    Y  = int(instance.Years.extract_values()[None])
    ST = int(instance.Steps_Number.extract_values()[None])
    R  = int(instance.RES_Sources.extract_values()[None])
    G  = int(instance.Generator_Types.extract_values()[None])

    RES_Names = instance.RES_Names.extract_values()
    Generator_Names = instance.Generator_Names.extract_values()
    Fuel_Names = instance.Fuel_Names.extract_values()
    idx = pd.IndexSlice
    
    "Generating years-steps tuples list"
    steps = [i for i in range(1, ST+1)]
    
    years_steps_list = [1 for i in range(1, ST+1)]
    s_dur = instance.Step_Duration.value
    for i in range(1, ST): 
        years_steps_list[i] = years_steps_list[i-1] + s_dur
    ys_tuples_list = [[] for i in range(1, Y+1)]
    for y in range(1, Y+1):  
        if len(years_steps_list) == 1:
            ys_tuples_list[y-1] = (y,1)
        else:
            for i in range(len(years_steps_list)-1):
                if y >= years_steps_list[i] and y < years_steps_list[i+1]:
                    ys_tuples_list[y-1] = (y, steps[i])       
                elif y >= years_steps_list[-1]:
                    ys_tuples_list[y-1] = (y, len(steps))

#%% Data preparation
     
    if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:  # generator share of demand, per scenario
        if instance.MILP_Formulation.value:
         "Generator"
         Generator_Names = instance.Generator_Names.extract_values()
         Generator_Energy_Total = instance.Generator_Energy_Total.get_values()
         Energy_Demand = instance.Energy_Demand.extract_values()
         Generator_Types = instance.Generator_Types.value
         gen_load_sc  = pd.DataFrame()
         for s in range(1,S+1):
            gen_load = pd.DataFrame()
            for g in range(1,Generator_Types+1):
                gen_load_types = pd.DataFrame()
                for (y,st) in ys_tuples_list:
                    gen_load_scenarios = pd.DataFrame(['Year '+str(y), sum(Generator_Energy_Total[(s,y,g,t)] for t in range(1,P+1))/sum(Energy_Demand[s,y,t] for t in range(1,P+1))]).T.set_index([0])   # generator g's energy / total demand, for scenario s, year y
                    gen_load_scenarios.columns  = pd.MultiIndex.from_arrays([['Generator share'],[Generator_Names[g]],[s],['%']], names=['','Component','Scenario',' ']) 
                    gen_load_types = pd.concat([gen_load_types, gen_load_scenarios], axis=0)
                gen_load = pd.concat([gen_load, gen_load_types], axis=0)
            gen_load_sc = pd.concat([gen_load_sc, gen_load], axis=1).fillna(0)
         gen_load_sc = gen_load_sc.groupby(level=[0], sort=False).sum()
        else:
         Generator_Names = instance.Generator_Names.extract_values()
         Generator_Energy_Production = instance.Generator_Energy_Production.get_values()
         Energy_Demand = instance.Energy_Demand.extract_values()
         Generator_Types = instance.Generator_Types.value
         gen_load_sc  = pd.DataFrame()
         for s in range(1,S+1):
            gen_load = pd.DataFrame()
            for g in range(1,Generator_Types+1):
                gen_load_types = pd.DataFrame()
                for (y,st) in ys_tuples_list:
                    gen_load_scenarios = pd.DataFrame(['Year '+str(y), sum(Generator_Energy_Production[(s,y,g,t)] for t in range(1,P+1))/sum(Energy_Demand[s,y,t] for t in range(1,P+1))]).T.set_index([0])   # generator g's energy / total demand, for scenario s, year y
                    gen_load_scenarios.columns  = pd.MultiIndex.from_arrays([['Generator share'],[Generator_Names[g]],[s],['%']], names=['','Component','Scenario',' ']) 
                    gen_load_types = pd.concat([gen_load_types, gen_load_scenarios], axis=0)
                gen_load = pd.concat([gen_load, gen_load_types], axis=0)
            gen_load_sc = pd.concat([gen_load_sc, gen_load], axis=1).fillna(0)
         gen_load_sc = gen_load_sc.groupby(level=[0], sort=False).sum()
    RES_Names = instance.RES_Names.extract_values()
    RES_Energy_Production = instance.RES_Energy_Production.get_values()
    Curtailment = instance.Energy_Curtailment.get_values()
    Energy_Demand = instance.Energy_Demand.extract_values()
    RES_Sources = instance.RES_Sources.value
    res_load_sc  = pd.DataFrame()
    for s in range(1,S+1):  # net (production minus curtailment) RES share of demand, per source, per scenario
        res_load = pd.DataFrame()
        for r in range(1,RES_Sources+1):
            res_load_types = pd.DataFrame()
            for (y,st) in ys_tuples_list:
                res_load_scenarios = pd.DataFrame(['Year '+str(y), sum(RES_Energy_Production[(s,y,r,t)]-Curtailment[(s,y,t)] for t in range(1,P+1))/sum(Energy_Demand[s,y,t] for t in range(1,P+1))]).T.set_index([0])
                res_load_scenarios.columns  = pd.MultiIndex.from_arrays([['RES load'],[RES_Names[r]],[s],['%']], names=['','Component','Scenario',' ']) 
                res_load_types = pd.concat([res_load_types, res_load_scenarios], axis=0)
            res_load = pd.concat([res_load, res_load_types], axis=0)
        res_load_sc = pd.concat([res_load_sc, res_load], axis=1).fillna(0)
    res_load_sc = res_load_sc.groupby(level=[0], sort=False).sum()
    if instance.MILP_Formulation.value:  # renewable penetration = RES production / total production (RES+generators+grid), per source, per scenario
     RES_Names = instance.RES_Names.extract_values()
     RES_Energy_Production = instance.RES_Energy_Production.get_values()
     Generator_Energy_Total = instance.Generator_Energy_Total.get_values()
     Electricity_From_Grid = instance.Energy_From_Grid.get_values() 
     res_pen_sc  = pd.DataFrame()
     for s in range(1,S+1):
        res_pen = pd.DataFrame()
        for r in range(1,RES_Sources+1):
            res_pen_types = pd.DataFrame()
            for (y,st) in ys_tuples_list:
                
                if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:
                    if instance.Grid_Connection.value == 1:
                        res_pen_scenarios = pd.DataFrame(['Year '+str(y), (sum(RES_Energy_Production[(s,y,r,t)] for t in range(1,P+1)))/(sum(sum(RES_Energy_Production[(s,y,r,t)] for r in range(1,R+1)) + sum(Generator_Energy_Total[(s,y,g,t)] for g in range(1,G+1)) + Electricity_From_Grid[(s,y,t)] for t in range(1,P+1)))]).T.set_index([0])
                    if instance.Grid_Connection.value == 0:
                        res_pen_scenarios = pd.DataFrame(['Year '+str(y), (sum(RES_Energy_Production[(s,y,r,t)] for t in range(1,P+1)))/(sum(sum(RES_Energy_Production[(s,y,r,t)] for r in range(1,R+1)) + sum(Generator_Energy_Total[(s,y,g,t)] for g in range(1,G+1)) for t in range(1,P+1)))]).T.set_index([0])
                   
                if instance.Model_Components.value == 1:
                    if instance.Grid_Connection.value == 1:
                        res_pen_scenarios = pd.DataFrame(['Year '+str(y), (sum(RES_Energy_Production[(s,y,r,t)] for t in range(1,P+1)))/(sum(sum(RES_Energy_Production[(s,y,r,t)] for r in range(1,R+1)) + Electricity_From_Grid[(s,y,t)] for t in range(1,P+1)))]).T.set_index([0])
                    if instance.Grid_Connection.value == 0:
                        res_pen_scenarios = pd.DataFrame(['Year '+str(y), (sum(RES_Energy_Production[(s,y,r,t)] for t in range(1,P+1)))/(sum(sum(RES_Energy_Production[(s,y,r,t)] for r in range(1,R+1)) for t in range(1,P+1)))]).T.set_index([0])
     
                res_pen_scenarios.columns  = pd.MultiIndex.from_arrays([['Renewable penetration'],[RES_Names[r]],[s],['%']], names=['','Component','Scenario',' ']) 
                res_pen_types = pd.concat([res_pen_types, res_pen_scenarios], axis=0)
            res_pen = pd.concat([res_pen, res_pen_types], axis=0)
        res_pen_sc = pd.concat([res_pen_sc, res_pen], axis=1).fillna(0)
     res_pen_sc = res_pen_sc.groupby(level=[0], sort=False).sum()
    else:  # same renewable-penetration logic but using the LP generator production variable
     RES_Names = instance.RES_Names.extract_values()
     RES_Energy_Production = instance.RES_Energy_Production.get_values()
     Generator_Energy_Production = instance.Generator_Energy_Production.get_values()
     Electricity_From_Grid = instance.Energy_From_Grid.get_values() 
     res_pen_sc  = pd.DataFrame()
     for s in range(1,S+1):
        res_pen = pd.DataFrame()
        for r in range(1,RES_Sources+1):
            res_pen_types = pd.DataFrame()
            for (y,st) in ys_tuples_list:
                
                if instance.Model_Components.value == 0 or instance.Model_Components.value == 2:
                    if instance.Grid_Connection.value == 1:
                            res_pen_scenarios = pd.DataFrame(['Year '+str(y), (sum(RES_Energy_Production[(s,y,r,t)] for t in range(1,P+1)))/(sum(sum(RES_Energy_Production[(s,y,r,t)] for r in range(1,R+1)) + sum(Generator_Energy_Production[(s,y,g,t)] for g in range(1,G+1)) + Electricity_From_Grid[(s,y,t)] for t in range(1,P+1)))]).T.set_index([0])
                    if instance.Grid_Connection.value == 0:
                            res_pen_scenarios = pd.DataFrame(['Year '+str(y), (sum(RES_Energy_Production[(s,y,r,t)] for t in range(1,P+1)))/(sum(sum(RES_Energy_Production[(s,y,r,t)] for r in range(1,R+1)) + sum(Generator_Energy_Production[(s,y,g,t)] for g in range(1,G+1)) for t in range(1,P+1)))]).T.set_index([0])
                if instance.Model_Components.value == 1:
                    if instance.Grid_Connection.value == 1:
                            res_pen_scenarios = pd.DataFrame(['Year '+str(y), (sum(RES_Energy_Production[(s,y,r,t)] for t in range(1,P+1)))/(sum(sum(RES_Energy_Production[(s,y,r,t)] for r in range(1,R+1)) + Electricity_From_Grid[(s,y,t)] for t in range(1,P+1)))]).T.set_index([0])
                    if instance.Grid_Connection.value == 0:
                            res_pen_scenarios = pd.DataFrame(['Year '+str(y), (sum(RES_Energy_Production[(s,y,r,t)] for t in range(1,P+1)))/(sum(sum(RES_Energy_Production[(s,y,r,t)] for r in range(1,R+1)) for t in range(1,P+1)))]).T.set_index([0])
                    
                res_pen_scenarios.columns  = pd.MultiIndex.from_arrays([['Renewable penetration'],[RES_Names[r]],[s],['%']], names=['','Component','Scenario',' ']) 
                res_pen_types = pd.concat([res_pen_types, res_pen_scenarios], axis=0)
            res_pen = pd.concat([res_pen, res_pen_types], axis=0)
        res_pen_sc = pd.concat([res_pen_sc, res_pen], axis=1).fillna(0)
     res_pen_sc = res_pen_sc.groupby(level=[0], sort=False).sum()
    curt_load_sc = pd.DataFrame()
    for s in range(1, S+1):  # curtailment as a share of total production, per scenario
        curt_sc = pd.DataFrame()
        for (y, st) in ys_tuples_list:
            if instance.MILP_Formulation.value:
                total_energy_production = sum(
                    sum(RES_Energy_Production[(s, y, r, t)] if RES_Energy_Production.get((s, y, r, t)) is not None else 0 for r in range(1, R+1)) +
                    sum(Generator_Energy_Total[(s, y, g, t)] if Generator_Energy_Total.get((s, y, g, t)) is not None else 0 for g in range(1, G+1))
                    for t in range(1, P+1)
                )
            else:
                total_energy_production = sum(
                    sum(RES_Energy_Production[(s, y, r, t)] if RES_Energy_Production.get((s, y, r, t)) is not None else 0 for r in range(1, R+1)) +
                    sum(Generator_Energy_Production[(s, y, g, t)] if Generator_Energy_Production.get((s, y, g, t)) is not None else 0 for g in range(1, G+1))
                    for t in range(1, P+1)
                )
                
            if total_energy_production > 0:
                curt_load = pd.DataFrame(
                    ['Year ' + str(y), sum(Curtailment.get((s, y, t), 0) for t in range(1, P+1)) / total_energy_production]
                ).T.set_index([0])
            else:
                curt_load = pd.DataFrame(['Year ' + str(y), 0]).T.set_index([0])  # Set a default value (e.g., 0) when there's no production

            curt_load.columns = pd.MultiIndex.from_arrays([['Curtailment share'], ['-'], [s], ['%']], names=['', 'Component', 'Scenario', ' '])
            curt_sc = pd.concat([curt_sc, curt_load], axis=0)
        curt_load_sc = pd.concat([curt_load_sc, curt_sc], axis=1).fillna(0)

    
    if instance.Model_Components.value == 0 or instance.Model_Components.value == 1:  # battery discharge share of demand, per scenario
        BESS_Outflow = instance.Battery_Outflow.get_values()    
        battery_usage_sc = pd.DataFrame()
        for s in range(1,S+1):
            battery_sc = pd.DataFrame()
            for (y,st) in ys_tuples_list:
                batt_sc = pd.DataFrame(['Year '+str(y), sum(BESS_Outflow[s,y,t] for t in range(1,P+1))/sum(Energy_Demand[s,y,t] for t in range(1,P+1))]).T.set_index([0]) 
                batt_sc.columns  = pd.MultiIndex.from_arrays([['Battery usage'],['Battery bank'],[s],['%']], names=['','Component','Scenario',' '])
                battery_sc = pd.concat([battery_sc, batt_sc], axis=0)
            battery_usage_sc = pd.concat([battery_usage_sc, battery_sc], axis=1).fillna(0)
    
    if instance.Grid_Connection.value == 1:  # grid import share of demand, per scenario
        Electricity_From_Grid = instance.Energy_From_Grid.get_values() 
        grid_usage_sc = pd.DataFrame()
        for s in range(1,S+1):
            grid_el_sc = pd.DataFrame()
            for (y,st) in ys_tuples_list:
                grid_sc = pd.DataFrame(['Year '+str(y), sum(Electricity_From_Grid[s,y,t] for t in range(1,P+1))/sum(Energy_Demand[s,y,t] for t in range(1,P+1))]).T.set_index([0]) 
                grid_sc.columns  = pd.MultiIndex.from_arrays([['Grid usage'],['Grid'],[s],['%']], names=['','Component','Scenario',' '])
                grid_el_sc = pd.concat([grid_el_sc, grid_sc], axis=0)
            grid_usage_sc = pd.concat([grid_usage_sc, grid_el_sc], axis=1).fillna(0)
        
    #%% Concatenating
    if instance.Model_Components.value == 0:
        if instance.Grid_Connection.value == 1:
            YearlyEnergyParamsSC = pd.concat([round(gen_load_sc.astype(float)*100,2),
                                              round(res_pen_sc.astype(float)*100,2),
                                              round(curt_load_sc.astype(float)*100,2),
                                              round(battery_usage_sc.astype(float)*100,2),
                                              round(grid_usage_sc.astype(float)*100,2)], axis=1)
        if instance.Grid_Connection.value == 0:
            YearlyEnergyParamsSC = pd.concat([round(gen_load_sc.astype(float)*100,2),
                                              round(res_pen_sc.astype(float)*100,2),
                                              round(curt_load_sc.astype(float)*100,2),
                                              round(battery_usage_sc.astype(float)*100,2)], axis=1)       
    if instance.Model_Components.value == 1:
        if instance.Grid_Connection.value == 1:
            YearlyEnergyParamsSC = pd.concat([round(res_pen_sc.astype(float)*100,2),
                                              round(curt_load_sc.astype(float)*100,2),
                                              round(battery_usage_sc.astype(float)*100,2),
                                              round(grid_usage_sc.astype(float)*100,2)], axis=1)
        if instance.Grid_Connection.value == 0:
            YearlyEnergyParamsSC = pd.concat([round(res_pen_sc.astype(float)*100,2),
                                              round(curt_load_sc.astype(float)*100,2),
                                              round(battery_usage_sc.astype(float)*100,2)], axis=1)
    if instance.Model_Components.value == 2:
        if instance.Grid_Connection.value == 1:
            YearlyEnergyParamsSC = pd.concat([round(gen_load_sc.astype(float)*100,2),
                                              round(res_pen_sc.astype(float)*100,2),
                                              round(curt_load_sc.astype(float)*100,2),
                                              round(grid_usage_sc.astype(float)*100,2)], axis=1)
        if instance.Grid_Connection.value == 0:
            YearlyEnergyParamsSC = pd.concat([round(gen_load_sc.astype(float)*100,2),
                                              round(res_pen_sc.astype(float)*100,2),
                                              round(curt_load_sc.astype(float)*100,2)], axis=1)

    
    return YearlyEnergyParamsSC, res_pen_sc    

def EnergySystemLandUse(instance):
    # Builds the land-use table (m2 occupied by RES installations per investment step).

    #%% Importing parameters
    S  = int(instance.Scenarios.extract_values()[None])
    P  = int(instance.Periods.extract_values()[None])
    Y  = int(instance.Years.extract_values()[None])
    ST = int(instance.Steps_Number.extract_values()[None])
    R  = int(instance.RES_Sources.extract_values()[None])
    G  = int(instance.Generator_Types.extract_values()[None])

    RES_Names = instance.RES_Names.extract_values()
    
    upgrade_years_list = [1 for i in range(ST)]
    s_dur = instance.Step_Duration.value
    for i in range(1, ST): 
        upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    
    yu_tuples_list = [0 for i in range(1,Y+1)]    
    if ST == 1:    
        for y in range(1,Y+1):            
            yu_tuples_list[y-1] = (y, 1)    
    else:        
        for y in range(1,Y+1):            
            for i in range(len(upgrade_years_list)-1):
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:
                    yu_tuples_list[y-1] = (y, [st for st in range(ST)][i+1])                
                elif y >= upgrade_years_list[-1]:
                    yu_tuples_list[y-1] = (y, ST)   
    tup_list = [[] for i in range(ST-1)]  
    for i in range(0,ST-1):
        tup_list[i] = yu_tuples_list[s_dur*i + s_dur]
        
    #%%
    if instance.MILP_Formulation.value:  # unit-based (integer) capacity expansion
     "Renewable sources"
     RES_Units_milp = instance.RES_Units_milp.get_values()
     RES_Specific_Area = instance.RES_Specific_Area.extract_values()      # land area required per unit of capacity
     RES_Nominal_Capacity = instance.RES_Nominal_Capacity.extract_values()
     RES_Land = pd.DataFrame()
     for r in range(1,R+1):
        r_land = (RES_Specific_Area[r])*RES_Units_milp[1,r]*RES_Nominal_Capacity[r]   # step-1 land use in m2
        res_land = pd.DataFrame([RES_Names[r], 'm2', r_land]).T.set_index([0,1])
        if ST == 1 :
            res_land.columns = ['Total']
        else:
            res_land.columns = ['Step 1']
        res_land.index.names = ['Component', 'Unit']
        RES_Land = pd.concat([RES_Land,res_land], axis=1).fillna(0)
        for (y,st) in tup_list:
            res_land = pd.DataFrame([RES_Names[r], 'm2', ((RES_Specific_Area[r])*(RES_Units_milp[st,r]-RES_Units_milp[st-1,r])*RES_Nominal_Capacity[r])]).T.set_index([0,1])   # incremental land use added at step st
            if ST == 1:
                res_land.columns = ['Total']
            else:
                res_land.columns = ['Step '+str(st)]
            res_land.index.names = ['Component', 'Unit']
            RES_Land = pd.concat([RES_Land,res_land], axis=1).fillna(0)
     RES_Land = RES_Land.T.groupby(level=[0], sort=False).sum().T
     res_land_tot = RES_Land.sum(1).to_frame()
     res_land_tot.columns = ['Total']
     if ST != 1:
        RES_Land = pd.concat([RES_Land, res_land_tot],axis=1)
         
    else:  # continuous (LP) capacity expansion
        
     "Renewable Sources"   
     RES_Units = instance.RES_Units.get_values()
     RES_Specific_Area = instance.RES_Specific_Area.extract_values()
     RES_Nominal_Capacity = instance.RES_Nominal_Capacity.extract_values()
     RES_Land = pd.DataFrame()
     for r in range(1,R+1):
        r_land = (RES_Specific_Area[r])*RES_Units[1,r]*RES_Nominal_Capacity[r]   # step-1 land use in m2
        res_land = pd.DataFrame([RES_Names[r], 'm2', r_land]).T.set_index([0,1])
        if ST == 1 :
            res_land.columns = ['Total']
        else:
            res_land.columns = ['Step 1']
        res_land.index.names = ['Component', 'Unit']
        RES_Land = pd.concat([RES_Land,res_land], axis=1).fillna(0)
        for (y,st) in tup_list:
            res_land = pd.DataFrame([RES_Names[r], 'm2', ((RES_Specific_Area[r])*(RES_Units[st,r]-RES_Units[st-1,r])*RES_Nominal_Capacity[r])]).T.set_index([0,1])   # incremental land use added at step st
            if ST == 1:
                res_land.columns = ['Total']
            else:
                res_land.columns = ['Step '+str(st)]
            res_land.index.names = ['Component', 'Unit']
            RES_Land = pd.concat([RES_Land,res_land], axis=1).fillna(0)
     RES_Land = RES_Land.T.groupby(level=[0], sort=False).sum().T
     res_land_tot = RES_Land.sum(1).to_frame()
     res_land_tot.columns = ['Total']
     if ST != 1:
        RES_Land = pd.concat([RES_Land, res_land_tot],axis=1)

               
    #%% Concatenating
    SystemLand = pd.concat([round(RES_Land.astype(float),2)], axis=0).fillna('-')
    print("\nSystem Land Use [m2]")
    print(SystemLand)
    print("\n------------------------------------------------------------------------------------")
    return SystemLand   
                  
#%%
def PrintResults(instance, Results, callback=None):
    # Console summary of the key headline results (NPC, investment/operation cost, revenues,
    # LCOE, renewable penetration, component usage shares, land use) extracted from the Results dict.

    Y = int(instance.Years.extract_values()[None])
    
    if int(instance.WACC_Calculation.extract_values()[None]) == 1:   # WACC was computed internally rather than supplied directly
        wacc_value = float(instance.Discount_Rate.extract_values()[None])
        print(f'\n\nWACC = {wacc_value:.4f} [-]')

    # Looked up by 'Cost item' label rather than fixed row position: Results['Costs'] has one
    # extra "Net present cost"/"Total variable O&M cost"/etc. row per scenario inserted after
    # each weighted total, so the row offset of everything below depends on Scenarios (S) and
    # hardcoded positions (as this used to do) silently read the wrong row for any S>1 run.
    Costs = Results['Costs']

    npc = float(Costs.xs('Weighted Net present cost', level='Cost item').iloc[0, 0])
    print(f'\nNPC = {round(npc, 2)} kUSD')

    total_investment_cost = float(Costs.xs('Total Investment cost', level='Cost item').iloc[0, 0])
    print(f'Total actualized Investment Cost = {round(total_investment_cost, 2)} kUSD')

    salvage_value = float(Costs.xs('Salvage value', level='Cost item').iloc[0, 0])

    # Operation cost (fixed + variable O&M) derived algebraically from already-correct
    # aggregate rows (NPC = Investment + Fixed O&M + Variable O&M + Salvage, with Salvage
    # stored as a signed value) instead of summing two more hardcoded row positions, since
    # "Total variable O&M cost" itself has one row per scenario, not a single aggregate row.
    operation_cost = npc - total_investment_cost - salvage_value
    print(f'Total actualized Operation Cost = {round(operation_cost, 2)} kUSD')

    if instance.Grid_Connection.value == 1:
        Scenario_Weight_map = instance.Scenario_Weight.extract_values()
        Grid_Rev = Costs.xs('Grid electricity revenue', level='Cost item')
        electricity_revenues = float(sum(Grid_Rev.iloc[i, 0] * Scenario_Weight_map[Grid_Rev.index[i][1]] for i in range(len(Grid_Rev))))   # scenario-weighted sum of grid revenue rows
        print(f'Total actualized Electricity Selling Revenues = {round(electricity_revenues, 2)} kUSD')

    print(f'Salvage Value = {round(salvage_value, 2)} kUSD')

    lcoe = float(Costs.xs('Levelized Cost of Energy ', level='Cost item').iloc[0, 0])
    print(f'LCOE = {lcoe} USD/kWh')
    
    print("\n------------------------------------------------------------------------------------")

    renewable_penetration = Results['Renewables Penetration'].sum().sum() / Y   # average yearly renewable penetration
    print(f'\nAverage renewable penetration per year = {round(renewable_penetration, 2)} %')
    
    if instance.Model_Components.value == 1 and instance.Grid_Connection == 1:   # Battery + Grid, no generators
        battery_usage = Results['Yearly energy parameters']['Battery usage'].sum().sum() / Y
        print(f'Average battery usage per year = {round(battery_usage, 2)} %')
        grid_usage = Results['Yearly energy parameters']['Grid usage'].sum().sum() / Y
        print(f'Average national grid usage per year = {round(grid_usage, 2)} %')
        curtailment = Results['Yearly energy parameters']['Curtailment share'].sum().sum() / Y
        print(f'Average curtailment per year = {round(curtailment, 2)} %')

    if instance.Model_Components.value == 1 and instance.Grid_Connection == 0:   # Battery only, off-grid
        battery_usage = Results['Yearly energy parameters']['Battery usage'].sum().sum() / Y
        print(f'Average battery usage per year = {round(battery_usage, 2)} %')
        curtailment = Results['Yearly energy parameters']['Curtailment share'].sum().sum() / Y
        print(f'Average curtailment per year = {round(curtailment, 2)} %')
    
    if instance.Model_Components.value == 2 and instance.Grid_Connection == 1:   # Generators + Grid, no battery
        generator_share = Results['Yearly energy parameters']['Generators share'].sum().sum() / Y
        print(f'Average generator share per year = {round(generator_share, 2)} %')
        grid_usage = Results['Yearly energy parameters']['Grid usage'].sum().sum() / Y
        print(f'Average national grid usage per year = {round(grid_usage, 2)} %')      
        curtailment = Results['Yearly energy parameters']['Curtailment share'].sum().sum() / Y
        print(f'Average curtailment per year = {round(curtailment, 2)} %')

    if instance.Model_Components.value == 2 and instance.Grid_Connection == 0:   # Generators only, off-grid
        generator_share = Results['Yearly energy parameters']['Generators share'].sum().sum() / Y
        print(f'Average generator share per year = {round(generator_share, 2)} %')
        curtailment = Results['Yearly energy parameters']['Curtailment share'].sum().sum() / Y
        print(f'Average curtailment per year = {round(curtailment, 2)} %')

    if instance.Model_Components.value == 0 and instance.Grid_Connection == 1:   # RES + Battery + Generators + Grid
        generator_share = Results['Yearly energy parameters']['Generators share'].sum().sum() / Y
        print(f'Average generator share per year = {round(generator_share, 2)} %')
        battery_usage = Results['Yearly energy parameters']['Battery usage'].sum().sum() / Y
        print(f'Average battery usage per year = {round(battery_usage, 2)} %')
        grid_usage = Results['Yearly energy parameters']['Grid usage'].sum().sum() / Y
        print(f'Average national grid usage per year = {round(grid_usage, 2)} %')
        curtailment = Results['Yearly energy parameters']['Curtailment share'].sum().sum() / Y
        print(f'Average curtailment per year = {round(curtailment, 2)} %')

    if instance.Model_Components.value == 0 and instance.Grid_Connection == 0:   # RES + Battery + Generators, off-grid
        generator_share = Results['Yearly energy parameters']['Generators share'].sum().sum() / Y
        print(f'Average generator share per year = {round(generator_share, 2)} %')
        battery_usage = Results['Yearly energy parameters']['Battery usage'].sum().sum() / Y
        print(f'Average battery usage per year = {round(battery_usage, 2)} %')
        curtailment = Results['Yearly energy parameters']['Curtailment share'].sum().sum() / Y
        print(f'Average curtailment per year = {round(curtailment, 2)} %')
            
    if instance.Land_Use.value == 1:
        total_land_use = Results['Land Use']['Total'].sum()
        share_land_use = (total_land_use / instance.Renewables_Total_Area.value)*100     # land actually used as a % of the total available renewable land
        print(f'Renewables Land Use = {round(share_land_use, 2)} %')


