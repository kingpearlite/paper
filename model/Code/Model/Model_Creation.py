

import os                # Reads MGPY_MAX_BATTERY_KWH for the linearised single-flow big-M
from pyomo.environ import Param, RangeSet, Any, NonNegativeReals, NonNegativeIntegers, Var, Set, Reals, Binary
from Initialize import * # Import library with initialitation functions for the parameters

# Pathway 1 (2026-08-25): relax the per-hour generator unit-commitment variables
# (Generator_Partial, Generator_Full) from binary/integer to continuous during free-sizing,
# leaving Generator_Units (the per-investment-step sizing decision, a tiny variable count)
# untouched. These two are what the ~3.15M binaries in #13/#14/#16's free-sizing MILPs come
# from (indexed by scenarios x years x generator_types x periods); relaxing them turns the
# free-sizing solve into essentially an LP with a handful of sizing integers, at the standard
# energy-systems-planning cost of a small (~1-2%, per the literature) sizing suboptimality --
# recovered exactly by re-solving the pinned design through the existing Fix_Design()/
# fixed-design pipeline, which already re-imposes true integer unit commitment. Default
# unset/0 preserves exact historical behaviour; opt in per-run, not a blanket default, since
# it changes what's being optimized, not just how it's solved.
RELAX_UC = os.environ.get('MGPY_RELAX_UC', '0') == '1'

def Model_Creation(model):

#%% PARAMETERS
############## 


    "RES Time Series Estimation parameters"
    model.base_URL= Param(within=Any)                                        # Base URL of the NASA POWER API endpoint
    model.loc_id = Param(within=Any)                                         # Location identifier used in the API request
    model.parameters_1 = Param(within=Any)                                   # First set of NASA POWER API query parameters (e.g. solar variables)			
    model.parameters_2 = Param(within=Any)                                   # Second set of NASA POWER API query parameters (e.g. wind variables)			
    model.parameters_3 = Param(within=Any)                                   # Third set of NASA POWER API query parameters				
    model.date_start =  Param(within=Any)                                    # Start date of the RES time series request						
    model.date_end =  Param(within=Any)                                      # End date of the RES time series request						
    model.community = Param(within=Any)                                      # NASA POWER community type (e.g. renewable energy)			
    model.temp_res_1 = Param(within=Any)                                     # Temporal resolution of the first parameter set (e.g. hourly)						
    model.temp_res_2 = Param(within=Any)                                     # Temporal resolution of the second parameter set							
    model.output_format = Param(within=Any)                                  # Output file format requested from the API (e.g. CSV, JSON)					
    model.user = Param(within=Any)                                           # User identifier for the API request						
    model.lat = Param(within=Any)                                            # Latitude of the site location [deg]						
    model.lon = Param(within=Any)                                            # Longitude of the site location [deg]							
    model.time_zone = Param(within=Any)                                      # Time zone of the site location								
    model.nom_power = Param(within=Any)                                      # Nominal power of the reference PV/wind module used for scaling						
    model.tilt = Param(within=Any)                                           # Tilt angle of the PV panel [deg]								   
    model.azim = Param(within=Any)                                           # Azimuth angle of the PV panel [deg]								   
    model.ro_ground = Param(within=Any)                                      # Ground albedo/reflectance coefficient						
    model.k_T = Param(within=Any)                                            # PV temperature coefficient of power							
    model.NMOT = Param(within=Any)                                           # Nominal Module Operating Temperature of the PV panel [°C]								   
    model.T_NMOT = Param(within=Any)                                         # Ambient temperature at NMOT test conditions [°C]						
    model.G_NMOT = Param(within=Any)                                         # Irradiance at NMOT test conditions [W/m^2]							
    model.turbine_type = Param(within=Any)                                   # Wind turbine type identifier							
    model.turbine_model = Param(within=Any)                                  # Wind turbine model name/identifier				 	
    model.drivetrain_efficiency = Param(within=Any)                          # Wind turbine drivetrain efficiency [%]			
        
    "Demand Estimation parameters"
    model.demand_growth = Param(within=Any)                                  # Annual demand growth rate used to scale load profile							
    model.cooling_period = Param(within=Any)                                 # Period of the year when cooling appliances are active							
    model.h_tier1 = Param(within=Any)                                        # Number of households in demand tier 1								
    model.h_tier2 = Param(within=Any)                                        # Number of households in demand tier 2							
    model.h_tier3 = Param(within=Any)                                        # Number of households in demand tier 3							
    model.h_tier4 = Param(within=Any)                                        # Number of households in demand tier 4								
    model.h_tier5 = Param(within=Any)                                        # Number of households in demand tier 5								
    model.schools = Param(within=Any)                                        # Number of schools included in the demand archetype							
    model.hospital_1 = Param(within=Any)                                     # Number of type-1 hospitals included in the demand archetype							
    model.hospital_2 = Param(within=Any)                                     # Number of type-2 hospitals included in the demand archetype							
    model.hospital_3 = Param(within=Any)                                     # Number of type-3 hospitals included in the demand archetype							
    model.hospital_4 = Param(within=Any)                                     # Number of type-4 hospitals included in the demand archetype						
    model.hospital_5 = Param(within=Any)                                     # Number of type-5 hospitals included in the demand archetype
      
    "Project parameters"
    model.Periods                           = Param(within=NonNegativeIntegers)                          # Number of periods of analysis of the energy variables
    model.Years                             = Param(within=NonNegativeIntegers)                          # Number of years of the project
    model.Step_Duration                     = Param(within=NonNegativeIntegers)                          # Duration (in years) of each investment decision step in which the project lifetime will be split
    model.Min_Last_Step_Duration            = Param(within=NonNegativeIntegers)                          # Minimum duration (in years) of the last investment decision step, in case of non-homogeneous divisions of the project lifetime
    model.Delta_Time                        = Param(within=NonNegativeReals)                             # Time step in hours
    model.StartDate                         = Param(within=Any)                                                    # Start date of the analisis
    model.Scenarios                         = Param(within=NonNegativeIntegers)                          # Number of scenarios to consider within the optimisation
    model.Real_Discount_Rate                = Param(within=NonNegativeReals)                             # Real Discount rate (default value) [%]
    model.Discount_Rate                     = Param(within=NonNegativeReals,
                                                    initialize = Initialize_Discount_Rate)               # Discount rate initialized according to WACC calculation [%]
    model.Investment_Cost_Limit             = Param(within=NonNegativeReals)                             # Upper limit to investment cost [USD] (considered only in case Optimization_Goal='Operation cost')
    model.Steps_Number                      = Param(initialize = Initialize_Upgrades_Number)             # Number of steps
    model.RES_Sources                       = Param(within=NonNegativeIntegers)                          # Number of Renewable Energy Sources (RES) types
    model.Generator_Types                   = Param(within=NonNegativeIntegers)                          # Number of different types of gensets
    model.Year_Grid_Connection              = Param(within=NonNegativeIntegers)                          # Year at which microgrid is connected to the national grid (starting from 1)
    model.Renewable_Penetration             = Param(within=NonNegativeReals)                             # Fraction of electricity produced by renewable sources. Number from 0 to 1. 
    model.Renewables_Total_Area             = Param(within=NonNegativeReals,
                                                    initialize = 10^6)                                   # Total available area for technology installation [m^2]
    model.Battery_Independence              = Param(within=NonNegativeIntegers)                          # Number of days of autonomy of the battery bank
                                                                                
    # WACC Calculation
    model.cost_of_equity                    = Param(within=NonNegativeReals)                             # Cost of equity (i.e., the return required by the equity shareholders) [-]
    model.cost_of_debt                      = Param(within=NonNegativeReals)                             # Cost of debt (i.e., the interest rate) [-]
    model.tax                               = Param(within=NonNegativeReals)                             # Corporate tax deduction (debt is assumed as tax deducible) [-]
    model.equity_share                      = Param(within=NonNegativeReals)                             # Total level of equity [kUSD]
    model.debt_share                        = Param(within=NonNegativeReals)                             # Total level of debt [kUSD]
    
    model.Pareto_points                     = Param(within=NonNegativeIntegers)                          # Points to be analysed in Multi-Objective Optimization
    model.Pareto_solution                   = Param(within=NonNegativeIntegers)                          # Solution (from 1 to pareto_points) of the Multi_Objective Optimization to be displayed
    
    "Model Switches"
    model.Optimization_Goal                 = Param(within=Binary)                                    # Options: 1 = NPC / 0 = Operation cost. It allows to switch between a NPC-oriented optimization and a NON-ACTUALIZED Operation Cost-oriented optimization
    model.Multiobjective_Optimization       = Param(within=Binary)                                    # 1 if optimization of NPC/operation cost and CO2 emissions,0 otherwise
    model.MILP_Formulation                  = Param(within=Binary)                                    # 1 to activate MILP formulation (for monodirectional energy flows), 0 otherwise
    model.Generator_Partial_Load            = Param(within=Binary)                                    # 1 to activate Partial Load effect on the operation costs of the generator, 0 otherwise
    model.Greenfield_Investment             = Param(within=Binary)                                    # 1 if Greenfield investment, 0 Brownfield investment
    model.RE_Supply_Calculation             = Param(within=Binary)                                    # 1 to select solar PV and wind production time series calculation (using NASA POWER data), 0 otherwise
    model.Demand_Profile_Generation         = Param(within=Binary)                                    # 1 to select load demand profile generation (with demand archetypes), 0 otherwise
    model.Grid_Connection                   = Param(within=Binary)                                    # 1 to select grid connection during project lifetime, 0 otherwise
    model.Grid_Availability_Simulation      = Param(within=Binary)                                    # 1 to simulate grid availability, 0 otherwise
    model.Grid_Connection_Type              = Param(within=Binary)                                    # 0 for sell/purchase power with the national grid, 1 for purchase only
    model.Plot_Max_Cost                     = Param(within=Binary)                                    # 1 if the Pareto curve has to include the point at maxNPC/maxOperationCost, 0 otherwise
    model.Model_Components                  = Param(within=NonNegativeIntegers)                       # 0 for batteries and generators, 1 for batteries only, 2 for generators only
    model.WACC_Calculation                  = Param(within=Binary)                                    # 1 to select Weighted Average Cost of Capital calculation, 0 otherwise
    model.Fuel_Specific_Cost_Import         = Param(within=Binary)                                    # 1 to import variable fuel specific cost from csv file (only if Fuel_Specific_Cost_Calculation activated)
    model.Fuel_Specific_Cost_Calculation    = Param(within=Binary)                                    # 1 to allows variable fuel specific cost across the years, 0 otherwise
    model.Land_Use                          = Param(within=Binary)                                    # 1 to activate the constraint on the total land use, 0 otherwise
    model.Solver                            = Param(within=NonNegativeIntegers)                       # 0 for Gurobi, 1 for GLPK and 2 for HiGHS (currently NOT available)
    
    "Sets"
    model.periods                           = RangeSet(1, model.Periods)                                  # Creation of a set from 1 to the number of periods in each year
    model.years                             = RangeSet(1, model.Years)                                    # Creation of a set from 1 to the number of years of the project
    model.scenarios                         = RangeSet(1, model.Scenarios)                                # Creation of a set from 1 to the number of scenarios to analized
    model.renewable_sources                 = RangeSet(1, model.RES_Sources)                              # Creation of a set from 1 to the number of RES technologies to analized
    model.generator_types                   = RangeSet(1, model.Generator_Types)                          # Creation of a set from 1 to the number of generators types to analized
    model.steps                             = RangeSet(1, model.Steps_Number)                             # Creation of a set from 1 to the number of investment decision steps
    model.years_steps                       = Set(dimen = 2, initialize=Initialize_YearUpgrade_Tuples)    # 2D set of tuples: it associates each year to the corresponding investment decision step
    model.years_grid_connection             = RangeSet(model.Year_Grid_Connection,model.Years)            # Creation of a set from year of grid connection to last year
    model.Scenario_Weight                   = Param(model.scenarios, within=NonNegativeReals)             # Probability/weight assigned to each scenario


    
    "Parameters of RES Technologies" 
    model.RES_Names                    = Param(model.renewable_sources,
                                               within=Any)                          # RES names
    model.RES_Nominal_Capacity         = Param(model.renewable_sources,
                                               within=NonNegativeReals)               # Nominal capacity of the RES in W/unit
    model.RES_Inverter_Efficiency      = Param(model.renewable_sources,
                                               within=NonNegativeReals)               # Efficiency of the inverter in %
    model.RES_Specific_Investment_Cost = Param(model.renewable_sources,
                                               within=NonNegativeReals)               # Cost of RES in USD/W
    model.RES_Specific_OM_Cost         = Param(model.renewable_sources,
                                               within=NonNegativeReals)               # Percentage of the total investment spend in operation and management of solar panels in each period in %                                             
    model.RES_Lifetime                 = Param(model.renewable_sources,
                                               within=NonNegativeIntegers)            # Lifetime of each Renewable Energy Source (RES) [y]
    model.RES_Specific_Area            = Param(model.renewable_sources,
                                               within=NonNegativeReals)               # Specific area of each PV technology in m^2/W
    model.RES_unit_CO2_emission        = Param(model.renewable_sources,
                                               within=NonNegativeReals)               # [kgCO2/kW]                                   
    model.RES_Unit_Energy_Production   = Param(model.scenarios,
                                              model.renewable_sources,
                                              model.periods, 
                                              within=NonNegativeReals, 
                                              initialize=Initialize_RES_Energy)      # Energy production of a RES in Wh
    # Brownfield
    model.RES_capacity                 = Param(model.renewable_sources,
                                               within=NonNegativeReals)               # Existing RES units [-] of nominal capacity
    model.RES_years                    = Param(model.renewable_sources,
                                               within=NonNegativeIntegers)            # How many years ago the component was installed [y]
    model.RES_existing_area            = Param(model.renewable_sources,
                                               within=NonNegativeReals)               # Existing area of each PV technology in brownfield [m^2] 



    "Parameters of the battery bank"
    model.Battery_Specific_Investment_Cost            = Param(within=NonNegativeReals)                                  # Specific investment cost of the battery bank [USD/Wh]
    model.Battery_Specific_Electronic_Investment_Cost = Param(within=NonNegativeReals)                                  # Specific investment cost of non-replaceable parts (electronics) of the battery bank [USD/Wh]
    model.Battery_Specific_OM_Cost                    = Param(within=NonNegativeReals)                                  # Percentage of the total investment spend in operation and management of batteries in each period in %
    model.Battery_Discharge_Battery_Efficiency        = Param(within=NonNegativeReals)                                  # Efficiency of the discharge of the battery in %
    model.Battery_Charge_Battery_Efficiency           = Param(within=NonNegativeReals)                                  # Efficiency of the charge of the battery in  %
    model.Battery_Depth_of_Discharge                  = Param(within=NonNegativeReals)                                  # Depth of discharge of the battery in %
    model.Maximum_Battery_Discharge_Time              = Param(within=NonNegativeReals)                                  # Minimum time of charge of the battery in hours
    model.Maximum_Battery_Charge_Time                 = Param(within=NonNegativeReals)                                  # Maximum time of discharge of the battery in hours                     
    model.Battery_Cycles                              = Param(within=NonNegativeReals)                                  # Maximum number of charge/discharge cycles of the battery
    model.Unitary_Battery_Replacement_Cost            = Param(within=NonNegativeReals, 
                                                              initialize=Initialize_Battery_Unit_Repl_Cost)             # Replacement cost per unit capacity, computed from investment/electronic costs
    model.Battery_Initial_SOC                         = Param(within=NonNegativeReals)                                  # Initial State of Charge of the battery bank [%]

    # Ceiling on energy charged into the battery in one time step [Wh], used ONLY as
    # the big-M of the linearised single-flow constraint (MGPY_BESS_FORM=linear).
    # Unlike the discharge side -- where Max_Bat_out already caps outflow at demand,
    # giving an exact, data-derived M for free -- nothing in the model bounds inflow,
    # because RES and generator output both scale with unbounded integer unit counts.
    # So the linearised form needs an explicit ceiling, taken from MGPY_MAX_BATTERY_KWH.
    # Defaults to 0 when unused; Constraints.py only reads it in linear mode, and
    # Model_Resolution refuses to start in that mode without the env var set.
    def _initialize_battery_max_inflow(model):
        max_kwh = float(os.environ.get('MGPY_MAX_BATTERY_KWH', 0.0))
        return max_kwh  / model.Maximum_Battery_Charge_Time * model.Delta_Time
    model.Battery_Max_Inflow                          = Param(within=NonNegativeReals,
                                                              initialize=_initialize_battery_max_inflow)                # Big-M for the linearised charge gate [Wh per time step]
    # Brownfield
    model.Battery_capacity                            = Param(within=NonNegativeReals)                                  # Existing battery bank capacity in brownfield [Wh]
    model.BESS_unit_CO2_emission                      = Param(within=NonNegativeReals)                                  # CO2 emission factor of the battery bank [kgCO2/kWh]
    model.Battery_Min_Capacity                        = Param(model.steps, 
                                                              initialize=Initialize_Battery_Minimum_Capacity)           # Minimum allowed battery capacity at each investment step
    model.BESS_Large_Constant                         = Param(within=NonNegativeReals,
                                                              initialize = 10**6)                                       # Big-M constant used in battery-related logic constraints
    # MILP Formulation
    model.Battery_Nominal_Capacity_milp               = Param(within=NonNegativeReals)         # Nominal Capacity of each battery

   
# --- Generators ---

    "Parameters of the genset"
    model.Generator_Names                    = Param(model.generator_types,
                                                     within=Any)                           # Generators names
    model.Generator_Efficiency               = Param(model.generator_types,
                                                   within=NonNegativeReals)                # Generator efficiency to trasform heat into electricity %
    model.Generator_Specific_Investment_Cost = Param(model.generator_types,
                                                     within=NonNegativeReals)              # Cost of the diesel generator
    model.Generator_Specific_OM_Cost         = Param(model.generator_types,
                                                     within=NonNegativeReals)              # Cost of the diesel generator
    model.Generator_Lifetime                 = Param(model.generator_types,
                                                     within=NonNegativeIntegers)                            # Lifetime of each generator type [y]
    model.Fuel_Names                         = Param(model.generator_types,
                                                     within=Any)                          # Fuel names
    model.Fuel_LHV                           = Param(model.generator_types,
                                                     within=NonNegativeReals)              # Low heating value of the fuel in kg/l
    model.GEN_years                          = Param(model.generator_types,
                                                     within=NonNegativeIntegers)                             # How many years ago the generator was installed [y] (brownfield)
    model.GEN_unit_CO2_emission              = Param(model.generator_types,
                                                     within=NonNegativeReals)                                # CO2 emission factor of the generator [kgCO2/kWh]
    model.FUEL_unit_CO2_emission             = Param(model.generator_types,
                                                     within=NonNegativeReals)                                # CO2 emission factor of the fuel [kgCO2/l]
    # MILP Formulation
    model.Generator_Nominal_Capacity_milp     = Param(model.generator_types,
                                                      within=NonNegativeReals)                             # During the MILP optimization 𝐶 is defined as a parameter!

    # Ceiling on total installed generator capacity [kW], used ONLY to tighten the LP
    # relaxation on Generator_Units -- that variable is NonNegativeIntegers with no
    # upper bound (Model_Creation.py:384), which was harmless at fine sizing-granularity
    # (Generator_Nominal_Capacity_milp in the low kW) but became a genuine B&B memory-
    # explosion risk once granularity was corrected to match real installed unit sizes
    # (470kW, Gili Ketapang) -- coarse blocks mean the same unboundedness translates into
    # a much looser fractional relaxation. Same opt-in pattern as Battery_Max_Inflow
    # above: defaults to a large inert constant when unused, so every design that
    # doesn't set MGPY_MAX_GENERATOR_KW is completely unaffected.
    def _initialize_generator_max_capacity(model, g):
        max_kw = float(os.environ.get('MGPY_MAX_GENERATOR_KW', 0.0))
        return max_kw if max_kw > 0 else 10**9   # unset/0 -> effectively unbounded, a no-op
    model.Generator_Max_Capacity              = Param(model.generator_types,
                                                      within=NonNegativeReals,
                                                      initialize=_initialize_generator_max_capacity)       # Opt-in ceiling on installed generator capacity [kW], via MGPY_MAX_GENERATOR_KW
    # Partial Load Effect
    model.Generator_Min_output                = Param(model.generator_types,
                                                      within=NonNegativeReals)                             # Minimum percentage of energy output for the generator in part load
    model.Generator_pgen                      = Param(model.generator_types,
                                                      within=NonNegativeReals)                             # Percentage of the total operation cost of the generator system at full load
    # Variable Fuel Cost
    #----------------------------------------------------------------------------------------------
    model.Fuel_Specific_Cost                 = Param(model.generator_types,
                                                     model.years,
                                                     within=Any,
                                                     initialize=Initialize_Fuel_Specific_Cost)               # Fuel cost per year, per generator type [USD/l]
    model.Fuel_Specific_Start_Cost           = Param(model.generator_types,
                                                     within=NonNegativeReals)                                # Fuel cost at project start, per generator type [USD/l]
    model.Fuel_Specific_Cost_Rate            = Param(model.generator_types,
                                                     within=Reals)                                           # Annual escalation rate of the fuel cost [%]
    model.Generator_Marginal_Cost            = Param(model.generator_types,
                                                     model.years,
                                                     within=Any,
                                                     initialize=Initialize_Generator_Marginal_Cost)          # Marginal generation cost per year, per generator type [USD/Wh]
    model.Generator_Start_Cost                = Param(model.generator_types,
                                                      model.years,
                                                      within=Any,
                                                      initialize=Initialize_Generator_Start_Cost)            # Cost of starting up the generator, per year and type [USD]
    #------------------------------------------------------------------------------------------------
    model.Fuel_Specific_Cost_1               = Param(model.generator_types,
                                                     within=Any,
                                                     initialize=Initialize_Fuel_Specific_Cost_1)             # Origin (intercept) of the fuel cost curve
    model.Generator_Marginal_Cost_1          = Param(model.generator_types,
                                                     within=Any,
                                                     initialize=Initialize_Generator_Marginal_Cost_1)        # Origin (intercept) of the marginal cost curve
    model.Generator_Marginal_Cost_milp        = Param(model.generator_types,
                                                      model.years,
                                                      within=Any,
                                                      initialize=Initialize_Generator_Marginal_Cost_milp)    # Marginal generation cost used in the MILP formulation
    model.Generator_capacity                  = Param(model.generator_types,
                                                      within=NonNegativeReals)                             # Existing capacity of generator

    model.Generator_Start_Cost_1              = Param(model.generator_types,
                                                      within=Any,
                                                      initialize=Initialize_Generator_Start_Cost_1)          # # Origin of the cost curve of the part load generator
    model.Generator_Marginal_Cost_milp_1      = Param(model.generator_types,
                                                      within=Any,
                                                      initialize=Initialize_Generator_Marginal_Cost_milp_1)  # Slope of the cost curve of the part load generator

# --- National Grid ---

    "Parameters of the National Grid"
    model.Grid_Sold_El_Price                   = Param(within=NonNegativeReals)                            # Price at which electricity is sold to the national grid [USD/Wh]
    model.Grid_Purchased_El_Price              = Param(within=NonNegativeReals)                            # Price at which electricity is purchased from the national grid [USD/Wh]
    model.Grid_Lifetime                        = Param(within=NonNegativeReals)                            # Lifetime of the grid connection infrastructure [y]
    model.Grid_Distance                        = Param(within=NonNegativeReals)                            # Distance from the site to the national grid [km]
    model.Grid_Connection_Cost                 = Param(within=NonNegativeReals)                            # Cost of connecting to the national grid per km [USD/km]
    model.Grid_Maintenance_Cost                = Param(within=NonNegativeReals)                            # Annual maintenance cost of the grid connection [USD]
    model.Maximum_Grid_Power                   = Param(within=NonNegativeReals)                            # Maximum power exchangeable with the national grid [W]
    model.Grid_Availability                    = Param(model.scenarios,
                                                       model.years,
                                                       model.periods,
                                                       within=Any,
                                                       initialize = Initialize_Grid_Availability)          # Binary/fractional availability of the national grid at each time step
    model.Grid_Average_Number_Outages          = Param(within=NonNegativeReals)                            # Average number of grid outages per year 
    model.Grid_Average_Outage_Duration         = Param(within=NonNegativeReals)                            # Average duration of each grid outage [h]                
    model.National_Grid_Specific_CO2_emissions = Param(within=NonNegativeReals)                             # CO2 emission factor of the national grid [kgCO2/kWh]  
    
 
    "Parameters of the electricity balance"                  
    model.Energy_Demand           = Param(model.scenarios, 
                                          model.years, 
                                          model.periods, 
                                          initialize=Initialize_Demand)             # Energy Energy_Demand in W 
    model.Lost_Load_Fraction      = Param(within=NonNegativeReals)                  # Lost load maxiumum admittable fraction in %
    model.Lost_Load_Specific_Cost = Param(within=NonNegativeReals)                  # Value of lost load in USD/Wh 
    model.Large_Constant          = Param(within=NonNegativeReals,
                                          initialize = 10**6)                       # Big-M constant used in logic constraints (e.g. mutually exclusive flows)
    
    "Parameters of the plot"
    model.RES_Colors             = Param(model.renewable_sources,within=Any)                       # HEX color codes for RES
    model.Battery_Color          = Param(within=Any)                                               # HEX color codes for Battery bank
    model.Generator_Colors       = Param(model.generator_types,within=Any)                         # HEX color codes for Generators
    model.Lost_Load_Color        = Param(within=Any)                                               # HEX color codes for Lost load
    model.Curtailment_Color      = Param(within=Any)                                               # HEX color codes for Curtailment
    model.Energy_To_Grid_Color   = Param(within=Any)                                               # HEX color codes for Energy to grid
    model.Energy_From_Grid_Color = Param(within=Any)                                               # HEX color codes for Energy from grid
    
#%% VARIABLES
##############


    "Variables associated to the RES"
    model.RES_Units             = Var(model.steps, 
                                      model.renewable_sources,
                                      within=NonNegativeReals)                      # Number of units of RES (LP Formulation)
    model.RES_Energy_Production = Var(model.scenarios, 
                                      model.years,
                                      model.renewable_sources,
                                      model.periods,
                                      within=NonNegativeReals)                      # Energy generated by the RES sistem in Wh
    model.RES_emission          = Var(within=NonNegativeReals)                          # Total CO2 emissions embodied in the RES units
    model.RES_Land_Use          = Var(model.scenarios, 
                                      model.years,
                                      model.steps,
                                      model.renewable_sources,
                                      model.periods,
                                      within=NonNegativeReals)                       # Land use of the RES in m^2
    
    # MILP Formulation
    model.RES_Units_milp        = Var(model.steps, 
                                      model.renewable_sources,
                                      within=NonNegativeIntegers)                   # Number of units of RES (MILP Formulation)


    "Variables associated to the battery bank"
    model.Battery_Nominal_Capacity        = Var(model.steps, 
                                                within=NonNegativeReals)            # Capacity of the battery bank in Wh
    model.Battery_Outflow                 = Var(model.scenarios, 
                                                model.years, 
                                                model.periods,
                                                within=NonNegativeReals)            # Battery discharge energy in Wh
    model.Battery_Inflow                  = Var(model.scenarios,
                                                model.years, 
                                                model.periods, 
                                                within=NonNegativeReals)            # Battery charge energy in Wh
    
    model.Battery_SOC                     = Var(model.scenarios, 
                                                model.years, 
                                                model.periods, 
                                                within=NonNegativeReals)            # State of Charge of the Battery in Wh
    model.Battery_Maximum_Charge_Power    = Var(model.steps, 
                                                within=NonNegativeReals)            # Maximum charge power of the battery bank in W
    model.Battery_Maximum_Discharge_Power = Var(model.steps,
                                                within=NonNegativeReals)            # Maximum discharge power of the battery bank in W
    model.Battery_Replacement_Cost_Act    = Var(model.scenarios,
                                                within=NonNegativeReals)            # Actualized (discounted) cost of battery replacement
    model.Battery_Replacement_Cost_NonAct = Var(model.scenarios,
                                                within=NonNegativeReals)            # Non-actualized cost of battery replacement
    model.BESS_emission                   = Var(within=NonNegativeReals)             # Total CO2 emissions embodied in the battery bank
    model.Single_Flow_BESS                     = Var(model.scenarios, 
                                                model.years, 
                                                model.periods,
                                                within = Binary)                    # Binary enforcing mutually exclusive charge/discharge flows
    # MILP Formulation
    model.Battery_Units         = Var(model.steps, 
                                      within=NonNegativeIntegers)                   # Number of units of Battery


    "Variables associated to the diesel generator"
    model.Generator_Nominal_Capacity  = Var(model.steps, 
                                            model.generator_types,
                                            within=NonNegativeReals)                # Capacity  of the diesel generator in Wh
    model.Generator_Energy_Production = Var(model.scenarios, 
                                            model.years,
                                            model.generator_types,
                                            model.periods, 
                                            within=NonNegativeReals)                # Energy generated by the Diesel generator
    model.Total_Fuel_Cost_Act         = Var(model.scenarios,
                                            model.generator_types,
                                            within=NonNegativeReals)                # Actualized (discounted) total fuel cost
    model.Total_Fuel_Cost_NonAct      = Var(model.scenarios, 
                                            model.generator_types,
                                            within=NonNegativeReals)                # Non-actualized total fuel cost
    model.GEN_emission                = Var(within=NonNegativeReals)                  # Total CO2 emissions embodied in the generators
    model.FUEL_emission               = Var(model.scenarios, 
                                            model.years,
                                            model.generator_types,
                                            model.periods, 
                                            within=NonNegativeReals)                # CO2 emissions from fuel combustion at each time step
    model.Scenario_FUEL_emission      = Var(model.scenarios,
                                            within=NonNegativeReals)                # Total fuel CO2 emissions per scenario

    # MILP Formulation
    model.Generator_Units                   = Var(model.steps, 
                                                  model.generator_types,
                                                  within=NonNegativeIntegers)                # Total number of generators
    model.Generator_Energy_Total            = Var(model.scenarios,
                                                  model.years,
                                                  model.generator_types,
                                                  model.periods,
                                                  within=NonNegativeReals)              # Total Energy Production of the generator
    # Partial Load Effect
    # MGPY_RELAX_UC=1 relaxes these from Binary/Integer to continuous for free-sizing (see
    # RELAX_UC comment at module top) -- Constraints.py's formulas are unchanged either way,
    # since they're linear expressions agnostic to the variable's domain.
    model.Generator_Partial                 = Var(model.scenarios,
                                                  model.years,
                                                  model.generator_types,
                                                  model.periods,
                                                  within=(NonNegativeReals if RELAX_UC else Binary),
                                                  bounds=((0,1) if RELAX_UC else None))               # Binary (or, relaxed, fraction in [0,1]) that controls if there will be a generator in part load
    model.Generator_Full                    = Var(model.scenarios,
                                                  model.years,
                                                  model.generator_types,
                                                  model.periods,
                                                  within=(NonNegativeReals if RELAX_UC else NonNegativeIntegers))                # Number of generator in full load (relaxed to continuous under MGPY_RELAX_UC)
    model.Generator_Energy_Partial          = Var(model.scenarios,
                                                  model.years,
                                                  model.generator_types,
                                                  model.periods,
                                                  within=NonNegativeReals)              # Energy produced by the last generator in partial load



    "Variable associated to the National Grid"  
    model.Total_Revenues_Act                = Var(model.scenarios,
                                                  within=NonNegativeReals)            # Actualized (discounted) revenues from selling energy to the grid
    model.Total_Revenues_NonAct             = Var(model.scenarios,
                                                  within=NonNegativeReals)            # Non-actualized revenues from selling energy to the grid
    model.Total_Electricity_Cost_Act        = Var(model.scenarios,
                                                  within=NonNegativeReals)            # Actualized (discounted) cost of electricity purchased from the grid
    model.Total_Electricity_Cost_NonAct     = Var(model.scenarios,
                                                  within=NonNegativeReals)            # Non-actualized cost of electricity purchased from the grid
    model.Energy_To_Grid                    = Var(model.scenarios, 
                                                  model.years,
                                                  model.periods, 
                                                  within=NonNegativeReals)            # Energy sold/exported to the national grid in Wh
    model.Energy_From_Grid                  = Var(model.scenarios, 
                                                  model.years,
                                                  model.periods, 
                                                  within=NonNegativeReals)            # Energy purchased/imported from the national grid in Wh
    model.GRID_emission                     = Var(model.scenarios, 
                                                  model.years,
                                                  model.periods, 
                                                  within=NonNegativeReals)             # CO2 emissions from grid electricity import at each time step
    model.Scenario_GRID_emission            = Var(model.scenarios,
                                                  within=NonNegativeReals)            # Total grid CO2 emissions per scenario
    model.Single_Flow_Grid                  = Var(model.scenarios,
                                                  model.years,
                                                  model.periods, 
                                                  within=Binary)                      # Binary enforcing mutually exclusive import/export flows with the grid
    
    "Variables associated to the energy balance"
    model.Lost_Load                      = Var(model.scenarios, 
                                               model.years, 
                                               model.periods, 
                                               within=NonNegativeReals)                      # Energy not supplied by the system kWh
    model.Energy_Curtailment             = Var(model.scenarios,
                                               model.years,
                                               model.periods, 
                                               within=NonNegativeReals)                      # Curtailment of RES in kWh
    model.Scenario_Lost_Load_Cost_Act    = Var(model.scenarios, 
                                               within=NonNegativeReals)              # Actualized (discounted) cost of lost load
    model.Scenario_Lost_Load_Cost_NonAct = Var(model.scenarios,
                                               within=NonNegativeReals)              # Non-actualized cost of lost load    
    

    "Variables associated to the project"
    model.Net_Present_Cost                    = Var(within=Reals)                                          # Net Present Cost of the whole project [USD]
    model.Scenario_Net_Present_Cost           = Var(model.scenarios, 
                                                    within=Reals)                                          # Net Present Cost computed for each scenario [USD]
    model.Total_Variable_Cost                 = Var(within=Reals)                                          # Total (non-actualized) operational/variable cost [USD]
    model.CO2_emission                        = Var(within=NonNegativeReals)                                 # Total CO2 emissions of the whole project
    model.Scenario_CO2_emission               = Var(model.scenarios, 
                                                    within=NonNegativeReals)                                # Total CO2 emissions per scenario
    model.Investment_Cost                     = Var(within=NonNegativeReals)                                # Total investment cost of the system [USD]
    model.Salvage_Value                       = Var(within=NonNegativeReals)                                # Residual value of components at the end of the project [USD]
    model.Salvage_Positive_Flag               = Var(within=Binary)                                          # Floor-at-zero helper for Salvage_Value: 1 iff the raw (undepreciated) residual-value formula is >=0. Needed because a component whose lifetime is much shorter than the project horizon and gets front-loaded (installed early rather than deferred) can drive the raw formula negative, which the NonNegativeReals domain above then rejects outright rather than flooring at zero (see Constraints.py's Salvage_Value rule)
    model.Total_Variable_Cost_Act             = Var(within=Reals)                                           # Actualized (discounted) total variable cost [USD]
    model.Operation_Maintenance_Cost_Act      = Var(within=Reals)                                           # Actualized (discounted) O&M cost [USD]
    model.Operation_Maintenance_Cost_NonAct   = Var(within=Reals)                                           # Non-actualized O&M cost [USD]
    model.Total_Scenario_Variable_Cost_Act    = Var(model.scenarios, 
                                                    within=Reals)                                           # Actualized (discounted) variable cost per scenario [USD]
    model.Total_Scenario_Variable_Cost_NonAct = Var(model.scenarios, 
                                                    within=Reals)                                           # Non-actualized variable cost per scenario [USD]


