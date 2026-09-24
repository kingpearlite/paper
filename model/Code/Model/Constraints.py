import re
import os
import params_path  # Resolves which parameters file this run reads (MGPY_PARAMS)

# BATTERY SINGLE-FLOW FORMULATION -- see Model_Resolution.Battery_Single_Flow_Form.
#   unset / 'bilinear'  historical default; one constraint enforces BOTH the C-rate
#                       cap and the no-simultaneous-flow gate, by multiplying the
#                       mode binary with the (variable) battery power. 525,600
#                       quadratic constraints, reformulated by Gurobi into 1,051,200
#                       SOS constraints.
#   'linear'            split into a linear cap and a linear big-M gate. Same
#                       feasible set for integral binaries; zero quadratic
#                       constraints. Requires MGPY_MAX_BATTERY_KWH.
#   'nobinary'          cap only, gate dropped entirely along with the binaries.
#                       NOT equivalent -- relies on simultaneous charge/discharge
#                       being uneconomic, which MUST be confirmed after every solve
#                       by the check Model_Resolution prints.
BESS_FORM = os.environ.get('MGPY_BESS_FORM', 'bilinear').strip().lower()
Single_Flow_Linear = (BESS_FORM == 'linear')
Single_Flow_NoBinary = (BESS_FORM == 'nobinary')

##############################################################################################################################################################
###################################################################### LP FORMULATION ########################################################################
##############################################################################################################################################################

#%%
# --- Greenfield ---

class Constraints_Greenfield():  # LP formulation — greenfield (no pre-existing assets)
    current_directory = os.path.dirname(os.path.abspath(__file__))  # Folder containing this module
    inputs_directory = os.path.join(current_directory, '..', 'Inputs')  # Sibling Inputs folder
    data_file_path = params_path.PARAMS_PATH  # Resolved once via MGPY_PARAMS; see params_path.py
    Data_import = open(data_file_path).readlines()  # Read the parameters file at class-definition time
    for i in range(len(Data_import)):  # Scan every line of the parameters file
        if "param: Fuel_Specific_Cost_Calculation" in Data_import[i]:      # Locate the flag line
            Fuel_Specific_Cost_Calculation = int((re.findall('\d+',Data_import[i])[0]))  # 0 = constant price, 1 = time-varying
        
    "Objective function"
    def Net_Present_Cost_Obj(model):  # Weighted expected NPC across all scenarios
        return (sum(model.Scenario_Net_Present_Cost[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected NPC = probability-weighted sum across scenarios
    
    def CO2_emission_Obj(model):  # Weighted expected total lifecycle CO2 across all scenarios
        return (sum(model.Scenario_CO2_emission[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected CO2 emissions = probability-weighted sum across scenarios
    
    def Total_Variable_Cost_Obj(model):  # Weighted expected variable cost (non-actualized) across all scenarios
        return (sum(model.Total_Scenario_Variable_Cost_NonAct[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected non-actualized variable cost = probability-weighted sum across scenarios
    
    "Net Present Cost"
    def Net_Present_Cost(model):   # Defines the scalar NPC as expected NPC over all scenarios
        return model.Net_Present_Cost == (sum(model.Scenario_Net_Present_Cost[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected NPC = probability-weighted sum across scenarios
    
    # def Scenario_Net_Present_Cost(model,s): 
    #     foo = []
    #     for g in range(1,model.Generator_Types+1):
    #             foo.append((s,g))            
    #     Fuel_Cost = sum(model.Total_Fuel_Cost_Act[s,g] for s,g in foo)    
    #     return model.Scenario_Net_Present_Cost[s] == (model.Investment_Cost + model.Operation_Maintenance_Cost_Act + model.Battery_Replacement_Cost_Act[s] 
    #             + model.Scenario_Lost_Load_Cost_Act[s] + Fuel_Cost - model.Salvage_Value)  
    
    def Total_Variable_Cost(model):  # Defines the scalar total variable cost (non-actualized)
        return model.Total_Variable_Cost == (sum(model.Total_Scenario_Variable_Cost_NonAct[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected non-actualized variable cost = probability-weighted sum across scenarios
    
    def Scenario_Net_Present_Cost(model,s):  # NPC for scenario s: CAPEX + variable OPEX - salvage value
        return model.Scenario_Net_Present_Cost[s] == (model.Investment_Cost + model.Total_Scenario_Variable_Cost_Act[s] - model.Salvage_Value)     # NPC = CAPEX + actualized variable cost - salvage value
    
    def CO2_emission(model):  # Defines the scalar CO2 emission as expected value over all scenarios
        return model.CO2_emission == (sum(model.Scenario_CO2_emission[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected CO2 emissions = probability-weighted sum across scenarios
    
    def Scenario_CO2_emission(model,s):  # Total lifecycle + operational CO2 for scenario s; components depend on active model parts
        if model.Grid_Connection == 1:  # Grid-connected: include grid import emissions
            if model.Model_Components == 0:  # RES + generator + battery + fuel burn + grid purchases
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.GEN_emission + model.BESS_emission + model.Scenario_FUEL_emission[s] + model.Scenario_GRID_emission[s])  # Full system with grid: RES + generator + battery + fuel + grid import emissions
            if model.Model_Components == 1:  # RES + battery + grid purchases (no generator)
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.BESS_emission + model.Scenario_GRID_emission[s])  # Grid-connected, no generator: RES + battery + grid import emissions
            if model.Model_Components == 2:  # RES + generator + fuel burn + grid purchases (no battery)
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.GEN_emission + model.Scenario_FUEL_emission[s] + model.Scenario_GRID_emission[s])  # Grid-connected, no battery: RES + generator + fuel + grid import emissions
        else:  # Off-grid: no grid-related emissions to add
            if model.Model_Components == 0:  # RES + generator + battery + fuel burn (off-grid)
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.GEN_emission + model.BESS_emission + model.Scenario_FUEL_emission[s])  # Off-grid full system: RES + generator + battery + fuel emissions
            if model.Model_Components == 1:  # RES + battery only (off-grid, no generator)
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.BESS_emission)  # Off-grid, no generator: RES + battery emissions only
            if model.Model_Components == 2:  # RES + generator + fuel burn (off-grid, no battery)
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.GEN_emission + model.Scenario_FUEL_emission[s])  # Off-grid, no battery: RES + generator + fuel emissions

        
    "Investment cost"
    def Investment_Cost(model):    # Total discounted CAPEX for RES + generator + battery + grid connection
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration  # Number of years in each step
        for i in range(1, len(model.steps)):  # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]  # Will map each project year to its (year, step_index) tuple
        for y in model.years:        # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls in investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])            # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))              # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]  # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[model.Step_Duration*i + model.Step_Duration]  # Upgrade at end of each step      
      
        Inv_Ren = sum((model.RES_Units[1,r]*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r])  # Step-1 RES CAPEX (undiscounted)
                        + sum((((model.RES_Units[ut,r] - model.RES_Units[ut-1,r])*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r]))/((1+model.Discount_Rate)**(yt-1))  # NPV of incremental RES added at each upgrade year
                        for (yt,ut) in tup_list) for r in model.renewable_sources)    # Sum incremental RES investment NPV over all upgrade points
        Inv_Gen = sum((model.Generator_Nominal_Capacity[1,g]*model.Generator_Specific_Investment_Cost[g])  # Step-1 generator CAPEX
                        + sum((((model.Generator_Nominal_Capacity[ut,g] - model.Generator_Nominal_Capacity[ut-1,g])*model.Generator_Specific_Investment_Cost[g]))/((1+model.Discount_Rate)**(yt-1))  # NPV of incremental generator capacity
                        for (yt,ut) in tup_list) for g in model.generator_types)    # Sum incremental generator investment NPV over all upgrade points
        Inv_Bat = ((model.Battery_Nominal_Capacity[1]*model.Battery_Specific_Investment_Cost)  # Step-1 battery CAPEX
                        + sum((((model.Battery_Nominal_Capacity[ut] - model.Battery_Nominal_Capacity[ut-1])*model.Battery_Specific_Investment_Cost))/((1+model.Discount_Rate)**(yt-1))  # NPV of incremental battery capacity
                        for (yt,ut) in tup_list))   # Sum incremental battery investment NPV over all upgrade points
        Inv_Grid = 0  # NPV of grid connection investment (summed over years grid is available)
        for y in model.years:  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only charge connection cost once grid becomes available
                Inv_Grid += (model.Grid_Connection_Cost * model.Grid_Distance)/((1+model.Discount_Rate)**(y-1))   # Discount grid connection cost to year 0 and accumulate
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            investment_cost = Inv_Ren + Inv_Gen + Inv_Bat  # Full system: RES + generator + battery CAPEX
            if model.Grid_Connection: investment_cost += Inv_Grid  # Add grid connection cost if enabled
            return model.Investment_Cost == investment_cost   # Return the assembled CAPEX constraint
        if model.Model_Components == 1:  # RES + battery only (no generator)
            investment_cost = Inv_Ren + Inv_Bat  # RES + battery CAPEX (no generator)
            if model.Grid_Connection: investment_cost += Inv_Grid  # Add grid connection cost if enabled
            return model.Investment_Cost == investment_cost        # Return the assembled CAPEX constraint
        if model.Model_Components == 2:  # RES + generator only (no battery)
            investment_cost = Inv_Ren + Inv_Gen  # RES + generator CAPEX (no battery)
            if model.Grid_Connection: investment_cost += Inv_Grid  # Add grid connection cost if enabled
            return model.Investment_Cost == investment_cost       # Return the assembled CAPEX constraint
    
    def Investment_Cost_Limit(model):  # Upper bound on total capital investment
        return model.Investment_Cost <= model.Investment_Cost_Limit  # Total CAPEX must not exceed the limit
       
    "Fixed O&M costs"
    def Operation_Maintenance_Cost_Act(model):  # Actualized (discounted) fixed O&M cost of RES + generator + battery + grid
        OyM_Ren = sum(sum((model.RES_Units[ut,r]*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r]*model.RES_Specific_OM_Cost[r])/((  # NPV of RES fixed O&M cost, discounted per year it is incurred
                        1+model.Discount_Rate)**yt)for (yt,ut) in model.years_steps)for r in model.renewable_sources)      # Sum discounted RES O&M cost over all years/steps
        OyM_Gen = sum(sum((model.Generator_Nominal_Capacity[ut,g]*model.Generator_Specific_Investment_Cost[g]*model.Generator_Specific_OM_Cost[g])/((  # NPV of generator fixed O&M cost, discounted per year it is incurred
                        1+model.Discount_Rate)**yt)for (yt,ut) in model.years_steps)for g in model.generator_types)  # Sum discounted generator O&M cost over all years/steps
        OyM_Bat = sum((model.Battery_Nominal_Capacity[ut]*model.Battery_Specific_Investment_Cost*model.Battery_Specific_OM_Cost)/((  # NPV of battery fixed O&M cost, discounted per year it is incurred
                        1+model.Discount_Rate)**yt)for (yt,ut) in model.years_steps)  # Sum discounted battery O&M cost over all years/steps
        OyM_Grid = 0  # Accumulator for grid connection O&M cost
        for y in model.years:  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur grid O&M cost once the connection year has been reached
                OyM_Grid += (model.Grid_Connection_Cost * model.Grid_Distance * model.Grid_Maintenance_Cost)/((1+model.Discount_Rate)**(y))   # Discounted grid maintenance cost, only once grid is connected
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            cost = OyM_Ren + OyM_Gen + OyM_Bat  # Full system: RES + generator + battery O&M
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_Act == cost  # Return the assembled actualized O&M constraint
        if model.Model_Components == 1:  # RES + battery only (no generator)
            cost = OyM_Ren + OyM_Bat  # RES + battery O&M (no generator)
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_Act == cost  # Return the assembled actualized O&M constraint
        if model.Model_Components == 2:  # RES + generator only (no battery)
            cost = OyM_Ren + OyM_Gen  # RES + generator O&M (no battery)
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_Act == cost   # Return the assembled actualized O&M constraint
    
    def Operation_Maintenance_Cost_NonAct(model):  # Non-actualized (undiscounted) fixed O&M cost, mirrors the Act version
        OyM_Ren = sum(sum((model.RES_Units[ut,r]*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r]*model.RES_Specific_OM_Cost[r])  # Undiscounted RES fixed O&M cost
                        for (yt,ut) in model.years_steps)for r in model.renewable_sources)      # Sum undiscounted RES O&M cost over all years/steps
        OyM_Gen = sum(sum((model.Generator_Nominal_Capacity[ut,g]*model.Generator_Specific_Investment_Cost[g]*model.Generator_Specific_OM_Cost[g])  # Undiscounted generator fixed O&M cost
                        for (yt,ut) in model.years_steps)for g in model.generator_types)  # Sum undiscounted generator O&M cost over all years/steps
        OyM_Bat = sum((model.Battery_Nominal_Capacity[ut]*model.Battery_Specific_Investment_Cost*model.Battery_Specific_OM_Cost)  # Undiscounted battery fixed O&M cost
                        for (yt,ut) in model.years_steps)  # Sum undiscounted battery O&M cost over all years/steps
        OyM_Grid = 0  # Accumulator for grid connection O&M cost
        for y in model.years:  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur grid O&M cost once the connection year has been reached
                OyM_Grid += (model.Grid_Connection_Cost * model.Grid_Distance * model.Grid_Maintenance_Cost)  # Undiscounted grid maintenance cost, only once grid is connected
        
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            cost = OyM_Ren + OyM_Gen + OyM_Bat  # Full system: RES + generator + battery O&M
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_NonAct == cost   # Return the assembled non-actualized O&M constraint
        if model.Model_Components == 1:  # RES + battery only (no generator)
            cost = OyM_Ren + OyM_Bat  # RES + battery O&M (no generator)
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_NonAct == cost   # Return the assembled non-actualized O&M constraint
        if model.Model_Components == 2:  # RES + generator only (no battery)
            cost = OyM_Ren + OyM_Gen  # RES + generator O&M (no battery)
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_NonAct == cost    # Return the assembled non-actualized O&M constraint
    
    "Variable costs"
    def Total_Variable_Cost_Act(model):  # Scalar actualized total variable cost = expected value over all scenarios
        return model.Total_Variable_Cost_Act == (sum(model.Total_Scenario_Variable_Cost_Act[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Probability-weighted sum across scenarios
    
    def Scenario_Variable_Cost_Act(model, s):  # Actualized variable cost for scenario s: O&M + battery replacement + lost load + fuel + net electricity cost
        foo = []  # Index list accumulator
        for g in range(1,model.Generator_Types+1):  # Loop over all generator types for this scenario
                foo.append((s,g))     # Collect (scenario, generator) index pair
        Fuel_Cost = sum(model.Total_Fuel_Cost_Act[s,g] for s,g in foo)     # Total actualized fuel cost across all generator types
        if model.Grid_Connection == 1:  # Grid-connected: include electricity purchase cost (and sale revenue if allowed)
            Electricity_Cost = model.Total_Electricity_Cost_Act[s]   # Actualized cost of electricity purchased from the grid
            if model.Grid_Connection_Type == 0:  # Bidirectional connection: exports earn revenue
                Electricity_Revenues = model.Total_Revenues_Act[s]  # Actualized revenue from electricity sold to the grid
            else: Electricity_Revenues = 0  # Import-only connection: no export revenue
        else:  # Off-grid: no electricity purchase or export cost
            Electricity_Cost = 0  # Off-grid: no electricity purchase cost
            Electricity_Revenues = 0  # Off-grid: no electricity sale revenue
            
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            return model.Total_Scenario_Variable_Cost_Act[s] == model.Operation_Maintenance_Cost_Act + model.Battery_Replacement_Cost_Act[s] + model.Scenario_Lost_Load_Cost_Act[s] + Fuel_Cost + Electricity_Cost - Electricity_Revenues  # Full system: O&M + battery replacement + lost load + fuel + net electricity
        if model.Model_Components == 1:  # RES + battery only (no generator)
            return model.Total_Scenario_Variable_Cost_Act[s] == model.Operation_Maintenance_Cost_Act + model.Battery_Replacement_Cost_Act[s] + model.Scenario_Lost_Load_Cost_Act[s] + Electricity_Cost - Electricity_Revenues  # No generator: O&M + battery replacement + lost load + net electricity
        if model.Model_Components == 2:  # RES + generator only (no battery)
            return model.Total_Scenario_Variable_Cost_Act[s] == model.Operation_Maintenance_Cost_Act + model.Scenario_Lost_Load_Cost_Act[s] + Fuel_Cost + Electricity_Cost - Electricity_Revenues  # No battery: O&M + lost load + fuel + net electricity
    
    def Scenario_Variable_Cost_NonAct(model, s):   # Non-actualized (undiscounted) mirror of Scenario_Variable_Cost_Act
        foo = []  # Index list accumulator
        for g in range(1,model.Generator_Types+1):  # Loop over all generator types for this scenario
            foo.append((s,g))     # Collect (scenario, generator) index pair
        Fuel_Cost = sum(model.Total_Fuel_Cost_NonAct[s,g] for s,g in foo)    # Total undiscounted fuel cost across all generator types
        if model.Grid_Connection == 1:  # Grid-connected: include electricity purchase cost (and sale revenue if allowed)
            Electricity_Cost = model.Total_Electricity_Cost_NonAct[s]   # Undiscounted cost of electricity purchased from the grid
            if model.Grid_Connection_Type == 0:  # Bidirectional connection: exports earn revenue
                Electricity_Revenues = model.Total_Revenues_NonAct[s]  # Undiscounted revenue from electricity sold to the grid
            else: Electricity_Revenues = 0  # Import-only connection: no export revenue
        else:  # Off-grid: no electricity purchase or export cost
            Electricity_Cost = 0  # Off-grid: no electricity purchase cost
            Electricity_Revenues = 0  # Off-grid: no electricity sale revenue
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            return model.Total_Scenario_Variable_Cost_NonAct[s] == model.Operation_Maintenance_Cost_NonAct + model.Battery_Replacement_Cost_NonAct[s] + model.Scenario_Lost_Load_Cost_NonAct[s] + Fuel_Cost + Electricity_Cost - Electricity_Revenues  # Full system: O&M + battery replacement + lost load + fuel + net electricity
        if model.Model_Components == 1:  # RES + battery only (no generator)
            return model.Total_Scenario_Variable_Cost_NonAct[s] == model.Operation_Maintenance_Cost_NonAct + model.Battery_Replacement_Cost_NonAct[s] + model.Scenario_Lost_Load_Cost_NonAct[s] + Electricity_Cost - Electricity_Revenues  # No generator: O&M + battery replacement + lost load + net electricity
        if model.Model_Components == 2:  # RES + generator only (no battery)
            return model.Total_Scenario_Variable_Cost_NonAct[s] == model.Operation_Maintenance_Cost_NonAct + model.Scenario_Lost_Load_Cost_NonAct[s] + Fuel_Cost + Electricity_Cost - Electricity_Revenues  # No battery: O&M + lost load + fuel + net electricity
    
    def Scenario_Lost_Load_Cost_Act(model,s):      # Actualized cost of unserved (lost load) energy for scenario s
        Cost_Lost_Load = 0           # Accumulator for discounted yearly lost-load cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(model.Lost_Load[s,y,t]*model.Lost_Load_Specific_Cost for t in model.periods)  # Yearly lost-load cost = unserved energy x penalty price
            Cost_Lost_Load += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
        return  model.Scenario_Lost_Load_Cost_Act[s] == Cost_Lost_Load  # Return actualized lost-load cost constraint
    
    def Scenario_Lost_Load_Cost_NonAct(model,s):  # Non-actualized (undiscounted) mirror of Scenario_Lost_Load_Cost_Act
        Cost_Lost_Load = 0           # Accumulator for discounted yearly lost-load cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(model.Lost_Load[s,y,t]*model.Lost_Load_Specific_Cost for t in model.periods)  # Yearly lost-load cost = unserved energy x penalty price
            Cost_Lost_Load += Num  # Accumulate undiscounted yearly lost-load cost
        return  model.Scenario_Lost_Load_Cost_NonAct[s] == Cost_Lost_Load  # Return non-actualized lost-load cost constraint
    
    if Fuel_Specific_Cost_Calculation == 0:  # Constant (year-independent) fuel marginal cost
        def Total_Fuel_Cost_Act(model,s,g):  # Actualized fuel cost for generator type g across the project horizon
            Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
            for y in range(1, model.Years +1):  # Loop over every project year
                Num = sum(model.Generator_Energy_Production[s,y,g,t]*model.Generator_Marginal_Cost_1[g] for t in model.periods)  # Yearly fuel cost = energy produced x constant marginal cost
                Fuel_Cost_Tot += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
            return model.Total_Fuel_Cost_Act[s,g] == Fuel_Cost_Tot  # Return actualized fuel cost constraint
        
        def Total_Fuel_Cost_NonAct(model,s,g):  # Non-actualized (undiscounted) mirror of Total_Fuel_Cost_Act
            Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
            for y in range(1, model.Years +1):  # Loop over every project year
                Num = sum(model.Generator_Energy_Production[s,y,g,t]*model.Generator_Marginal_Cost_1[g] for t in model.periods)  # Yearly fuel cost = energy produced x constant marginal cost
                Fuel_Cost_Tot += Num  # Accumulate undiscounted yearly fuel cost
            return model.Total_Fuel_Cost_NonAct[s,g] == Fuel_Cost_Tot  # Return non-actualized fuel cost constraint
    else:  # Time-varying (year-specific) fuel price mode
        def Total_Fuel_Cost_Act(model,s,g):  # Actualized fuel cost for generator type g across the project horizon
            Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
            for y in range(1, model.Years +1):  # Loop over every project year
                Num = sum(model.Generator_Energy_Production[s,y,g,t]*model.Generator_Marginal_Cost[g,y] for t in model.periods)  # Yearly fuel cost = energy produced x year-specific marginal cost
                Fuel_Cost_Tot += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
            return model.Total_Fuel_Cost_Act[s,g] == Fuel_Cost_Tot  # Return actualized fuel cost constraint
        
        def Total_Fuel_Cost_NonAct(model,s,g):  # Non-actualized (undiscounted) mirror of Total_Fuel_Cost_Act
            Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
            for y in range(1, model.Years +1):  # Loop over every project year
                Num = sum(model.Generator_Energy_Production[s,y,g,t]*model.Generator_Marginal_Cost[g,y] for t in model.periods)  # Yearly fuel cost = energy produced x year-specific marginal cost
                Fuel_Cost_Tot += Num  # Accumulate undiscounted yearly fuel cost
            return model.Total_Fuel_Cost_NonAct[s,g] == Fuel_Cost_Tot  # Return non-actualized fuel cost constraint

    
    def Total_Electricity_Cost_Act(model,s):   # Actualized cost of electricity purchased from the national grid
        Electricity_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly electricity cost
        for y in range(1, model.Years +1):  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur grid purchase cost once the connection year has been reached
                Num = sum(model.Energy_From_Grid[s,y,t]*model.Grid_Availability[s,y,t]*model.Grid_Purchased_El_Price  for t in model.periods)  # Yearly cost = grid imports (masked by availability) x purchase price / 1000
                Electricity_Cost_Tot += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
        return model.Total_Electricity_Cost_Act[s] == Electricity_Cost_Tot  # Return actualized electricity cost constraint
       
    def Total_Electricity_Cost_NonAct(model,s):   # Non-actualized (undiscounted) mirror of Total_Electricity_Cost_Act
        Electricity_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly electricity cost
        for y in range(1, model.Years +1):  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur grid purchase cost once the connection year has been reached
                Electricity_Cost_Tot += sum(model.Energy_From_Grid[s,y,t]*model.Grid_Availability[s,y,t]*model.Grid_Purchased_El_Price  for t in model.periods)  # Accumulate undiscounted yearly electricity cost
        return model.Total_Electricity_Cost_NonAct[s] == Electricity_Cost_Tot  # Return non-actualized electricity cost constraint
    
    
    def Total_Revenues_NonAct(model, s):   # Undiscounted revenue from electricity exported to the grid
        Revenues_Yearly = [0 for _ in model.years]  # Per-year export revenue, zero before grid connection
        for y in range(1, model.Years + 1):  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only count grid export revenue once the connection year has been reached
                Revenues_Yearly[y - 1] = sum(model.Energy_To_Grid[s, y, t] * model.Grid_Availability[s, y, t] * model.Grid_Sold_El_Price  for t in model.periods)  # Yearly revenue = grid exports (masked by availability) x sale price / 1000
        return model.Total_Revenues_NonAct[s] == sum(Revenues_Yearly[y - 1] for y in model.years)  # Sum undiscounted yearly revenues

    def Total_Revenues_Act(model, s):   # Actualized (discounted) revenue from electricity exported to the grid
        Revenues_Yearly = [0 for _ in model.years]  # Per-year export revenue, zero before grid connection
        for y in range(1, model.Years + 1):  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only count grid export revenue once the connection year has been reached
                Revenues_Yearly[y - 1] = sum(model.Energy_To_Grid[s, y, t] * model.Grid_Availability[s, y, t] * model.Grid_Sold_El_Price  for t in model.periods)  # Yearly revenue = grid exports (masked by availability) x sale price / 1000
        return model.Total_Revenues_Act[s] == sum(Revenues_Yearly[y - 1] / ((1 + model.Discount_Rate) ** y) for y in model.years)  # Discount each year's export revenue to year 0 and sum
    
    def Battery_Replacement_Cost_Act(model,s):  # Actualized cost of replacing battery cells as they cycle (charge/discharge wear)
        Battery_cost_in = [0 for y in model.years]  # Per-year cost attributable to charging throughput
        Battery_cost_out = [0 for y in model.years]  # Per-year cost attributable to discharging throughput
        Battery_Yearly_cost = [0 for y in model.years]      # Per-year total replacement cost (charge + discharge)
        for y in range(1,model.Years+1):      # Loop over every project year
            Battery_cost_in[y-1] = sum(model.Battery_Inflow[s,y,t]*model.Unitary_Battery_Replacement_Cost for t in model.periods)  # Charging energy x unitary replacement cost, summed over the year
            Battery_cost_out[y-1] = sum(model.Battery_Outflow[s,y,t]*model.Unitary_Battery_Replacement_Cost for t in model.periods)  # Discharging energy x unitary replacement cost, summed over the year
            Battery_Yearly_cost[y-1] = Battery_cost_out[y-1] + Battery_cost_in[y-1]  # Total yearly replacement cost = charge + discharge wear
        return model.Battery_Replacement_Cost_Act[s] == sum(Battery_Yearly_cost[y-1]/((1+model.Discount_Rate)**y) for y in model.years)   # Discount each year's replacement cost to year 0 and sum
        
    def Battery_Replacement_Cost_NonAct(model,s):  # Non-actualized (undiscounted) mirror of Battery_Replacement_Cost_Act
        Battery_cost_in = [0 for y in model.years]  # Per-year cost attributable to charging throughput
        Battery_cost_out = [0 for y in model.years]  # Per-year cost attributable to discharging throughput
        Battery_Yearly_cost = [0 for y in model.years]      # Per-year total replacement cost (charge + discharge)
        for y in range(1,model.Years+1):      # Loop over every project year
            Battery_cost_in[y-1] = sum(model.Battery_Inflow[s,y,t]*model.Unitary_Battery_Replacement_Cost for t in model.periods)  # Charging energy x unitary replacement cost, summed over the year
            Battery_cost_out[y-1] = sum(model.Battery_Outflow[s,y,t]*model.Unitary_Battery_Replacement_Cost for t in model.periods)  # Discharging energy x unitary replacement cost, summed over the year
            Battery_Yearly_cost[y-1] = Battery_cost_out[y-1] + Battery_cost_in[y-1]  # Total yearly replacement cost = charge + discharge wear
        return model.Battery_Replacement_Cost_NonAct[s] == sum(Battery_Yearly_cost[y-1] for y in model.years)  # Sum undiscounted yearly replacement cost
    
    
    "Salvage Value"
    def _Salvage_Value_Raw(model):     # Residual (undepreciated) value of all assets at project end, discounted to year 0 -- NOT floored at zero, see Salvage_Value_Upper_Raw/Flag below
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration  # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur  # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]  # Will map each project year to its (year, step_index) tuple
        for y in model.years:      # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])          # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))            # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]  # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[s_dur*i + s_dur]  # Upgrade decision occurs at the end of each step

        if model.Steps_Number == 1:      # Single investment step: only the initial build can have residual value
            SV_Ren_1 = sum(model.RES_Units[1,r]*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]-model.Years)/model.RES_Lifetime[r] /   # Residual value of step-1 RES capacity (no pre-existing capacity to credit)
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)          # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_2 = 0   # No second-step RES capacity exists yet
            SV_Ren_3 = 0      # No later-step RES capacity exists yet
            SV_Gen_1 = sum(model.Generator_Nominal_Capacity[1,g]*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]-model.Years)/model.Generator_Lifetime[g] /   # Residual value of step-1 generator capacity (no pre-existing capacity to credit)
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)          # Discount generator residual value to year 0, summed over generator types
            SV_Gen_2 = 0  # No second-step generator capacity exists yet
            SV_Gen_3 = 0  # No later-step generator capacity exists yet
            SV_Grid = model.Grid_Distance*model.Grid_Connection_Cost*model.Grid_Connection / ((1 + model.Discount_Rate)**(model.Years - model.Year_Grid_Connection))  # Discounted residual value of the grid connection investment

        if model.Steps_Number == 2:      # Two investment steps: initial build + one upgrade can have residual value
            yt_last_up = upgrade_years_list[1]         # Year of the (only) capacity upgrade
            SV_Ren_1 = sum(model.RES_Units[1,r]*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]-model.Years)/model.RES_Lifetime[r] /   # Residual value of step-1 RES capacity (no pre-existing capacity to credit)
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)          # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_2 = sum((model.RES_Units[2,r]-model.RES_Units[1,r])*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]+(yt_last_up-1)-model.Years)/model.RES_Lifetime[r] /   # Residual value of the step-2 RES capacity increment
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)          # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_3 = 0      # No later-step RES capacity exists yet
            SV_Gen_1 = sum(model.Generator_Nominal_Capacity[1,g]*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]-model.Years)/model.Generator_Lifetime[g] /   # Residual value of step-1 generator capacity (no pre-existing capacity to credit)
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)          # Discount generator residual value to year 0, summed over generator types
            SV_Gen_2 = sum((model.Generator_Nominal_Capacity[2,g]-model.Generator_Nominal_Capacity[1,g])*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]+(yt_last_up-1)-model.Years)/model.Generator_Lifetime[g] /   # Residual value of the step-2 generator capacity increment
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)  # Discount generator residual value to year 0, summed over generator types
            SV_Gen_3 = 0  # No later-step generator capacity exists yet
            SV_Grid = model.Grid_Distance*model.Grid_Connection_Cost*model.Grid_Connection / ((1 + model.Discount_Rate)**(model.Years - model.Year_Grid_Connection))   # Discounted residual value of the grid connection investment
        if model.Steps_Number > 2:  # Three or more steps: initial build + last upgrade + all intermediate upgrades
            tup_list_2 = [[] for i in range(len(model.steps)-2)]  # (year, step) pairs for upgrades before the last one
            for i in range(len(model.steps) - 2):  # Build the intermediate-upgrade list (excludes the final upgrade)
                tup_list_2[i] = yu_tuples_list[s_dur*i + s_dur]  # Intermediate upgrade occurs at the end of each step
            yt_last_up = upgrade_years_list[-1]  # Year of the final capacity upgrade
            ut_last_up = tup_list[-1][1]      # Step index of the final upgrade
            ut_seclast_up = tup_list[-2][1]  # Step index of the second-to-last upgrade

            SV_Ren_1 = sum(model.RES_Units[1,r]*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]-model.Years)/model.RES_Lifetime[r] /   # Residual value of step-1 RES capacity (no pre-existing capacity to credit)
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)      # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_2 = sum((model.RES_Units[ut_last_up,r] - model.RES_Units[ut_seclast_up,r])*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]+(yt_last_up-1)-model.Years)/model.RES_Lifetime[r] /   # Residual value of the final RES capacity increment
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)  # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_3 = sum(sum((model.RES_Units[ut,r] - model.RES_Units[ut-1,r])*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]+(yt-1)-model.Years)/model.RES_Lifetime[r] /   # Residual value of all intermediate RES capacity increments
                            ((1+model.Discount_Rate)**model.Years) for (yt,ut) in tup_list_2) for r in model.renewable_sources)          # Sum intermediate-upgrade RES residual value, discounted to year 0
            SV_Gen_1 = sum(model.Generator_Nominal_Capacity[1,g]*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]-model.Years)/model.Generator_Lifetime[g] /   # Residual value of step-1 generator capacity (no pre-existing capacity to credit)
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)  # Discount generator residual value to year 0, summed over generator types
            SV_Gen_2 = sum((model.Generator_Nominal_Capacity[ut_last_up,g] - model.Generator_Nominal_Capacity[ut_seclast_up,g])*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]+(yt_last_up-1)-model.Years)/model.Generator_Lifetime[g] /   # Residual value of the final generator capacity increment
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)  # Discount generator residual value to year 0, summed over generator types
            SV_Gen_3 = sum(sum((model.Generator_Nominal_Capacity[ut,g] - model.Generator_Nominal_Capacity[ut-1,g])*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]+(yt-1)-model.Years)/model.Generator_Lifetime[g] /   # Residual value of all intermediate generator capacity increments
                            ((1+model.Discount_Rate)**model.Years) for (yt,ut) in tup_list_2) for g in model.generator_types)  # Sum intermediate-upgrade generator residual value, discounted to year 0
            SV_Grid = model.Grid_Distance*model.Grid_Connection_Cost*model.Grid_Connection / ((1 + model.Discount_Rate)**(model.Years - model.Year_Grid_Connection))   # Discounted residual value of the grid connection investment

        if model.Model_Components == 0 or model.Model_Components == 2:  # System includes a generator: credit RES + generator + grid salvage value
            return SV_Ren_1 + SV_Gen_1 + SV_Ren_2 + SV_Gen_2 + SV_Ren_3 + SV_Gen_3 + SV_Grid   # Sum residual value of all RES/generator vintages plus grid connection
        if model.Model_Components == 1:  # RES + battery only (no generator)
            return SV_Ren_1 + SV_Ren_2 + SV_Ren_3 + SV_Grid   # No generator: sum residual value of RES vintages plus grid connection

    def Salvage_Value_Upper_Raw(model):   # Floor-at-zero linearization (2026-08-28): Salvage_Value <= raw formula, but only enforced when Salvage_Positive_Flag=1
        BIG_M_SALVAGE = 2.0e6  # tightened 2026-08-29: anchored to Generator_Max_Units's hard cap (9400kW x $160.58/kW = ~$1.51M for this project's designs) plus headroom -- still ~230x the only real negative-salvage magnitude observed (-$8,538). RES has no formal capacity ceiling in this model, so this isn't a rigorous proof-bound, just a much tighter one than the earlier 5.0e6/2.0e7 -- an oversized Big-M here visibly degrades barrier/crossover conditioning (RHS/matrix range balloons relative to the rest of the model's coefficients)
        return model.Salvage_Value <= Constraints_Greenfield._Salvage_Value_Raw(model) + BIG_M_SALVAGE * (1 - model.Salvage_Positive_Flag)

    def Salvage_Value_Upper_Flag(model):   # Floor-at-zero linearization: Salvage_Value <= 0, forced whenever Salvage_Positive_Flag=0 (i.e. whenever the raw formula is negative)
        BIG_M_SALVAGE = 2.0e6
        return model.Salvage_Value <= BIG_M_SALVAGE * model.Salvage_Positive_Flag

    def Energy_balance(model,s,yt,ut,t): # Energy balance: supply == demand each hour
        Foo = []  # Index list for RES energy production in this period
        for r in model.renewable_sources:  # Loop over all renewable source types
            Foo.append((s,yt,r,t))      # Collect index for this scenario/year/source/time
        Total_Renewable_Energy = sum(model.RES_Energy_Production[j] for j in Foo)  # Sum of all RES output
        foo=[]  # Index list for generator energy production in this period
        for g in model.generator_types:  # Loop over all generator types
            foo.append((s,yt,g,t))      # Collect index for this scenario/year/generator/time
        Total_Generator_Energy = sum(model.Generator_Energy_Production[i] for i in foo)  # Sum of all generator output
        if model.Grid_Connection == 1:  # Grid-connected: allow energy exchange with the grid
            En_From_Grid = model.Energy_From_Grid[s,yt,t]*model.Grid_Availability[s,yt,t]  # Grid imports masked by outage availability
            if model.Grid_Connection_Type == 0:  # Bidirectional connection: exports earn revenue
                En_To_Grid = model.Energy_To_Grid[s,yt,t]*model.Grid_Availability[s,yt,t]  # Grid exports (only when grid is up)
            else: En_To_Grid = 0  # Export not allowed for this connection type
        else:  # Off-grid: no grid imports or exports
            En_From_Grid = 0  # Off-grid: no imports
            En_To_Grid = 0    # Off-grid: no exports
       
        if model.Model_Components == 0:  # Demand = RES + generator + grid import - export + bat.discharge - bat.charge + lost_load - curtailment
           return model.Energy_Demand[s,yt,t] == (Total_Renewable_Energy   # Full system: demand = RES + generator + grid import/export + battery +/- lost load/curtailment
                                                   + Total_Generator_Energy  # Add generator output
                                                   + En_From_Grid  # Add grid imports
                                                   - En_To_Grid   # Subtract grid exports
                                                   + model.Battery_Outflow[s,yt,t]  # Add battery discharge
                                                   - model.Battery_Inflow[s,yt,t]   # Subtract battery charge
                                                   + model.Lost_Load[s,yt,t]    # Add unserved (lost load) energy
                                                   - model.Energy_Curtailment[s,yt,t] )       # Subtract curtailed (wasted) energy
        if model.Model_Components == 1:  # No generator
           return model.Energy_Demand[s,yt,t] == (Total_Renewable_Energy   # Full system: demand = RES + generator + grid import/export + battery +/- lost load/curtailment
                                                   + En_From_Grid  # Add grid imports
                                                   - En_To_Grid   # Subtract grid exports
                                                   + model.Battery_Outflow[s,yt,t]  # Add battery discharge
                                                   - model.Battery_Inflow[s,yt,t]   # Subtract battery charge
                                                   + model.Lost_Load[s,yt,t]    # Add unserved (lost load) energy
                                                   - model.Energy_Curtailment[s,yt,t] )       # Subtract curtailed (wasted) energy
        if model.Model_Components == 2:  # No battery (no charge/discharge terms)
           return model.Energy_Demand[s,yt,t] == (Total_Renewable_Energy   # Full system: demand = RES + generator + grid import/export + battery +/- lost load/curtailment
                                                   + Total_Generator_Energy  # Add generator output
                                                   + En_From_Grid  # Add grid imports
                                                   - En_To_Grid   # Subtract grid exports
                                                   + model.Lost_Load[s,yt,t]    # Add unserved (lost load) energy
                                                   - model.Energy_Curtailment[s,yt,t] )       # Subtract curtailed (wasted) energy
    
    
    "Renewable Energy Sources constraints"
    def Renewable_Energy(model,s,yt,ut,r,t): # Energy output of the RES
        return model.RES_Energy_Production[s,yt,r,t] == model.RES_Unit_Energy_Production[s,r,t]*model.RES_Inverter_Efficiency[r]*model.RES_Units[ut,r]  # Unit yield x inverter efficiency x number of units
    
    def Renewable_Energy_Penetration(model,ut):      # Enforce minimum renewable energy penetration target for step ut
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration  # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur  # Start year of step i+1
        yu_tuples_list = [0 for i in model.years]  # Will map each project year to its (year, step_index) tuple
        if model.Steps_Number == 1:  # Single investment step: only the initial build can have residual value
            for y in model.years:          # Loop over every project year
                yu_tuples_list[y-1] = (y, 1)  # Single-step case: every year maps to step 1
        else:      # Multiple investment steps: determine which step each year belongs to
            for y in model.years:          # Loop over every project year
                for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                    if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                        yu_tuples_list[y-1] = (y, model.steps[i+1])              # Record (year, step) pair
                    elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                        yu_tuples_list[y-1] = (y, len(model.steps))         # Record (year, final step) pair
        years_list = []  # Project years that belong to investment step ut
        for i in yu_tuples_list:  # Loop over all (year, step) pairs
            if i[1]==ut:  # Keep only the years belonging to the current step ut
                years_list += [i[0]]              # Record this year
        Foo=[]  # Index list accumulator
        for s in range(1, model.Scenarios + 1):  # Loop over all scenarios
            for y in years_list:  # Loop over years belonging to step ut
                for g in range(1, model.Generator_Types+1):  # Loop over all generator types
                    for t in range(1,model.Periods+1):  # Loop over all time periods
                        Foo.append((s,y,g,t))                          # Collect index for this scenario/year/generator/time
        foo=[]  # Index list for generator energy production in this period
        for s in range(1, model.Scenarios + 1):  # Loop over all scenarios
            for y in years_list:  # Loop over years belonging to step ut
                for r in range(1, model.RES_Sources+1):  # Loop over all RES sources
                    for t in range(1,model.Periods+1):  # Loop over all time periods
                        foo.append((s,y,r,t))   # Collect index for this scenario/year/source/time
        goo=[]  # Index list for grid import in this period
        for s in range(1, model.Scenarios + 1):  # Loop over all scenarios
            for y in years_list:  # Loop over years belonging to step ut
                    for t in range(1,model.Periods+1):  # Loop over all time periods
                        goo.append((s,y,t))  # Collect index for this scenario/year/time
                        
        E_gen = sum(model.Generator_Energy_Production[s,y,g,t]*model.Scenario_Weight[s]  # Probability-weighted generator energy over step ut
                    for s,y,g,t in Foo)  # Sum over the collected scenario/year/generator/time indices
        E_ren = sum(model.RES_Energy_Production[s,y,r,t]*model.Scenario_Weight[s]  # Probability-weighted RES energy over step ut
                    for s,y,r,t in foo)  # Sum over the collected scenario/year/source/time indices
        if model.Grid_Connection == 1:  # Grid-connected: include grid imports in the penetration calculation
            E_From_Grid = sum(model.Energy_From_Grid[s,y,t]*model.Grid_Availability[s,y,t]*model.Scenario_Weight[s]  # Probability-weighted grid imports over step ut
                    for s,y,t in goo)  # Sum over the collected scenario/year/time indices
        else: E_From_Grid = 0  # Off-grid: no imports to include in penetration calculation
 
        return  (1 - model.Renewable_Penetration)*E_ren >= model.Renewable_Penetration*(E_gen + E_From_Grid)  # RES share of total energy supplied must meet the penetration target
    
    def Renewables_Min_Step_Units(model,yt,ut,r):  # RES units must not decrease between investment steps (no decommissioning)
        if ut > 1:  # After the first step: value must not decrease from the previous step
            return model.RES_Units[ut,r] >= model.RES_Units[ut-1,r]  # RES units must not decrease from the previous step
        elif ut == 1:  # First step: no prior step to compare against
            return model.RES_Units[ut,r] == model.RES_Units[ut,r]  # First step: trivially satisfied (no previous step)
    
    "Maximum land use constraint for Renewables"
    def Renewables_Max_Land_Use(model,ut):  # Total land footprint of all RES must not exceed available area
        return (sum((model.RES_Units[ut,r]*model.RES_Nominal_Capacity[r]*(model.RES_Specific_Area[r])) for r in model.renewable_sources)) <= model.Renewables_Total_Area  # Total land footprint of installed RES units must not exceed available area
    
    "Battery Energy Storage constraints"
    def State_of_Charge(model,s,yt,ut,t): # State of Charge of the battery
        if t==1 and yt==1: # The state of charge (State_Of_Charge) for the period 0 is equal to the Battery size.
            return model.Battery_SOC[s,yt,t] == model.Battery_Nominal_Capacity[ut]*model.Battery_Initial_SOC - model.Battery_Outflow[s,yt,t]/model.Battery_Discharge_Battery_Efficiency + model.Battery_Inflow[s,yt,t]*model.Battery_Charge_Battery_Efficiency  # Initial SOC = starting fraction of capacity, adjusted for first-period flows
        if t==1 and yt!=1:  # First period of a year after the first project year
            return model.Battery_SOC[s,yt,t] == model.Battery_SOC[s,yt-1,model.Periods] - model.Battery_Outflow[s,yt,t]/model.Battery_Discharge_Battery_Efficiency + model.Battery_Inflow[s,yt,t]*model.Battery_Charge_Battery_Efficiency  # Carry over SOC from the last period of the previous year
        else:    # Any period after the first: carry over SOC from the previous period
            return model.Battery_SOC[s,yt,t] == model.Battery_SOC[s,yt,t-1] - model.Battery_Outflow[s,yt,t]/model.Battery_Discharge_Battery_Efficiency + model.Battery_Inflow[s,yt,t]*model.Battery_Charge_Battery_Efficiency      # Carry over SOC from the previous period within the same year
    
    def Maximum_Charge(model,s,yt,ut,t): # Maximun state of charge of the Battery
        return model.Battery_SOC[s,yt,t] <= model.Battery_Nominal_Capacity[ut]  # SOC cannot exceed battery capacity
    
    def Minimum_Charge(model,s,yt,ut,t): # Minimun state of charge
        return model.Battery_SOC[s,yt,t] >= model.Battery_Nominal_Capacity[ut]*(1-model.Battery_Depth_of_Discharge)  # SOC cannot fall below the minimum allowed by the depth of discharge
    
    def Max_Power_Battery_Charge(model,ut):  # Max charge power = capacity / min charge time (C-rate limit)
        return model.Battery_Maximum_Charge_Power[ut] == model.Battery_Nominal_Capacity[ut]/model.Maximum_Battery_Charge_Time  # Max charge power = capacity / minimum charge time
    
    def Max_Power_Battery_Discharge(model,ut):  # Max discharge power = capacity / min discharge time (C-rate limit)
        return model.Battery_Maximum_Discharge_Power[ut] == model.Battery_Nominal_Capacity[ut]/model.Maximum_Battery_Discharge_Time  # Max discharge power = capacity / minimum discharge time
    
    def Max_Bat_flow_in(model,s,yt,ut,t): # Minimun flow of energy for the charge fase
        return model.Battery_Inflow[s,yt,t] <= model.Battery_Maximum_Charge_Power[ut]*model.Delta_Time  # Cap charging power at the maximum charge rate
    
    def Max_Bat_flow_out(model,s,yt,ut,t): # Minimun flow of energy for the discharge fase
        return model.Battery_Outflow[s,yt,t] <= model.Battery_Maximum_Discharge_Power[ut]*model.Delta_Time  # Discharge power cannot exceed the maximum discharge rate
    
    def Max_Bat_out(model,s,yt,ut,t):  # Discharge cannot exceed hourly demand
        return model.Battery_Outflow[s,yt,t] <= model.Energy_Demand[s,yt,t]  # Discharge cannot exceed current demand
        
    def Battery_Min_Capacity(model,ut):  # Battery must meet the minimum autonomy requirement    
        return   model.Battery_Nominal_Capacity[ut] >= model.Battery_Min_Capacity[ut]  # Battery capacity must meet the minimum autonomy requirement
    
    def Battery_Min_Step_Capacity(model,yt,ut):  # Battery capacity must not fall between investment steps    
        if ut > 1:  # After the first step: value must not decrease from the previous step
            return model.Battery_Nominal_Capacity[ut] >= model.Battery_Nominal_Capacity[ut-1]  # Battery capacity must not decrease from the previous step
        elif ut == 1:  # First step: no prior step to compare against
            return model.Battery_Nominal_Capacity[ut] == model.Battery_Nominal_Capacity[ut]  # First step: trivially satisfied (no previous step)
    
    def Battery_Single_Flow_Discharge(model,s,yt,ut,t):  # Binary prevents simultaneous charge and discharge
        return   model.Battery_Outflow[s,yt,t] <= model.Single_Flow_BESS[s,yt,t]*model.Battery_Maximum_Discharge_Power[ut]*model.Delta_Time  # Discharge allowed only when the discharge-mode binary is active
    
    def Battery_Single_Flow_Charge(model,s,yt,ut,t):  # Charge only allowed when not discharging (complementary binary)
        return   model.Battery_Inflow[s,yt,t] <= (1-model.Single_Flow_BESS[s,yt,t])*model.Battery_Maximum_Charge_Power[ut]*model.Delta_Time  # Charge allowed only when not in discharge mode

    
    "Diesel generator constraints"
    def Maximum_Generator_Energy_1(model,s,yt,ut,g,t):  # LP generator output <= installed continuous capacity x time step
        return model.Generator_Energy_Production[s,yt,g,t] <= model.Generator_Nominal_Capacity[ut,g]*model.Delta_Time  # Cap generator output at installed nominal capacity
    
    def Maximum_Generator_Energy_2(model,s,yt,ut,g,t):  # Generator cannot supply more than the current demand
        return model.Generator_Energy_Production[s,yt,g,t] <= model.Energy_Demand[s,yt,t]*model.Delta_Time  # Generator output cannot exceed demand
    
    def Generator_Min_Step_Capacity(model,yt,ut,g):  # Capacity must not shrink between steps (no decommissioning in LP)
        if ut > 1:  # After the first step: value must not decrease from the previous step
            return model.Generator_Nominal_Capacity[ut,g] >= model.Generator_Nominal_Capacity[ut-1,g]  # Generator capacity must not decrease from the previous step
        elif ut ==1:  # First step: no prior step to compare against
            return model.Generator_Nominal_Capacity[ut,g] == model.Generator_Nominal_Capacity[ut,g]  # First step: trivially satisfied (no previous step)
    
    "Lost load constraints"
    def Maximum_Lost_Load(model,s,yt): # Maximum admittable lost load
        # Algebraically identical to the original form, which was written as
        #   Lost_Load_Fraction >= sum(Lost_Load) / sum(Energy_Demand)
        # but multiplied through by sum(Energy_Demand) instead of dividing by it.
        # Total annual demand is a constant, so moving it to the right-hand side
        # turns it into an RHS value rather than a matrix coefficient.
        #
        # WHY THIS MATTERS (measured 2026-07-26, not inferred): the divided form put
        # a coefficient of 1/sum(Energy_Demand) on every Lost_Load variable. With
        # demand in Wh that is ~9.9e-09 -- the SMALLEST coefficient anywhere in the
        # model, and the reason the logged matrix range reached [1e-08, 3e+06], a
        # ratio of 3.2e14 against Gurobi's guidance of below 1e9. It came from just
        # 30 rows out of 3.15 million. In this form the Lost_Load coefficients are
        # 1.0, and the worst remaining coefficient is 4.19e-06 in
        # BatteryReplacementCostAct -- roughly a 420x narrowing of the range.
        # Denominator is a sum of demands and is always strictly positive, so the
        # inequality direction is preserved.
        return sum(model.Lost_Load[s,yt,t] for t in model.periods) <= model.Lost_Load_Fraction * sum(model.Energy_Demand[s,yt,t] for t in model.periods)  # Lost load must not exceed the allowed fraction of total demand
    
    "Emission constraints"
    def RES_emission(model): #LCA emissions of RES
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration   # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]    # Will map each project year to its (year, step_index) tuple
        for y in model.years:        # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])            # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))              # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]    # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[model.Step_Duration*i + model.Step_Duration]    # Upgrade decision occurs at the end of each step
            
        return model.RES_emission == sum(model.RES_unit_CO2_emission[r]*model.RES_Units[1,r]*model.RES_Nominal_Capacity[r] for r in model.renewable_sources)+sum(sum(model.RES_unit_CO2_emission[r]*(model.RES_Units[ut,r]-model.RES_Units[ut-1,r])*model.RES_Nominal_Capacity[r] for (yt,ut) in tup_list) for r in model.renewable_sources)  # LCA emissions from step-1 RES installation (no pre-existing capacity to credit)
    
    def GEN_emission(model): #LCA emissions of generator
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration   # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]    # Will map each project year to its (year, step_index) tuple
        for y in model.years:        # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])            # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))              # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]    # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[model.Step_Duration*i + model.Step_Duration]  # Upgrade decision occurs at the end of each step
            
        return model.GEN_emission == sum(model.Generator_Nominal_Capacity[1,g]*model.GEN_unit_CO2_emission[g] for g in model.generator_types)+sum(sum((model.Generator_Nominal_Capacity[ut,g]-model.Generator_Nominal_Capacity[ut-1,g])*model.GEN_unit_CO2_emission[g] for (yt,ut) in tup_list) for g in model.generator_types)  # LCA emissions = step-1 capacity emissions + incremental capacity emissions at each upgrade
    
    def FUEL_emission(model,s,yt,ut,g,t): #Emissions from fuel consumption
        return model.FUEL_emission[s,yt,g,t] == model.Generator_Energy_Production[s,yt,g,t]/model.Fuel_LHV[g]/model.Generator_Efficiency[g]*model.FUEL_unit_CO2_emission[g]   # Emissions = energy / (fuel LHV x generator efficiency) x fuel emission factor
    
    def GRID_emission(model, s, y, t):  # Emissions from grid electricity imports, using the national grid CO2 factor
        if y >= model.Year_Grid_Connection:  # Only count grid emissions once the connection year has been reached
            return model.GRID_emission[s,y,t] == model.Energy_From_Grid[s,y,t] * model.National_Grid_Specific_CO2_emissions  # Import energy x national grid CO2 factor / 1000
        else:  # Before the connection year: no grid emissions
            return model.GRID_emission[s, y, t] == 0  # No grid emissions before the connection year
     
    def BESS_emission(model): #LCA emissions of battery
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration   # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]    # Will map each project year to its (year, step_index) tuple
        for y in model.years:        # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])            # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))              # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]    # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[model.Step_Duration*i + model.Step_Duration]  # Upgrade decision occurs at the end of each step
            
        return model.BESS_emission == model.Battery_Nominal_Capacity[1]*model.BESS_unit_CO2_emission+sum((model.Battery_Nominal_Capacity[ut]-model.Battery_Nominal_Capacity[ut-1])*model.BESS_unit_CO2_emission for (yt,ut) in tup_list)  # LCA emissions = step-1 capacity emissions + incremental capacity emissions at each upgrade
        
    def Scenario_FUEL_emission(model,s):   # Total fuel-burn CO2 emissions for scenario s, summed over years/generators/periods
        return model.Scenario_FUEL_emission[s] == sum(sum(sum(model.Generator_Energy_Production[s,y,g,t]/model.Fuel_LHV[g]/model.Generator_Efficiency[g]*model.FUEL_unit_CO2_emission[g]  for t in model.periods) for y in model.years) for g in model.generator_types)   # Sum fuel emissions over all periods, years and generator types
    
    def Scenario_GRID_emission(model, s):  # Total grid-import CO2 emissions for scenario s, summed over years/periods
        Total_Grid_Emission = 0  # Accumulator for total grid emissions across all years
        for y in range(1, model.Years + 1):  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only count grid emissions once the connection year has been reached
                Total_Grid_Emission += sum(model.Energy_From_Grid[s, y, t] * model.National_Grid_Specific_CO2_emissions  for t in model.periods)  # Yearly grid emissions = imports x national grid CO2 factor / 1000
        return model.Scenario_GRID_emission[s] == Total_Grid_Emission  # Return total scenario grid emissions constraint
    
    "Grid constraints" 
    def Maximum_Power_From_Grid(model,s,y,t):  # Cap imports from grid; zero before connection year or during outages
        if y < model.Year_Grid_Connection or model.Grid_Availability[s, y, t] == 0:  # Grid not yet connected or currently unavailable
            return model.Energy_From_Grid[s,y,t] == 0  # Grid not yet connected or currently unavailable
        else:  # Grid connected and available: cap imports at the max grid power
            return model.Energy_From_Grid[s,y,t] <= model.Maximum_Grid_Power  # kW converted to W
    
    def Maximum_Power_To_Grid(model,s,y,t):  # Cap exports to grid; zero if not allowed or grid unavailable
        if y < model.Year_Grid_Connection or model.Grid_Connection_Type == 1 or model.Grid_Availability[s, y, t] == 0:  # Export not allowed (import-only) or grid unavailable/not yet connected
            return model.Energy_To_Grid[s, y, t] == 0  # Export not allowed or grid unavailable
        elif model.Grid_Connection_Type == 0:  # Bidirectional connection: exports are allowed
            return model.Energy_To_Grid[s, y, t] <= model.Maximum_Grid_Power  # kW converted to W
        
    def Single_Flow_Energy_To_Grid(model,s,yt,ut,t):  # Binary Single_Flow_Grid prevents exporting while importing
        return model.Energy_To_Grid[s,yt,t] <= model.Single_Flow_Grid[s,yt,t]*model.Large_Constant  # Export allowed only when the export-mode binary is active
        
    def Single_Flow_Energy_From_Grid(model,s,yt,ut,t):  # Complementary binary: import only when not exporting
        return model.Energy_From_Grid[s,yt,t] <= (1-model.Single_Flow_Grid[s,yt,t])*model.Large_Constant  # Import allowed only when the export-mode binary is inactive

class Constraints_Brownfield():  # LP formulation — brownfield (existing assets accounted for)
    current_directory = os.path.dirname(os.path.abspath(__file__))  # Folder containing this module
    inputs_directory = os.path.join(current_directory, '..', 'Inputs')  # Sibling Inputs folder
    data_file_path = params_path.PARAMS_PATH  # Resolved once via MGPY_PARAMS; see params_path.py
    Data_import = open(data_file_path).readlines()  # Read the parameters file at class-definition time
    for i in range(len(Data_import)):  # Scan every line of the parameters file
        if "param: Fuel_Specific_Cost_Calculation" in Data_import[i]:      # Locate the flag line
            Fuel_Specific_Cost_Calculation = int((re.findall('\d+',Data_import[i])[0]))  # 0 = constant price, 1 = time-varying
    
    "Objective function"
    def Net_Present_Cost_Obj(model):   # Pyomo objective wrapper for total Net Present Cost
        return (sum(model.Scenario_Net_Present_Cost[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected NPC = probability-weighted sum across scenarios
    
    def CO2_emission_Obj(model):  # Pyomo objective wrapper for total CO2 emissions
        return (sum(model.Scenario_CO2_emission[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected CO2 emissions = probability-weighted sum across scenarios
    
    def Total_Variable_Cost_Obj(model):  # Pyomo objective wrapper for total variable cost
        return (sum(model.Total_Scenario_Variable_Cost_NonAct[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected non-actualized variable cost = probability-weighted sum across scenarios
    
    
    "Net Present Cost"
    def Net_Present_Cost(model):     # Defines the scalar Net Present Cost as expected value over all scenarios
        return model.Net_Present_Cost == (sum(model.Scenario_Net_Present_Cost[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected NPC = probability-weighted sum across scenarios
    
    # def Scenario_Net_Present_Cost(model,s): 
    #     foo = []
    #     for g in range(1,model.Generator_Types+1):
    #             foo.append((s,g))            
    #     Fuel_Cost = sum(model.Total_Fuel_Cost_Act[s,g] for s,g in foo)    
    #     return model.Scenario_Net_Present_Cost[s] == (model.Investment_Cost + model.Operation_Maintenance_Cost_Act + model.Battery_Replacement_Cost_Act[s] 
    #             + model.Scenario_Lost_Load_Cost_Act[s] + Fuel_Cost - model.Salvage_Value)   
    
    def Scenario_Net_Present_Cost(model,s):   # NPC for scenario s: CAPEX + variable OPEX - salvage value
        return model.Scenario_Net_Present_Cost[s] == (model.Investment_Cost + model.Total_Scenario_Variable_Cost_Act[s] - model.Salvage_Value)     # NPC = CAPEX + actualized variable cost - salvage value
    
    def Total_Variable_Cost(model):  # Defines the scalar total variable cost (non-actualized)
        return model.Total_Variable_Cost == (sum(model.Total_Scenario_Variable_Cost_NonAct[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected non-actualized variable cost = probability-weighted sum across scenarios
    
    def CO2_emission(model):  # Defines the scalar CO2 emission as expected value over all scenarios
        return model.CO2_emission == (sum(model.Scenario_CO2_emission[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected CO2 emissions = probability-weighted sum across scenarios
    
    def Scenario_CO2_emission(model,s):  # Total lifecycle + operational CO2 for scenario s; components depend on active model parts
        if model.Grid_Connection == 1:  # Grid-connected: include grid import emissions
            if model.Model_Components == 0:  # Full system: RES + generator + battery
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.GEN_emission + model.BESS_emission + model.Scenario_FUEL_emission[s] + model.Scenario_GRID_emission[s])  # Full system with grid: RES + generator + battery + fuel + grid import emissions
            if model.Model_Components == 1:  # RES + battery only (no generator)
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.BESS_emission + model.Scenario_GRID_emission[s])  # Grid-connected, no generator: RES + battery + grid import emissions
            if model.Model_Components == 2:  # RES + generator only (no battery)
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.GEN_emission + model.Scenario_FUEL_emission[s] + model.Scenario_GRID_emission[s])  # Grid-connected, no battery: RES + generator + fuel + grid import emissions
        else:  # Off-grid: no grid-related emissions to add
            if model.Model_Components == 0:  # Full system: RES + generator + battery
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.GEN_emission + model.BESS_emission + model.Scenario_FUEL_emission[s])  # Off-grid full system: RES + generator + battery + fuel emissions
            if model.Model_Components == 1:  # RES + battery only (no generator)
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.BESS_emission)  # Off-grid, no generator: RES + battery emissions only
            if model.Model_Components == 2:  # RES + generator only (no battery)
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.GEN_emission + model.Scenario_FUEL_emission[s])  # Off-grid, no battery: RES + generator + fuel emissions
    
    "Investment cost"
    def Investment_Cost(model):    # Total discounted CAPEX for RES + generator + battery + grid connection
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration   # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]    # Will map each project year to its (year, step_index) tuple
        for y in model.years:        # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])            # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))              # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]    # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[model.Step_Duration*i + model.Step_Duration]   # Upgrade decision occurs at the end of each step
          
        Inv_Ren = sum(((model.RES_Units[1,r]*model.RES_Nominal_Capacity[r]-model.RES_capacity[r])*model.RES_Specific_Investment_Cost[r])  # Step-1 RES CAPEX minus credit for pre-existing capacity
                        + sum((((model.RES_Units[ut,r] - model.RES_Units[ut-1,r])*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r]))/((1+model.Discount_Rate)**(yt-1))  # NPV of incremental RES added at each upgrade year
                        for (yt,ut) in tup_list) for r in model.renewable_sources)    # Sum incremental RES investment NPV over all upgrade points
        Inv_Gen = sum(((model.Generator_Nominal_Capacity[1,g]-model.Generator_capacity[g])*model.Generator_Specific_Investment_Cost[g])  # Step-1 generator CAPEX minus credit for pre-existing capacity
                        + sum((((model.Generator_Nominal_Capacity[ut,g] - model.Generator_Nominal_Capacity[ut-1,g])*model.Generator_Specific_Investment_Cost[g]))/((1+model.Discount_Rate)**(yt-1))  # NPV of incremental generator capacity
                        for (yt,ut) in tup_list) for g in model.generator_types)  # Sum incremental generator investment NPV over all upgrade points
        Inv_Bat = (((model.Battery_Nominal_Capacity[1]-model.Battery_capacity)*model.Battery_Specific_Investment_Cost)  # Step-1 battery CAPEX minus credit for pre-existing capacity
                        + sum((((model.Battery_Nominal_Capacity[ut] - model.Battery_Nominal_Capacity[ut-1])*model.Battery_Specific_Investment_Cost))/((1+model.Discount_Rate)**(yt-1))  # NPV of incremental battery capacity
                        for (yt,ut) in tup_list))  # Sum incremental battery investment NPV over all upgrade points
        Inv_Grid = 0  # Accumulator for NPV of grid connection investment
        for y in model.years:  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur the grid connection cost once the connection year has been reached
                Inv_Grid += (model.Grid_Connection_Cost * model.Grid_Distance)/((1+model.Discount_Rate)**(y-1))   # Discount grid connection cost to year 0 and accumulate
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            investment_cost = Inv_Ren + Inv_Gen + Inv_Bat  # Full system: RES + generator + battery CAPEX
            if model.Grid_Connection: investment_cost += Inv_Grid  # Add grid connection cost if enabled
            return model.Investment_Cost == investment_cost   # Return the assembled CAPEX constraint
        if model.Model_Components == 1:  # RES + battery only (no generator)
            investment_cost = Inv_Ren + Inv_Bat  # RES + battery CAPEX (no generator)
            if model.Grid_Connection: investment_cost += Inv_Grid  # Add grid connection cost if enabled
            return model.Investment_Cost == investment_cost        # Return the assembled CAPEX constraint
        if model.Model_Components == 2:  # RES + generator only (no battery)
            investment_cost = Inv_Ren + Inv_Gen  # RES + generator CAPEX (no battery)
            if model.Grid_Connection: investment_cost += Inv_Grid  # Add grid connection cost if enabled
            return model.Investment_Cost == investment_cost    # Return the assembled CAPEX constraint
    
    def Investment_Cost_Limit(model):  # Enforce a user-defined cap on total CAPEX
        return model.Investment_Cost <= model.Investment_Cost_Limit  # Total CAPEX must not exceed the limit
    
    "Fixed O&M costs"
    def Operation_Maintenance_Cost_Act(model):  # Actualized (discounted) fixed O&M cost of RES + generator + battery + grid
        OyM_Ren = sum(sum((model.RES_Units[ut,r]*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r]*model.RES_Specific_OM_Cost[r])/((  # NPV of RES fixed O&M cost, discounted per year it is incurred
                        1+model.Discount_Rate)**yt)for (yt,ut) in model.years_steps)for r in model.renewable_sources)      # Sum discounted RES O&M cost over all years/steps
        OyM_Gen = sum(sum((model.Generator_Nominal_Capacity[ut,g]*model.Generator_Specific_Investment_Cost[g]*model.Generator_Specific_OM_Cost[g])/((  # NPV of generator fixed O&M cost, discounted per year it is incurred
                        1+model.Discount_Rate)**yt)for (yt,ut) in model.years_steps)for g in model.generator_types)  # Sum discounted generator O&M cost over all years/steps
        OyM_Bat = sum((model.Battery_Nominal_Capacity[ut]*model.Battery_Specific_Investment_Cost*model.Battery_Specific_OM_Cost)/((  # NPV of battery fixed O&M cost, discounted per year it is incurred
                        1+model.Discount_Rate)**yt)for (yt,ut) in model.years_steps)  # Sum discounted battery O&M cost over all years/steps
        OyM_Grid = 0  # Accumulator for grid connection O&M cost
        for y in model.years:  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur grid O&M cost once the connection year has been reached
                OyM_Grid += (model.Grid_Connection_Cost * model.Grid_Distance * model.Grid_Maintenance_Cost)/((1+model.Discount_Rate)**(y))   # Discounted grid maintenance cost, only once grid is connected
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            cost = OyM_Ren + OyM_Gen + OyM_Bat  # Full system: RES + generator + battery O&M
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_Act == cost  # Return the assembled actualized O&M constraint
        if model.Model_Components == 1:  # RES + battery only (no generator)
            cost = OyM_Ren + OyM_Bat  # RES + battery O&M (no generator)
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_Act == cost  # Return the assembled actualized O&M constraint
        if model.Model_Components == 2:  # RES + generator only (no battery)
            cost = OyM_Ren + OyM_Gen  # RES + generator O&M (no battery)
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_Act == cost   # Return the assembled actualized O&M constraint
    
    def Operation_Maintenance_Cost_NonAct(model):  # Non-actualized (undiscounted) fixed O&M cost, mirrors the Act version
        OyM_Ren = sum(sum((model.RES_Units[ut,r]*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r]*model.RES_Specific_OM_Cost[r])  # Undiscounted RES fixed O&M cost
                        for (yt,ut) in model.years_steps)for r in model.renewable_sources)      # Sum undiscounted RES O&M cost over all years/steps
        OyM_Gen = sum(sum((model.Generator_Nominal_Capacity[ut,g]*model.Generator_Specific_Investment_Cost[g]*model.Generator_Specific_OM_Cost[g])  # Undiscounted generator fixed O&M cost
                        for (yt,ut) in model.years_steps)for g in model.generator_types)  # Sum undiscounted generator O&M cost over all years/steps
        OyM_Bat = sum((model.Battery_Nominal_Capacity[ut]*model.Battery_Specific_Investment_Cost*model.Battery_Specific_OM_Cost)  # Undiscounted battery fixed O&M cost
                        for (yt,ut) in model.years_steps)  # Sum undiscounted battery O&M cost over all years/steps
        OyM_Grid = 0  # Accumulator for grid connection O&M cost
        for y in model.years:  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur grid O&M cost once the connection year has been reached
                OyM_Grid += (model.Grid_Connection_Cost * model.Grid_Distance * model.Grid_Maintenance_Cost)  # Undiscounted grid maintenance cost, only once grid is connected
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            cost = OyM_Ren + OyM_Gen + OyM_Bat  # Full system: RES + generator + battery O&M
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_NonAct == cost   # Return the assembled non-actualized O&M constraint
        if model.Model_Components == 1:  # RES + battery only (no generator)
            cost = OyM_Ren + OyM_Bat  # RES + battery O&M (no generator)
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_NonAct == cost   # Return the assembled non-actualized O&M constraint
        if model.Model_Components == 2:  # RES + generator only (no battery)
            cost = OyM_Ren + OyM_Gen  # RES + generator O&M (no battery)
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_NonAct == cost    # Return the assembled non-actualized O&M constraint
    
    "Variable costs"
    def Total_Variable_Cost_Act(model):  # Scalar actualized total variable cost = expected value over all scenarios
        return model.Total_Variable_Cost_Act == (sum(model.Total_Scenario_Variable_Cost_Act[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Probability-weighted sum across scenarios
    
    def Scenario_Variable_Cost_Act(model, s):  # Actualized variable cost for scenario s: O&M + battery replacement + lost load + fuel + net electricity cost
        foo = []  # Index list accumulator
        for g in range(1,model.Generator_Types+1):  # Loop over all generator types for this scenario
                foo.append((s,g))     # Collect (scenario, generator) index pair
        Fuel_Cost = sum(model.Total_Fuel_Cost_Act[s,g] for s,g in foo)      # Total actualized fuel cost across all generator types
        if model.Grid_Connection == 1:  # Grid-connected: include electricity purchase cost (and sale revenue if allowed)
            Electricity_Cost = model.Total_Electricity_Cost_Act[s]   # Actualized cost of electricity purchased from the grid
            if model.Grid_Connection_Type == 0:  # Bidirectional connection: exports earn revenue
                Electricity_Revenues = model.Total_Revenues_Act[s]  # Actualized revenue from electricity sold to the grid
            else: Electricity_Revenues = 0  # Import-only connection: no export revenue
        else:  # Off-grid: no electricity purchase or export cost
            Electricity_Cost = 0  # Off-grid: no electricity purchase cost
            Electricity_Revenues = 0      # Off-grid: no electricity sale revenue
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            return model.Total_Scenario_Variable_Cost_Act[s] == model.Operation_Maintenance_Cost_Act + model.Battery_Replacement_Cost_Act[s] + model.Scenario_Lost_Load_Cost_Act[s] + Fuel_Cost + Electricity_Cost - Electricity_Revenues       # Full system: O&M + battery replacement + lost load + fuel + net electricity
        if model.Model_Components == 1:  # RES + battery only (no generator)
            return model.Total_Scenario_Variable_Cost_Act[s] == model.Operation_Maintenance_Cost_Act + model.Battery_Replacement_Cost_Act[s] + model.Scenario_Lost_Load_Cost_Act[s] + Electricity_Cost - Electricity_Revenues       # No generator: O&M + battery replacement + lost load + net electricity
        if model.Model_Components == 2:  # RES + generator only (no battery)
            return model.Total_Scenario_Variable_Cost_Act[s] == model.Operation_Maintenance_Cost_Act + model.Scenario_Lost_Load_Cost_Act[s] + Fuel_Cost + Electricity_Cost - Electricity_Revenues       # No battery: O&M + lost load + fuel + net electricity
    
    def Scenario_Variable_Cost_NonAct(model, s):  # Non-actualized (undiscounted) mirror of Scenario_Variable_Cost_Act
        foo = []  # Index list accumulator
        for g in range(1,model.Generator_Types+1):  # Loop over all generator types for this scenario
                foo.append((s,g))     # Collect (scenario, generator) index pair
        Fuel_Cost = sum(model.Total_Fuel_Cost_NonAct[s,g] for s,g in foo)      # Total undiscounted fuel cost across all generator types
        if model.Grid_Connection == 1:  # Grid-connected: include electricity purchase cost (and sale revenue if allowed)
            Electricity_Cost = model.Total_Electricity_Cost_NonAct[s]   # Undiscounted cost of electricity purchased from the grid
            if model.Grid_Connection_Type == 0:  # Bidirectional connection: exports earn revenue
                Electricity_Revenues = model.Total_Revenues_NonAct[s]  # Undiscounted revenue from electricity sold to the grid
            else: Electricity_Revenues = 0  # Import-only connection: no export revenue
        else:  # Off-grid: no electricity purchase or export cost
            Electricity_Cost = 0  # Off-grid: no electricity purchase cost
            Electricity_Revenues = 0       # Off-grid: no electricity sale revenue
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            return model.Total_Scenario_Variable_Cost_NonAct[s] == model.Operation_Maintenance_Cost_NonAct + model.Battery_Replacement_Cost_NonAct[s] + model.Scenario_Lost_Load_Cost_NonAct[s] + Fuel_Cost + Electricity_Cost - Electricity_Revenues   # Full system: O&M + battery replacement + lost load + fuel + net electricity
        if model.Model_Components == 1:  # RES + battery only (no generator)
            return model.Total_Scenario_Variable_Cost_NonAct[s] == model.Operation_Maintenance_Cost_NonAct + model.Battery_Replacement_Cost_NonAct[s] + model.Scenario_Lost_Load_Cost_NonAct[s] + Electricity_Cost - Electricity_Revenues   # No generator: O&M + battery replacement + lost load + net electricity
        if model.Model_Components == 2:  # RES + generator only (no battery)
            return model.Total_Scenario_Variable_Cost_NonAct[s] == model.Operation_Maintenance_Cost_NonAct + model.Scenario_Lost_Load_Cost_NonAct[s] + Fuel_Cost + Electricity_Cost - Electricity_Revenues   # No battery: O&M + lost load + fuel + net electricity
    
    def Scenario_Lost_Load_Cost_Act(model,s):      # Actualized cost of unserved (lost load) energy for scenario s
        Cost_Lost_Load = 0           # Accumulator for discounted yearly lost-load cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(model.Lost_Load[s,y,t]*model.Lost_Load_Specific_Cost for t in model.periods)  # Yearly lost-load cost = unserved energy x penalty price
            Cost_Lost_Load += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
        return  model.Scenario_Lost_Load_Cost_Act[s] == Cost_Lost_Load  # Return actualized lost-load cost constraint
    
    def Scenario_Lost_Load_Cost_NonAct(model,s):  # Non-actualized (undiscounted) mirror of Scenario_Lost_Load_Cost_Act
        Cost_Lost_Load = 0           # Accumulator for discounted yearly lost-load cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(model.Lost_Load[s,y,t]*model.Lost_Load_Specific_Cost for t in model.periods)  # Yearly lost-load cost = unserved energy x penalty price
            Cost_Lost_Load += Num  # Accumulate undiscounted yearly lost-load cost
        return  model.Scenario_Lost_Load_Cost_NonAct[s] == Cost_Lost_Load  # Return non-actualized lost-load cost constraint
    
    if Fuel_Specific_Cost_Calculation == 0:  # Constant (year-independent) fuel marginal cost
        def Total_Fuel_Cost_Act(model,s,g):  # Actualized fuel cost for generator type g across the project horizon
            Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
            for y in range(1, model.Years +1):  # Loop over every project year
                Num = sum(model.Generator_Energy_Production[s,y,g,t]*model.Generator_Marginal_Cost_1[g] for t in model.periods)  # Yearly fuel cost = energy produced x constant marginal cost
                Fuel_Cost_Tot += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
            return model.Total_Fuel_Cost_Act[s,g] == Fuel_Cost_Tot  # Return actualized fuel cost constraint
        
        def Total_Fuel_Cost_NonAct(model,s,g):  # Non-actualized (undiscounted) mirror of Total_Fuel_Cost_Act
            Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
            for y in range(1, model.Years +1):  # Loop over every project year
                Num = sum(model.Generator_Energy_Production[s,y,g,t]*model.Generator_Marginal_Cost_1[g] for t in model.periods)  # Yearly fuel cost = energy produced x constant marginal cost
                Fuel_Cost_Tot += Num  # Accumulate undiscounted yearly fuel cost
            return model.Total_Fuel_Cost_NonAct[s,g] == Fuel_Cost_Tot  # Return non-actualized fuel cost constraint
    else:  # Time-varying (year-specific) fuel price mode
        def Total_Fuel_Cost_Act(model,s,g):  # Actualized fuel cost for generator type g across the project horizon
            Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
            for y in range(1, model.Years +1):  # Loop over every project year
                Num = sum(model.Generator_Energy_Production[s,y,g,t]*model.Generator_Marginal_Cost[g,y] for t in model.periods)  # Yearly fuel cost = energy produced x year-specific marginal cost
                Fuel_Cost_Tot += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
            return model.Total_Fuel_Cost_Act[s,g] == Fuel_Cost_Tot  # Return actualized fuel cost constraint
        
        def Total_Fuel_Cost_NonAct(model,s,g):  # Non-actualized (undiscounted) mirror of Total_Fuel_Cost_Act
            Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
            for y in range(1, model.Years +1):  # Loop over every project year
                Num = sum(model.Generator_Energy_Production[s,y,g,t]*model.Generator_Marginal_Cost[g,y] for t in model.periods)  # Yearly fuel cost = energy produced x year-specific marginal cost
                Fuel_Cost_Tot += Num  # Accumulate undiscounted yearly fuel cost
            return model.Total_Fuel_Cost_NonAct[s,g] == Fuel_Cost_Tot  # Return non-actualized fuel cost constraint

    def Total_Electricity_Cost_Act(model,s):   # Actualized cost of electricity purchased from the national grid
        Electricity_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly electricity cost
        for y in range(1, model.Years +1):  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur grid purchase cost once the connection year has been reached
                Num = sum(model.Energy_From_Grid[s,y,t]*model.Grid_Availability[s,y,t]*model.Grid_Purchased_El_Price  for t in model.periods)  # Yearly cost = grid imports (masked by availability) x purchase price / 1000
                Electricity_Cost_Tot += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
        return model.Total_Electricity_Cost_Act[s] == Electricity_Cost_Tot  # Return actualized electricity cost constraint
       
    def Total_Electricity_Cost_NonAct(model,s):   # Non-actualized (undiscounted) mirror of Total_Electricity_Cost_Act
        Electricity_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly electricity cost
        for y in range(1, model.Years +1):  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur grid purchase cost once the connection year has been reached
                Electricity_Cost_Tot += sum(model.Energy_From_Grid[s,y,t]*model.Grid_Availability[s,y,t]*model.Grid_Purchased_El_Price  for t in model.periods)  # Accumulate undiscounted yearly electricity cost
        return model.Total_Electricity_Cost_NonAct[s] == Electricity_Cost_Tot  # Return non-actualized electricity cost constraint
    
    
    def Total_Revenues_NonAct(model,s):   # Undiscounted revenue from electricity exported to the grid
        Revenues_Yearly = [0 for y in model.years]  # Per-year export revenue, zero before grid connection
        for y in range(1, model.Years +1):  # Loop over every project year
            Revenues_Yearly[y-1] = sum(model.Energy_To_Grid[s,y,t]*model.Grid_Availability[s,y,t] * model.Grid_Sold_El_Price for t in model.periods)  # Yearly revenue = grid exports (masked by availability) x sale price / 1000
        return model.Total_Revenues_NonAct [s] == sum(Revenues_Yearly[y-1] for y in model.years)  # Sum undiscounted yearly revenues
    
    def Total_Revenues_Act(model,s):   # Actualized (discounted) revenue from electricity exported to the grid
        Revenues_Yearly = [0 for y in model.years]  # Per-year export revenue, zero before grid connection
        for y in range(1,model.Years+1):  # Loop over every project year
            Revenues_Yearly [y-1] = sum(model.Energy_To_Grid[s,y,t]*model.Grid_Availability[s,y,t] * model.Grid_Sold_El_Price for t in model.periods)  # Yearly revenue = grid exports (masked by availability) x sale price / 1000
        return model.Total_Revenues_Act [s] == sum(Revenues_Yearly[y-1]/((1+model.Discount_Rate)**y)  for y in model.years)  # Discount each year's export revenue to year 0 and sum
    
    def Battery_Replacement_Cost_Act(model,s):  # Actualized cost of replacing battery cells as they cycle (charge/discharge wear)
        Battery_cost_in = [0 for y in model.years]  # Per-year cost attributable to charging throughput
        Battery_cost_out = [0 for y in model.years]  # Per-year cost attributable to discharging throughput
        Battery_Yearly_cost = [0 for y in model.years]      # Per-year total replacement cost (charge + discharge)
        for y in range(1,model.Years+1):      # Loop over every project year
            Battery_cost_in[y-1] = sum(model.Battery_Inflow[s,y,t]*model.Unitary_Battery_Replacement_Cost for t in model.periods)  # Charging energy x unitary replacement cost, summed over the year
            Battery_cost_out[y-1] = sum(model.Battery_Outflow[s,y,t]*model.Unitary_Battery_Replacement_Cost for t in model.periods)  # Discharging energy x unitary replacement cost, summed over the year
            Battery_Yearly_cost[y-1] = Battery_cost_in[y-1] + Battery_cost_out[y-1]  # Total yearly replacement cost = charge + discharge wear
        return model.Battery_Replacement_Cost_Act[s] == sum(Battery_Yearly_cost[y-1]/((1+model.Discount_Rate)**y) for y in model.years)   # Discount each year's replacement cost to year 0 and sum
        
    def Battery_Replacement_Cost_NonAct(model,s):  # Non-actualized (undiscounted) mirror of Battery_Replacement_Cost_Act
        Battery_cost_in = [0 for y in model.years]  # Per-year cost attributable to charging throughput
        Battery_cost_out = [0 for y in model.years]  # Per-year cost attributable to discharging throughput
        Battery_Yearly_cost = [0 for y in model.years]      # Per-year total replacement cost (charge + discharge)
        for y in range(1,model.Years+1):      # Loop over every project year
            Battery_cost_in[y-1] = sum(model.Battery_Inflow[s,y,t]*model.Unitary_Battery_Replacement_Cost for t in model.periods)  # Charging energy x unitary replacement cost, summed over the year
            Battery_cost_out[y-1] = sum(model.Battery_Outflow[s,y,t]*model.Unitary_Battery_Replacement_Cost for t in model.periods)  # Discharging energy x unitary replacement cost, summed over the year
            Battery_Yearly_cost[y-1] = Battery_cost_in[y-1] + Battery_cost_out[y-1]  # Total yearly replacement cost = charge + discharge wear
        return model.Battery_Replacement_Cost_NonAct[s] == sum(Battery_Yearly_cost[y-1] for y in model.years)    # Sum undiscounted yearly replacement cost
    
    
    "Salvage Value"
    def _Salvage_Value_Raw(model):     # Residual (undepreciated) value of all assets at project end, discounted to year 0 -- NOT floored at zero, see Salvage_Value_Upper_Raw/Flag below
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration  # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur  # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]  # Will map each project year to its (year, step_index) tuple
        for y in model.years:      # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])          # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))            # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]  # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[s_dur*i + s_dur]  # Upgrade decision occurs at the end of each step

        if model.Steps_Number == 1:      # Single investment step: only the initial build can have residual value
            SV_Ren_1 = sum(((model.RES_Units[1,r]*model.RES_Nominal_Capacity[r])-model.RES_capacity[r])*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]-model.Years)/model.RES_Lifetime[r] /   # Residual value of step-1 RES units, net of credit for pre-existing capacity
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)+sum((model.RES_capacity[r])*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]-model.RES_years[r]-model.Years)/model.RES_Lifetime[r] /   # Plus residual value of the pre-existing RES capacity itself
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)          # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_2 = 0   # No second-step RES capacity exists yet
            SV_Ren_3 = 0      # No later-step RES capacity exists yet
            SV_Gen_1 = sum((model.Generator_Nominal_Capacity[1,g]-model.Generator_capacity[g])*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]-model.Years)/model.Generator_Lifetime[g] /   # Residual value of step-1 generator capacity, net of credit for pre-existing capacity
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)+sum(model.Generator_capacity[g]*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]-model.GEN_years[g]-model.Years)/model.Generator_Lifetime[g] /   # Plus residual value of the pre-existing generator capacity itself
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)          # Discount generator residual value to year 0, summed over generator types
            SV_Gen_2 = 0  # No second-step generator capacity exists yet
            SV_Gen_3 = 0  # No later-step generator capacity exists yet
            SV_Grid = model.Grid_Distance*model.Grid_Connection_Cost*model.Grid_Connection / ((1 + model.Discount_Rate)**(model.Years - model.Year_Grid_Connection))   # Discounted residual value of the grid connection investment
    
        if model.Steps_Number == 2:      # Two investment steps: initial build + one upgrade can have residual value
            yt_last_up = upgrade_years_list[1]         # Year of the (only) capacity upgrade
            SV_Ren_1 = sum(((model.RES_Units[1,r]*model.RES_Nominal_Capacity[r])-model.RES_capacity[r])*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]-model.Years)/model.RES_Lifetime[r] /   # Residual value of step-1 RES units, net of credit for pre-existing capacity
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)+sum((model.RES_capacity[r])*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]-model.RES_years[r]-model.Years)/model.RES_Lifetime[r] /   # Plus residual value of the pre-existing RES capacity itself
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)  # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_2 = sum((model.RES_Units[2,r]-model.RES_Units[1,r])*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]+(yt_last_up-1)-model.Years)/model.RES_Lifetime[r] /   # Residual value of the step-2 RES capacity increment
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)          # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_3 = 0      # No later-step RES capacity exists yet
            SV_Gen_1 = sum((model.Generator_Nominal_Capacity[1,g]-model.Generator_capacity[g])*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]-model.Years)/model.Generator_Lifetime[g] /   # Residual value of step-1 generator capacity, net of credit for pre-existing capacity
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)+sum(model.Generator_capacity[g]*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]-model.GEN_years[g]-model.Years)/model.Generator_Lifetime[g] /   # Plus residual value of the pre-existing generator capacity itself
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)          # Discount generator residual value to year 0, summed over generator types
            SV_Gen_2 = sum((model.Generator_Nominal_Capacity[2,g]-model.Generator_Nominal_Capacity[1,g])*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]+(yt_last_up-1)-model.Years)/model.Generator_Lifetime[g] /   # Residual value of the step-2 generator capacity increment
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)  # Discount generator residual value to year 0, summed over generator types
            SV_Gen_3 = 0  # No later-step generator capacity exists yet
            SV_Grid = model.Grid_Distance*model.Grid_Connection_Cost*model.Grid_Connection / ((1 + model.Discount_Rate)**(model.Years - model.Year_Grid_Connection))   # Discounted residual value of the grid connection investment
            
        if model.Steps_Number > 2:  # Three or more steps: initial build + last upgrade + all intermediate upgrades
            tup_list_2 = [[] for i in range(len(model.steps)-2)]  # (year, step) pairs for upgrades before the last one
            for i in range(len(model.steps) - 2):  # Build the intermediate-upgrade list (excludes the final upgrade)
                tup_list_2[i] = yu_tuples_list[s_dur*i + s_dur]  # Intermediate upgrade occurs at the end of each step
            yt_last_up = upgrade_years_list[-1]  # Year of the final capacity upgrade
            ut_last_up = tup_list[-1][1]      # Step index of the final upgrade
            ut_seclast_up = tup_list[-2][1]  # Step index of the second-to-last upgrade
    
            SV_Ren_1 = sum(((model.RES_Units[1,r]*model.RES_Nominal_Capacity[r])-model.RES_capacity[r])*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]-model.Years)/model.RES_Lifetime[r] /   # Residual value of step-1 RES units, net of credit for pre-existing capacity
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)+sum(model.RES_capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]-model.RES_years[r]-model.Years)/model.RES_Lifetime[r] /   # Plus residual value of the pre-existing RES capacity itself
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)      # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_2 = sum((model.RES_Units[ut_last_up,r] - model.RES_Units[ut_seclast_up,r])*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]+(yt_last_up-1)-model.Years)/model.RES_Lifetime[r] /   # Residual value of the final RES capacity increment
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)  # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_3 = sum(sum((model.RES_Units[ut,r] - model.RES_Units[ut-1,r])*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]+(yt-1)-model.Years)/model.RES_Lifetime[r] /   # Residual value of all intermediate RES capacity increments
                            ((1+model.Discount_Rate)**model.Years) for (yt,ut) in tup_list_2) for r in model.renewable_sources)          # Sum intermediate-upgrade RES residual value, discounted to year 0
            SV_Gen_1 = sum((model.Generator_Nominal_Capacity[1,g]-model.Generator_capacity[g])*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]-model.Years)/model.Generator_Lifetime[g] /   # Residual value of step-1 generator capacity, net of credit for pre-existing capacity
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)+sum(model.Generator_capacity[g]*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]-model.GEN_years[g]-model.Years)/model.Generator_Lifetime[g] /   # Plus residual value of the pre-existing generator capacity itself
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)  # Discount generator residual value to year 0, summed over generator types
            SV_Gen_2 = sum((model.Generator_Nominal_Capacity[ut_last_up,g] - model.Generator_Nominal_Capacity[ut_seclast_up,g])*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]+(yt_last_up-1)-model.Years)/model.Generator_Lifetime[g] /   # Residual value of the final generator capacity increment
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)  # Discount generator residual value to year 0, summed over generator types
            SV_Gen_3 = sum(sum((model.Generator_Nominal_Capacity[ut,g] - model.Generator_Nominal_Capacity[ut-1,g])*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]+(yt-1)-model.Years)/model.Generator_Lifetime[g] /   # Residual value of all intermediate generator capacity increments
                            ((1+model.Discount_Rate)**model.Years) for (yt,ut) in tup_list_2) for g in model.generator_types)  # Sum intermediate-upgrade generator residual value, discounted to year 0
            SV_Grid = model.Grid_Distance*model.Grid_Connection_Cost*model.Grid_Connection / ((1 + model.Discount_Rate)**(model.Years - model.Year_Grid_Connection))   # Discounted residual value of the grid connection investment
       
        if model.Model_Components == 0 or model.Model_Components == 2:  # System includes a generator: credit RES + generator + grid salvage value
            return SV_Ren_1 + SV_Gen_1 + SV_Ren_2 + SV_Gen_2 + SV_Ren_3 + SV_Gen_3 + SV_Grid  # Sum residual value of all RES/generator vintages plus grid connection
        if model.Model_Components == 1:  # RES + battery only (no generator)
            return SV_Ren_1 + SV_Ren_2 + SV_Ren_3 + SV_Grid  # No generator: sum residual value of RES vintages plus grid connection

    def Salvage_Value_Upper_Raw(model):   # Floor-at-zero linearization (2026-08-28): Salvage_Value <= raw formula, but only enforced when Salvage_Positive_Flag=1
        BIG_M_SALVAGE = 2.0e6  # tightened 2026-08-29: anchored to Generator_Max_Units's hard cap (9400kW x $160.58/kW = ~$1.51M for this project's designs) plus headroom -- still ~230x the only real negative-salvage magnitude observed (-$8,538). RES has no formal capacity ceiling in this model, so this isn't a rigorous proof-bound, just a much tighter one than the earlier 5.0e6/2.0e7 -- an oversized Big-M here visibly degrades barrier/crossover conditioning (RHS/matrix range balloons relative to the rest of the model's coefficients)
        return model.Salvage_Value <= Constraints_Brownfield._Salvage_Value_Raw(model) + BIG_M_SALVAGE * (1 - model.Salvage_Positive_Flag)

    def Salvage_Value_Upper_Flag(model):   # Floor-at-zero linearization: Salvage_Value <= 0, forced whenever Salvage_Positive_Flag=0 (i.e. whenever the raw formula is negative)
        BIG_M_SALVAGE = 2.0e6
        return model.Salvage_Value <= BIG_M_SALVAGE * model.Salvage_Positive_Flag

    def BESS_Capacity(model,ut): #Minimum battery capacity: total must cover the pre-existing battery
        return model.Battery_Nominal_Capacity[1] >= model.Battery_capacity  # Installed capacity must cover the pre-existing battery
    
    def GEN_Capacity(model,ut,g): #Minimum generator capacity: total must cover the pre-existing generator
        return model.Generator_Nominal_Capacity[1,g] >= model.Generator_capacity[g]  # Installed capacity must cover the pre-existing generator
    
    def RES_Capacity(model,s,yt,ut,r,t): #Minimum RES energy production: must at least match pre-existing unit output
        return model.RES_Energy_Production[s,yt,r,t] >= model.RES_Unit_Energy_Production[s,r,t]*model.RES_Inverter_Efficiency[r]*(model.RES_capacity[r] / model.RES_Nominal_Capacity[r])  # Minimum RES energy production must at least match pre-existing unit output
    
    def Energy_balance(model,s,yt,ut,t): # Energy balance
        Foo = []  # Index list for RES energy production in this period
        for r in model.renewable_sources:  # Loop over all renewable source types
            Foo.append((s,yt,r,t))      # Collect index for this scenario/year/source/time
        Total_Renewable_Energy = sum(model.RES_Energy_Production[j] for j in Foo)      # Sum of all RES output in this period
        foo=[]  # Index list for generator energy production in this period
        for g in model.generator_types:  # Loop over all generator types
            foo.append((s,yt,g,t))      # Collect index for this scenario/year/generator/time
        Total_Generator_Energy = sum(model.Generator_Energy_Production[i] for i in foo)    # Sum of all generator output in this period
        if model.Grid_Connection == 1:  # Grid-connected: allow energy exchange with the grid
            En_From_Grid = model.Energy_From_Grid[s,yt,t]*model.Grid_Availability[s,yt,t]  # Grid imports masked by outage availability
            if model.Grid_Connection_Type == 0:  # Bidirectional connection: exports earn revenue
                En_To_Grid = model.Energy_To_Grid[s,yt,t]*model.Grid_Availability[s,yt,t]  # Grid exports masked by outage availability (only if export allowed)
            else: En_To_Grid = 0  # Export not allowed for this connection type
        else:  # Off-grid: no grid imports or exports
            En_From_Grid = 0  # Off-grid: no imports
            En_To_Grid = 0  # Off-grid: no exports
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            return model.Energy_Demand[s,yt,t] == (Total_Renewable_Energy   # Full system: demand = RES + generator + grid import/export + battery +/- lost load/curtailment
                                                   + Total_Generator_Energy  # Add generator output
                                                   + En_From_Grid  # Add grid imports
                                                   - En_To_Grid   # Subtract grid exports
                                                   - model.Battery_Inflow[s,yt,t]   # Subtract battery charge
                                                   + model.Battery_Outflow[s,yt,t]   # Add battery discharge
                                                   + model.Lost_Load[s,yt,t]    # Add unserved (lost load) energy
                                                   - model.Energy_Curtailment[s,yt,t] )       # Subtract curtailed (wasted) energy
        if model.Model_Components == 1:  # RES + battery only (no generator)
            return model.Energy_Demand[s,yt,t] == (Total_Renewable_Energy   # Full system: demand = RES + generator + grid import/export + battery +/- lost load/curtailment
                                                   + En_From_Grid  # Add grid imports
                                                   - En_To_Grid   # Subtract grid exports
                                                   - model.Battery_Inflow[s,yt,t]   # Subtract battery charge
                                                   + model.Battery_Outflow[s,yt,t]   # Add battery discharge
                                                   + model.Lost_Load[s,yt,t]    # Add unserved (lost load) energy
                                                   - model.Energy_Curtailment[s,yt,t] )      # Subtract curtailed (wasted) energy
        if model.Model_Components == 2:  # RES + generator only (no battery)
            return model.Energy_Demand[s,yt,t] == (Total_Renewable_Energy   # Full system: demand = RES + generator + grid import/export + battery +/- lost load/curtailment
                                                   + Total_Generator_Energy  # Add generator output
                                                   + En_From_Grid  # Add grid imports
                                                   - En_To_Grid   # Subtract grid exports
                                                   + model.Lost_Load[s,yt,t]    # Add unserved (lost load) energy
                                                   - model.Energy_Curtailment[s,yt,t] )      # Subtract curtailed (wasted) energy
    
    "Renewable Energy Sources constraints"
    def Renewable_Energy(model,s,yt,ut,r,t): # Energy output of the solar panels
        return model.RES_Energy_Production[s,yt,r,t] == model.RES_Unit_Energy_Production[s,r,t]*model.RES_Inverter_Efficiency[r]*model.RES_Units[ut,r]  # Unit yield x inverter efficiency x number of units
    
    def Renewable_Energy_Penetration(model,ut):      # Enforce minimum renewable energy penetration target for step ut
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration  # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur  # Start year of step i+1
        yu_tuples_list = [0 for i in model.years]  # Will map each project year to its (year, step_index) tuple
        if model.Steps_Number == 1:  # Single investment step: only the initial build can have residual value
            for y in model.years:          # Loop over every project year
                yu_tuples_list[y-1] = (y, 1)  # Single-step case: every year maps to step 1
        else:      # Multiple investment steps: determine which step each year belongs to
            for y in model.years:          # Loop over every project year
                for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                    if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                        yu_tuples_list[y-1] = (y, model.steps[i+1])              # Record (year, step) pair
                    elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                        yu_tuples_list[y-1] = (y, len(model.steps))         # Record (year, final step) pair
        years_list = []  # Project years that belong to investment step ut
        for i in yu_tuples_list:  # Loop over all (year, step) pairs
            if i[1]==ut:  # Keep only the years belonging to the current step ut
                years_list += [i[0]]              # Record this year
        Foo=[]  # Index list accumulator
        for s in range(1, model.Scenarios + 1):  # Loop over all scenarios
            for y in years_list:  # Loop over years belonging to step ut
                for g in range(1, model.Generator_Types+1):  # Loop over all generator types
                    for t in range(1,model.Periods+1):  # Loop over all time periods
                        Foo.append((s,y,g,t))                          # Collect index for this scenario/year/generator/time
        foo=[]  # Index list for generator energy production in this period
        for s in range(1, model.Scenarios + 1):  # Loop over all scenarios
            for y in years_list:  # Loop over years belonging to step ut
                for r in range(1, model.RES_Sources+1):  # Loop over all RES sources
                    for t in range(1,model.Periods+1):  # Loop over all time periods
                        foo.append((s,y,r,t))          # Collect index for this scenario/year/source/time
        goo=[]  # Index list for grid import in this period
        for s in range(1, model.Scenarios + 1):  # Loop over all scenarios
            for y in years_list:  # Loop over years belonging to step ut
                    for t in range(1,model.Periods+1):  # Loop over all time periods
                        goo.append((s,y,t))  # Collect index for this scenario/year/time
                        
        E_gen = sum(model.Generator_Energy_Production[s,y,g,t]*model.Scenario_Weight[s]  # Probability-weighted generator energy over step ut
                    for s,y,g,t in Foo)  # Sum over the collected scenario/year/generator/time indices
        E_ren = sum(model.RES_Energy_Production[s,y,r,t]*model.Scenario_Weight[s]  # Probability-weighted RES energy over step ut
                    for s,y,r,t in foo)  # Sum over the collected scenario/year/source/time indices
        if model.Grid_Connection == 1:  # Grid-connected: include grid imports in the penetration calculation
            E_From_Grid = sum(model.Energy_From_Grid[s,y,t]*model.Grid_Availability[s,y,t]*model.Scenario_Weight[s]  # Probability-weighted grid imports over step ut
                    for s,y,t in goo)  # Sum over the collected scenario/year/time indices
        else: E_From_Grid = 0  # Off-grid: no imports to include in penetration calculation
 
        return  (1 - model.Renewable_Penetration)*E_ren >= model.Renewable_Penetration*(E_gen + E_From_Grid)     # RES share of total energy supplied must meet the penetration target
    
    def Renewables_Min_Step_Units(model,yt,ut,r):  # RES unit count must not decrease between investment steps (no decommissioning)
        if ut > 1:  # After the first step: value must not decrease from the previous step
            return model.RES_Units[ut,r] >= model.RES_Units[ut-1,r]  # RES units must not decrease from the previous step
        elif ut == 1:  # First step: no prior step to compare against
            return model.RES_Units[ut,r] == model.RES_Units[ut,r]  # First step: trivially satisfied (no previous step)
        
    "Maximum land use constraint for Renewables"
    def Renewables_Max_Land_Use(model,ut):  # Total land footprint of RES (existing + new units) must not exceed available area -- signature fixed 2026-09-04 (was (model,s,yt,ut,r,t), which never matched the Constraint(model.steps, ...) declaration in Model_Resolution.py nor this body's actual dependence on `ut` alone; crashed with "missing 4 required positional arguments" the first time Land_Use=1 was ever exercised, via the linopy Stage 2 cross-validation. Matches Constraints_Greenfield's already-correct (model,ut) signature for the same rule.
        return  (sum((model.RES_existing_area[r] + (model.RES_Units[ut,r]*model.RES_Nominal_Capacity[r]*(model.RES_Specific_Area[r]))) for r in model.renewable_sources)) <= model.Renewables_Total_Area  # Land use = pre-existing footprint + new units' footprint, capped at available area

    "Battery Energy Storage constraints"
    def State_of_Charge(model,s,yt,ut,t): # State of Charge of the battery
        if t==1 and yt==1: # The state of charge (State_Of_Charge) for the period 0 is equal to the Battery size.
            return model.Battery_SOC[s,yt,t] == model.Battery_Nominal_Capacity[ut]*model.Battery_Initial_SOC - model.Battery_Outflow[s,yt,t]/model.Battery_Discharge_Battery_Efficiency + model.Battery_Inflow[s,yt,t]*model.Battery_Charge_Battery_Efficiency  # Initial SOC = starting fraction of capacity, adjusted for first-period flows
        if t==1 and yt!=1:  # First period of a year after the first project year
            return model.Battery_SOC[s,yt,t] == model.Battery_SOC[s,yt-1,model.Periods] - model.Battery_Outflow[s,yt,t]/model.Battery_Discharge_Battery_Efficiency + model.Battery_Inflow[s,yt,t]*model.Battery_Charge_Battery_Efficiency  # Carry over SOC from the last period of the previous year
        else:    # Any period after the first: carry over SOC from the previous period
            return model.Battery_SOC[s,yt,t] == model.Battery_SOC[s,yt,t-1] - model.Battery_Outflow[s,yt,t]/model.Battery_Discharge_Battery_Efficiency + model.Battery_Inflow[s,yt,t]*model.Battery_Charge_Battery_Efficiency      # Carry over SOC from the previous period within the same year
    
    def Maximum_Charge(model,s,yt,ut,t): # Maximun state of charge of the Battery
        return model.Battery_SOC[s,yt,t] <= model.Battery_Nominal_Capacity[ut]  # SOC cannot exceed battery capacity
    
    def Minimum_Charge(model,s,yt,ut,t): # Minimun state of charge
        return model.Battery_SOC[s,yt,t] >= model.Battery_Nominal_Capacity[ut]*(1-model.Battery_Depth_of_Discharge)  # SOC cannot fall below the minimum allowed by the depth of discharge
    
    def Max_Power_Battery_Charge(model,ut):   # Max charge power = capacity / min charge time (C-rate limit)
        return model.Battery_Maximum_Charge_Power[ut] == model.Battery_Nominal_Capacity[ut]/model.Maximum_Battery_Charge_Time  # Max charge power = capacity / minimum charge time
    
    def Max_Power_Battery_Discharge(model,ut):  # Max discharge power = capacity / min discharge time (C-rate limit)
        return model.Battery_Maximum_Discharge_Power[ut] == model.Battery_Nominal_Capacity[ut]/model.Maximum_Battery_Discharge_Time  # Max discharge power = capacity / minimum discharge time
    
    def Max_Bat_flow_in(model,s,yt,ut,t): # Minimun flow of energy for the charge fase
        return model.Battery_Inflow[s,yt,t] <= model.Battery_Maximum_Charge_Power[ut]*model.Delta_Time  # Cap charging power at the maximum charge rate
    
    def Max_Bat_flow_out(model,s,yt,ut,t): # Minimun flow of energy for the discharge fase
        return model.Battery_Outflow[s,yt,t] <= model.Battery_Maximum_Discharge_Power[ut]*model.Delta_Time  # Discharge power cannot exceed the maximum discharge rate
    
    def Max_Bat_out(model,s,yt,ut,t):  # Discharge cannot exceed the current demand
        return model.Battery_Outflow[s,yt,t] <= model.Energy_Demand[s,yt,t]  # Discharge cannot exceed current demand
        
    def Battery_Min_Capacity(model,ut):      # Battery must meet the minimum autonomy requirement
        return   model.Battery_Nominal_Capacity[ut] >= model.Battery_Min_Capacity[ut]  # Battery capacity must meet the minimum autonomy requirement
    
    def Battery_Min_Step_Capacity(model,yt,ut):      # Battery capacity must not shrink between investment steps
        if ut > 1:  # After the first step: value must not decrease from the previous step
            return model.Battery_Nominal_Capacity[ut] >= model.Battery_Nominal_Capacity[ut-1]  # Battery capacity must not decrease from the previous step
        elif ut == 1:  # First step: no prior step to compare against
            return model.Battery_Nominal_Capacity[ut] == model.Battery_Nominal_Capacity[ut]  # First step: trivially satisfied (no previous step)
    
    def Battery_Single_Flow_Discharge(model,s,yt,ut,t):  # Binary flag prevents simultaneous charge and discharge
        return   model.Battery_Outflow[s,yt,t] <= model.Single_Flow_BESS[s,yt,t]*model.Battery_Maximum_Discharge_Power[ut]*model.Delta_Time  # Discharge allowed only when the discharge-mode binary is active
    
    def Battery_Single_Flow_Charge(model,s,yt,ut,t):  # Charge only allowed when not discharging (complementary binary)
        return   model.Battery_Inflow[s,yt,t] <= (1-model.Single_Flow_BESS[s,yt,t])*model.Battery_Maximum_Charge_Power[ut]*model.Delta_Time  # Charge allowed only when not in discharge mode
        
    "Diesel generator constraints"
    def Maximum_Generator_Energy_1(model,s,yt,ut,g,t):   # Cap generator output at installed nominal capacity
        return model.Generator_Energy_Production[s,yt,g,t] <= model.Generator_Nominal_Capacity[ut,g]*model.Delta_Time  # Cap generator output at installed nominal capacity
    
    def Maximum_Generator_Energy_2(model,s,yt,ut,g,t):   # Generator output cannot exceed demand
        return model.Generator_Energy_Production[s,yt,g,t] <= model.Energy_Demand[s,yt,t]*model.Delta_Time  # Generator output cannot exceed demand
    
    def Generator_Min_Step_Capacity(model,yt,ut,g):  # Capacity/unit count must not shrink between investment steps
        if ut > 1:  # After the first step: value must not decrease from the previous step
            return model.Generator_Nominal_Capacity[ut,g] >= model.Generator_Nominal_Capacity[ut-1,g]  # Generator capacity must not decrease from the previous step
        elif ut ==1:  # First step: no prior step to compare against
            return model.Generator_Nominal_Capacity[ut,g] == model.Generator_Nominal_Capacity[ut,g]  # First step: trivially satisfied (no previous step)
    
    "Lost load constraints"
    def Maximum_Lost_Load(model,s,yt): # Maximum admittable lost load
        # Algebraically identical to the original form, which was written as
        #   Lost_Load_Fraction >= sum(Lost_Load) / sum(Energy_Demand)
        # but multiplied through by sum(Energy_Demand) instead of dividing by it.
        # Total annual demand is a constant, so moving it to the right-hand side
        # turns it into an RHS value rather than a matrix coefficient.
        #
        # WHY THIS MATTERS (measured 2026-07-26, not inferred): the divided form put
        # a coefficient of 1/sum(Energy_Demand) on every Lost_Load variable. With
        # demand in Wh that is ~9.9e-09 -- the SMALLEST coefficient anywhere in the
        # model, and the reason the logged matrix range reached [1e-08, 3e+06], a
        # ratio of 3.2e14 against Gurobi's guidance of below 1e9. It came from just
        # 30 rows out of 3.15 million. In this form the Lost_Load coefficients are
        # 1.0, and the worst remaining coefficient is 4.19e-06 in
        # BatteryReplacementCostAct -- roughly a 420x narrowing of the range.
        # Denominator is a sum of demands and is always strictly positive, so the
        # inequality direction is preserved.
        return sum(model.Lost_Load[s,yt,t] for t in model.periods) <= model.Lost_Load_Fraction * sum(model.Energy_Demand[s,yt,t] for t in model.periods)  # Lost load must not exceed the allowed fraction of total demand
    
    "Emission constraints"
    def RES_emission(model): #LCA emissions of RES
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration   # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]    # Will map each project year to its (year, step_index) tuple
        for y in model.years:        # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])            # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))              # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]    # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[model.Step_Duration*i + model.Step_Duration]  # Upgrade decision occurs at the end of each step
            
        return model.RES_emission == sum(model.RES_unit_CO2_emission[r]*((model.RES_Units[1,r]*model.RES_Nominal_Capacity[r])-model.RES_capacity[r]) for r in model.renewable_sources)+sum(sum(model.RES_unit_CO2_emission[r]*(model.RES_Units[ut,r]-model.RES_Units[ut-1,r])*model.RES_Nominal_Capacity[r] for (yt,ut) in tup_list) for r in model.renewable_sources)  # LCA emissions net of credit for pre-existing RES capacity, plus incremental upgrades
    
    def GEN_emission(model): #LCA emissions of generator
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration   # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]    # Will map each project year to its (year, step_index) tuple
        for y in model.years:        # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])            # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))              # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]    # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[model.Step_Duration*i + model.Step_Duration]  # Upgrade decision occurs at the end of each step
            
        return model.GEN_emission == sum((model.Generator_Nominal_Capacity[1,g]-model.Generator_capacity[g])*model.GEN_unit_CO2_emission[g] for g in model.generator_types)+sum(sum((model.Generator_Nominal_Capacity[ut,g]-model.Generator_Nominal_Capacity[ut-1,g])*model.GEN_unit_CO2_emission[g] for (yt,ut) in tup_list) for g in model.generator_types)  # LCA emissions net of credit for pre-existing generator capacity, plus incremental upgrades
    
    def FUEL_emission(model,s,yt,ut,g,t): #Emissions from fuel consumption
        return model.FUEL_emission[s,yt,g,t] == model.Generator_Energy_Production[s,yt,g,t]/model.Fuel_LHV[g]/model.Generator_Efficiency[g]*model.FUEL_unit_CO2_emission[g]   # Emissions = energy / (fuel LHV x generator efficiency) x fuel emission factor
    
    def GRID_emission(model, s, y, t):  # Emissions from grid electricity imports, using the national grid CO2 factor
        if y >= model.Year_Grid_Connection:  # Only count grid emissions once the connection year has been reached
            return model.GRID_emission[s,y,t] == model.Energy_From_Grid[s,y,t] * model.National_Grid_Specific_CO2_emissions  # Import energy x national grid CO2 factor / 1000
        else:  # Before the connection year: no grid emissions
            return model.GRID_emission[s, y, t] == 0  # No grid emissions before the connection year
    
    def BESS_emission(model): #LCA emissions of generator
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration   # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]    # Will map each project year to its (year, step_index) tuple
        for y in model.years:        # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])            # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))              # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]    # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[model.Step_Duration*i + model.Step_Duration]  # Upgrade decision occurs at the end of each step
            
        return model.BESS_emission == (model.Battery_Nominal_Capacity[1]-model.Battery_capacity)*model.BESS_unit_CO2_emission+sum((model.Battery_Nominal_Capacity[ut]-model.Battery_Nominal_Capacity[ut-1])*model.BESS_unit_CO2_emission for (yt,ut) in tup_list)  # LCA emissions net of credit for pre-existing battery capacity, plus incremental upgrades
        
    def Scenario_FUEL_emission(model,s):   # Total fuel-burn CO2 emissions for scenario s, summed over years/generators/periods
        return model.Scenario_FUEL_emission[s] == sum(sum(sum(model.Generator_Energy_Production[s,y,g,t]/model.Fuel_LHV[g]/model.Generator_Efficiency[g]*model.FUEL_unit_CO2_emission[g]  for t in model.periods) for y in model.years) for g in model.generator_types)   # Sum fuel emissions over all periods, years and generator types
    
    def Scenario_GRID_emission(model, s):  # Total grid-import CO2 emissions for scenario s, summed over years/periods
        Total_Grid_Emission = 0  # Accumulator for total grid emissions across all years
        for y in range(1, model.Years + 1):  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only count grid emissions once the connection year has been reached
                Total_Grid_Emission += sum(model.Energy_From_Grid[s, y, t] * model.National_Grid_Specific_CO2_emissions  for t in model.periods)  # Yearly grid emissions = imports x national grid CO2 factor / 1000
        return model.Scenario_GRID_emission[s] == Total_Grid_Emission  # Return total scenario grid emissions constraint
    
    "Grid constraints"  
    def Maximum_Power_From_Grid(model,s,y,t):  # Cap imports from grid; zero before connection year or during outages
        if y < model.Year_Grid_Connection or model.Grid_Availability[s, y, t] == 0:  # Grid not yet connected or currently unavailable
            return model.Energy_From_Grid[s,y,t] == 0  # No imports possible
        else:  # Grid connected and available: cap imports at the max grid power
            return model.Energy_From_Grid[s,y,t] <= model.Maximum_Grid_Power   # Cap imports at maximum grid power (kW to W)
    
    def Maximum_Power_To_Grid(model,s,y,t):  # Cap exports to grid; zero if not allowed or grid unavailable
        if y < model.Year_Grid_Connection or model.Grid_Connection_Type == 1 or model.Grid_Availability[s, y, t] == 0:  # Export not allowed (import-only) or grid unavailable/not yet connected
            return model.Energy_To_Grid[s, y, t] == 0  # No exports possible
        elif model.Grid_Connection_Type == 0:  # Bidirectional connection: exports are allowed
            return model.Energy_To_Grid[s, y, t] <= model.Maximum_Grid_Power  # Cap exports at maximum grid power (kW to W)
        
    def Single_Flow_Energy_To_Grid(model,s,yt,ut,t):  # Binary Single_Flow_Grid prevents exporting while importing
        return model.Energy_To_Grid[s,yt,t] <= model.Single_Flow_Grid[s,yt,t]*model.Large_Constant  # Export allowed only when the export-mode binary is active
        
    def Single_Flow_Energy_From_Grid(model,s,yt,ut,t):  # Complementary binary: import only when not exporting
        return model.Energy_From_Grid[s,yt,t] <= (1-model.Single_Flow_Grid[s,yt,t])*model.Large_Constant  # Import allowed only when the export-mode binary is inactive


#########################################################################################################################################################################################################
############################################### MILP FORMULATION ########################################################################################################################################
#########################################################################################################################################################################################################

     
#%% 

# --- Greenfield ---

class Constraints_Greenfield_Milp():  # MILP formulation — greenfield; RES/battery/generator sized in discrete integer units
    current_directory = os.path.dirname(os.path.abspath(__file__))  # Folder containing this module
    inputs_directory = os.path.join(current_directory, '..', 'Inputs')  # Sibling Inputs folder
    data_file_path = params_path.PARAMS_PATH  # Resolved once via MGPY_PARAMS; see params_path.py
    Data_import = open(data_file_path).readlines()  # Read the parameters file at class-definition time
    for i in range(len(Data_import)):  # Scan every line of the parameters file
     if "param: Generator_Partial_Load" in Data_import[i]:       # Locate partial-load flag
        Generator_Partial_Load = int((re.findall('\d+',Data_import[i])[0]))  # 1 = enable partial-load MILP constraints
     if "param: Fuel_Specific_Cost_Calculation" in Data_import[i]:       # Locate fuel-cost flag
            Fuel_Specific_Cost_Calculation = int((re.findall('\d+',Data_import[i])[0]))  # 0 = constant price, 1 = time-varying
        
    "Objective function"
    def Net_Present_Cost_Obj(model):   # Pyomo objective wrapper for total Net Present Cost
        return (sum(model.Scenario_Net_Present_Cost[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected NPC = probability-weighted sum across scenarios
    
    def CO2_emission_Obj(model):  # Pyomo objective wrapper for total CO2 emissions
        return (sum(model.Scenario_CO2_emission[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected CO2 emissions = probability-weighted sum across scenarios
    
    def Total_Variable_Cost_Obj(model):  # Pyomo objective wrapper for total variable cost
        return (sum(model.Total_Scenario_Variable_Cost_NonAct[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected non-actualized variable cost = probability-weighted sum across scenarios
    
    
    "Net Present Cost"
    def Net_Present_Cost(model):     # Defines the scalar Net Present Cost as expected value over all scenarios
        return model.Net_Present_Cost == (sum(model.Scenario_Net_Present_Cost[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected NPC = probability-weighted sum across scenarios
    
    # def Scenario_Net_Present_Cost(model,s): 
    #     foo = []
    #     for g in range(1,model.Generator_Types+1):
    #             foo.append((s,g))            
    #     Fuel_Cost = sum(model.Total_Fuel_Cost_Act[s,g] for s,g in foo)    
    #     return model.Scenario_Net_Present_Cost[s] == (model.Investment_Cost + model.Operation_Maintenance_Cost_Act + model.Battery_Replacement_Cost_Act[s] 
    #             + model.Scenario_Lost_Load_Cost_Act[s] + Fuel_Cost - model.Salvage_Value)   
    def Total_Variable_Cost(model):  # Defines the scalar total variable cost (non-actualized)
        return model.Total_Variable_Cost == (sum(model.Total_Scenario_Variable_Cost_NonAct[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected non-actualized variable cost = probability-weighted sum across scenarios
    
    def Scenario_Net_Present_Cost(model,s):   # NPC for scenario s: CAPEX + variable OPEX - salvage value
        return model.Scenario_Net_Present_Cost[s] == (model.Investment_Cost + model.Total_Scenario_Variable_Cost_Act[s] - model.Salvage_Value)     # NPC = CAPEX + actualized variable cost - salvage value
    
    def CO2_emission(model):  # Defines the scalar CO2 emission as expected value over all scenarios
        return model.CO2_emission == (sum(model.Scenario_CO2_emission[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected CO2 emissions = probability-weighted sum across scenarios
    
    def Scenario_CO2_emission(model,s):  # Total lifecycle + operational CO2 for scenario s; components depend on active model parts
        if model.Grid_Connection == 1:  # Grid-connected: include grid import emissions
            if model.Model_Components == 0:  # Full system: RES + generator + battery
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.GEN_emission + model.BESS_emission + model.Scenario_FUEL_emission[s] + model.Scenario_GRID_emission[s])  # Full system with grid: RES + generator + battery + fuel + grid import emissions
            if model.Model_Components == 1:  # RES + battery only (no generator)
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.BESS_emission + model.Scenario_GRID_emission[s])  # Grid-connected, no generator: RES + battery + grid import emissions
            if model.Model_Components == 2:  # RES + generator only (no battery)
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.GEN_emission + model.Scenario_FUEL_emission[s] + model.Scenario_GRID_emission[s])  # Grid-connected, no battery: RES + generator + fuel + grid import emissions
        else:  # Off-grid: no grid-related emissions to add
            if model.Model_Components == 0:  # Full system: RES + generator + battery
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.GEN_emission + model.BESS_emission + model.Scenario_FUEL_emission[s])  # Off-grid full system: RES + generator + battery + fuel emissions
            if model.Model_Components == 1:  # RES + battery only (no generator)
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.BESS_emission)  # Off-grid, no generator: RES + battery emissions only
            if model.Model_Components == 2:  # RES + generator only (no battery)
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.GEN_emission + model.Scenario_FUEL_emission[s])  # Off-grid, no battery: RES + generator + fuel emissions
    
    "Investment cost"
    def Investment_Cost(model):    # Total discounted CAPEX for RES + generator + battery + grid connection
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration   # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]    # Will map each project year to its (year, step_index) tuple
        for y in model.years:        # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])            # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))              # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]    # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[model.Step_Duration*i + model.Step_Duration]        # Upgrade decision occurs at the end of each step
      
        Inv_Ren = sum((model.RES_Units_milp[1,r]*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r])  # Step-1 RES CAPEX using integer unit count
                        + sum((((model.RES_Units_milp[ut,r] - model.RES_Units_milp[ut-1,r])*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r]))/((1+model.Discount_Rate)**(yt-1))  # NPV of incremental integer RES units added
                        for (yt,ut) in tup_list) for r in model.renewable_sources)  # Sum incremental RES investment NPV over all upgrade points
        Inv_Gen = sum(((model.Generator_Units[1,g]*model.Generator_Nominal_Capacity_milp[g])*model.Generator_Specific_Investment_Cost[g])  # Step-1 generator CAPEX: integer unit count × nominal unit size
                        + sum(((((model.Generator_Units[ut,g]*model.Generator_Nominal_Capacity_milp[g]) - (model.Generator_Units[ut-1,g]*model.Generator_Nominal_Capacity_milp[g]))*model.Generator_Specific_Investment_Cost[g]))/((1+model.Discount_Rate)**(yt-1))  # NPV of incremental integer generator units
                        for (yt,ut) in tup_list) for g in model.generator_types)  # Sum incremental generator investment NPV over all upgrade points
        Inv_Bat = ((model.Battery_Units[1]*model.Battery_Nominal_Capacity_milp*model.Battery_Specific_Investment_Cost)  # Step-1 battery CAPEX: integer unit count × nominal unit capacity
                        + sum((((model.Battery_Units[ut] - model.Battery_Units[ut-1])*model.Battery_Nominal_Capacity_milp*model.Battery_Specific_Investment_Cost))/((1+model.Discount_Rate)**(yt-1))  # NPV of incremental integer battery units
                        for (yt,ut) in tup_list))   # Sum incremental battery investment NPV over all upgrade points
        Inv_Grid = 0  # Accumulator for NPV of grid connection investment
        for y in model.years:  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur the grid connection cost once the connection year has been reached
                Inv_Grid += (model.Grid_Connection_Cost * model.Grid_Distance)/((1+model.Discount_Rate)**(y-1))   # Discount grid connection cost to year 0 and accumulate
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            investment_cost = Inv_Ren + Inv_Gen + Inv_Bat  # Full system: RES + generator + battery CAPEX
            if model.Grid_Connection: investment_cost += Inv_Grid  # Add grid connection cost if enabled
            return model.Investment_Cost == investment_cost   # Return the assembled CAPEX constraint
        if model.Model_Components == 1:  # RES + battery only (no generator)
            investment_cost = Inv_Ren + Inv_Bat  # RES + battery CAPEX (no generator)
            if model.Grid_Connection: investment_cost += Inv_Grid  # Add grid connection cost if enabled
            return model.Investment_Cost == investment_cost        # Return the assembled CAPEX constraint
        if model.Model_Components == 2:  # RES + generator only (no battery)
            investment_cost = Inv_Ren + Inv_Gen  # RES + generator CAPEX (no battery)
            if model.Grid_Connection: investment_cost += Inv_Grid  # Add grid connection cost if enabled
            return model.Investment_Cost == investment_cost    # Return the assembled CAPEX constraint

    def Investment_Cost_Limit(model):  # Enforce a user-defined cap on total CAPEX
        return model.Investment_Cost <= model.Investment_Cost_Limit  # Total CAPEX must not exceed the limit
       
    "Fixed O&M costs"
    def Operation_Maintenance_Cost_Act(model):  # Actualized (discounted) fixed O&M cost of RES + generator + battery + grid
        OyM_Ren = sum(sum((model.RES_Units_milp[ut,r]*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r]*model.RES_Specific_OM_Cost[r])/((  # NPV of RES fixed O&M cost (integer units), discounted per year it is incurred
                        1+model.Discount_Rate)**yt)for (yt,ut) in model.years_steps)for r in model.renewable_sources)  # Sum discounted RES O&M cost over all years/steps
        OyM_Gen = sum(sum(((model.Generator_Units[ut,g]*model.Generator_Nominal_Capacity_milp[g])*model.Generator_Specific_Investment_Cost[g]*model.Generator_Specific_OM_Cost[g])/((  # NPV of generator fixed O&M cost (integer units), discounted per year it is incurred
                        1+model.Discount_Rate)**yt)for (yt,ut) in model.years_steps)for g in model.generator_types)  # Sum discounted generator O&M cost over all years/steps
        OyM_Bat = sum((model.Battery_Units[ut]*model.Battery_Nominal_Capacity_milp*model.Battery_Specific_Investment_Cost*model.Battery_Specific_OM_Cost)/((  # NPV of battery fixed O&M cost (integer units), discounted per year it is incurred
                        1+model.Discount_Rate)**yt)for (yt,ut) in model.years_steps)  # Sum discounted battery O&M cost over all years/steps
        OyM_Grid = 0  # Accumulator for grid connection O&M cost
        for y in model.years:  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur grid O&M cost once the connection year has been reached
                OyM_Grid += (model.Grid_Connection_Cost * model.Grid_Distance * model.Grid_Maintenance_Cost)/((1+model.Discount_Rate)**(y))   # Discounted grid maintenance cost, only once grid is connected
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            cost = OyM_Ren + OyM_Gen + OyM_Bat  # Full system: RES + generator + battery O&M
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_Act == cost  # Return the assembled actualized O&M constraint
        if model.Model_Components == 1:  # RES + battery only (no generator)
            cost = OyM_Ren + OyM_Bat  # RES + battery O&M (no generator)
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_Act == cost  # Return the assembled actualized O&M constraint
        if model.Model_Components == 2:  # RES + generator only (no battery)
            cost = OyM_Ren + OyM_Gen  # RES + generator O&M (no battery)
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_Act == cost   # Return the assembled actualized O&M constraint

    
    def Operation_Maintenance_Cost_NonAct(model):  # Non-actualized (undiscounted) fixed O&M cost, mirrors the Act version
        OyM_Ren = sum(sum((model.RES_Units_milp[ut,r]*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r]*model.RES_Specific_OM_Cost[r])  # Undiscounted RES fixed O&M cost (integer units)
                        for (yt,ut) in model.years_steps)for r in model.renewable_sources)  # Sum undiscounted RES O&M cost over all years/steps
        OyM_Gen = sum(sum(((model.Generator_Units[ut,g]*model.Generator_Nominal_Capacity_milp[g])*model.Generator_Specific_Investment_Cost[g]*model.Generator_Specific_OM_Cost[g])  # Undiscounted generator fixed O&M cost (integer units)
                        for (yt,ut) in model.years_steps)for g in model.generator_types)  # Sum undiscounted generator O&M cost over all years/steps
        OyM_Bat = sum((model.Battery_Units[ut]*model.Battery_Nominal_Capacity_milp*model.Battery_Specific_Investment_Cost*model.Battery_Specific_OM_Cost)  # Undiscounted battery fixed O&M cost (integer units)
                        for (yt,ut) in model.years_steps)  # Sum undiscounted battery O&M cost over all years/steps
        OyM_Grid = 0  # Accumulator for grid connection O&M cost
        for y in model.years:  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur grid O&M cost once the connection year has been reached
                OyM_Grid += (model.Grid_Connection_Cost * model.Grid_Distance * model.Grid_Maintenance_Cost)  # Undiscounted grid maintenance cost, only once grid is connected
        
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            cost = OyM_Ren + OyM_Gen + OyM_Bat  # Full system: RES + generator + battery O&M
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_NonAct == cost   # Return the assembled non-actualized O&M constraint
        if model.Model_Components == 1:  # RES + battery only (no generator)
            cost = OyM_Ren + OyM_Bat  # RES + battery O&M (no generator)
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_NonAct == cost   # Return the assembled non-actualized O&M constraint
        if model.Model_Components == 2:  # RES + generator only (no battery)
            cost = OyM_Ren + OyM_Gen  # RES + generator O&M (no battery)
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_NonAct == cost    # Return the assembled non-actualized O&M constraint

    "Variable costs"
    def Total_Variable_Cost_Act(model):  # Scalar actualized total variable cost = expected value over all scenarios
        return model.Total_Variable_Cost_Act == (sum(model.Total_Scenario_Variable_Cost_Act[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Probability-weighted sum across scenarios
    
    def Scenario_Variable_Cost_Act(model, s):  # Actualized variable cost for scenario s: O&M + battery replacement + lost load + fuel + net electricity cost
        foo = []  # Index list accumulator
        for g in range(1,model.Generator_Types+1):  # Loop over all generator types for this scenario
                foo.append((s,g))     # Collect (scenario, generator) index pair
        Fuel_Cost = sum(model.Total_Fuel_Cost_Act[s,g] for s,g in foo)     # Total actualized fuel cost across all generator types
        if model.Grid_Connection == 1:  # Grid-connected: include electricity purchase cost (and sale revenue if allowed)
            Electricity_Cost = model.Total_Electricity_Cost_Act[s]   # Actualized cost of electricity purchased from the grid
            if model.Grid_Connection_Type == 0:  # Bidirectional connection: exports earn revenue
                Electricity_Revenues = model.Total_Revenues_Act[s]  # Actualized revenue from electricity sold to the grid
            else: Electricity_Revenues = 0  # Import-only connection: no export revenue
        else:  # Off-grid: no electricity purchase or export cost
            Electricity_Cost = 0  # Off-grid: no electricity purchase cost
            Electricity_Revenues = 0  # Off-grid: no electricity sale revenue
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            return model.Total_Scenario_Variable_Cost_Act[s] == model.Operation_Maintenance_Cost_Act + model.Battery_Replacement_Cost_Act[s] + model.Scenario_Lost_Load_Cost_Act[s] + Fuel_Cost + Electricity_Cost - Electricity_Revenues  # Full system: O&M + battery replacement + lost load + fuel + net electricity
        if model.Model_Components == 1:  # RES + battery only (no generator)
            return model.Total_Scenario_Variable_Cost_Act[s] == model.Operation_Maintenance_Cost_Act + model.Battery_Replacement_Cost_Act[s] + model.Scenario_Lost_Load_Cost_Act[s] + Electricity_Cost - Electricity_Revenues  # No generator: O&M + battery replacement + lost load + net electricity
        if model.Model_Components == 2:  # RES + generator only (no battery)
            return model.Total_Scenario_Variable_Cost_Act[s] == model.Operation_Maintenance_Cost_Act + model.Scenario_Lost_Load_Cost_Act[s] + Fuel_Cost + Electricity_Cost - Electricity_Revenues  # No battery: O&M + lost load + fuel + net electricity
    
    def Scenario_Variable_Cost_NonAct(model, s):   # Non-actualized (undiscounted) mirror of Scenario_Variable_Cost_Act
        foo = []  # Index list accumulator
        for g in range(1,model.Generator_Types+1):  # Loop over all generator types for this scenario
            foo.append((s,g))     # Collect (scenario, generator) index pair
        Fuel_Cost = sum(model.Total_Fuel_Cost_NonAct[s,g] for s,g in foo)    # Total undiscounted fuel cost across all generator types
        if model.Grid_Connection == 1:  # Grid-connected: include electricity purchase cost (and sale revenue if allowed)
            Electricity_Cost = model.Total_Electricity_Cost_NonAct[s]   # Undiscounted cost of electricity purchased from the grid
            if model.Grid_Connection_Type == 0:  # Bidirectional connection: exports earn revenue
                Electricity_Revenues = model.Total_Revenues_NonAct[s]  # Undiscounted revenue from electricity sold to the grid
            else: Electricity_Revenues = 0  # Import-only connection: no export revenue
        else:  # Off-grid: no electricity purchase or export cost
            Electricity_Cost = 0  # Off-grid: no electricity purchase cost
            Electricity_Revenues = 0  # Off-grid: no electricity sale revenue
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            return model.Total_Scenario_Variable_Cost_NonAct[s] == model.Operation_Maintenance_Cost_NonAct + model.Battery_Replacement_Cost_NonAct[s] + model.Scenario_Lost_Load_Cost_NonAct[s] + Fuel_Cost + Electricity_Cost - Electricity_Revenues  # Full system: O&M + battery replacement + lost load + fuel + net electricity
        if model.Model_Components == 1:  # RES + battery only (no generator)
            return model.Total_Scenario_Variable_Cost_NonAct[s] == model.Operation_Maintenance_Cost_NonAct + model.Battery_Replacement_Cost_NonAct[s] + model.Scenario_Lost_Load_Cost_NonAct[s] + Electricity_Cost - Electricity_Revenues  # No generator: O&M + battery replacement + lost load + net electricity
        if model.Model_Components == 2:  # RES + generator only (no battery)
            return model.Total_Scenario_Variable_Cost_NonAct[s] == model.Operation_Maintenance_Cost_NonAct + model.Scenario_Lost_Load_Cost_NonAct[s] + Fuel_Cost + Electricity_Cost - Electricity_Revenues  # No battery: O&M + lost load + fuel + net electricity
    
    def Scenario_Lost_Load_Cost_Act(model,s):      # Actualized cost of unserved (lost load) energy for scenario s
        Cost_Lost_Load = 0           # Accumulator for discounted yearly lost-load cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(model.Lost_Load[s,y,t]*model.Lost_Load_Specific_Cost for t in model.periods)  # Yearly lost-load cost = unserved energy x penalty price
            Cost_Lost_Load += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
        return  model.Scenario_Lost_Load_Cost_Act[s] == Cost_Lost_Load  # Return actualized lost-load cost constraint
    
    def Scenario_Lost_Load_Cost_NonAct(model,s):  # Non-actualized (undiscounted) mirror of Scenario_Lost_Load_Cost_Act
        Cost_Lost_Load = 0           # Accumulator for discounted yearly lost-load cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(model.Lost_Load[s,y,t]*model.Lost_Load_Specific_Cost for t in model.periods)  # Yearly lost-load cost = unserved energy x penalty price
            Cost_Lost_Load += Num  # Accumulate undiscounted yearly lost-load cost
        return  model.Scenario_Lost_Load_Cost_NonAct[s] == Cost_Lost_Load  # Return non-actualized lost-load cost constraint
    
    if Generator_Partial_Load == 1 and Fuel_Specific_Cost_Calculation == 0:  # Partial-load MILP with constant fuel price
     "Partial Load Effect"
     def Total_Fuel_Cost_Act(model,s,g):  # Cost includes: full-load run cost + partial-load marginal + start-up cost
        Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(((model.Generator_Full[s,y,g,t]*model.Generator_Nominal_Capacity_milp[g]*model.Generator_Marginal_Cost_1[g])  # Full-load blocks: integer units fully committed × rated capacity × marginal cost
                      +(model.Generator_Marginal_Cost_milp_1[g]*model.Generator_Energy_Partial[s,y,g,t])  # Partial-load energy at adjusted (lower) marginal cost
                      +(model.Generator_Partial[s,y,g,t]*model.Generator_Start_Cost_1[g])) for t in model.periods)  # Start-up events × start-up cost
            Fuel_Cost_Tot += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
        return model.Total_Fuel_Cost_Act[s,g] == Fuel_Cost_Tot  # Return actualized fuel cost constraint
       
     def Total_Fuel_Cost_NonAct(model,s,g):  # Undiscounted version of partial-load fuel cost with constant price
        Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(((model.Generator_Full[s,y,g,t]*model.Generator_Nominal_Capacity_milp[g]*model.Generator_Marginal_Cost_1[g])+(model.Generator_Marginal_Cost_milp_1[g]*model.Generator_Energy_Partial[s,y,g,t])+(model.Generator_Partial[s,y,g,t]*model.Generator_Start_Cost_1[g])) for t in model.periods)  # Full-load blocks + partial-load energy + start-up cost, all at constant prices
            Fuel_Cost_Tot += Num  # Accumulate undiscounted yearly fuel cost
        return model.Total_Fuel_Cost_NonAct[s,g] == Fuel_Cost_Tot  # Return non-actualized fuel cost constraint
    elif Generator_Partial_Load == 1 and Fuel_Specific_Cost_Calculation == 1:  # Partial-load MILP with time-varying fuel price
     def Total_Fuel_Cost_Act(model,s,g):  # Cost includes: full-load + partial-load + start-up; year-indexed prices
        Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(((model.Generator_Full[s,y,g,t]*model.Generator_Nominal_Capacity_milp[g]*model.Generator_Marginal_Cost[g,y])  # Full-load blocks at year-specific marginal cost
                      +(model.Generator_Marginal_Cost_milp[g,y]*model.Generator_Energy_Partial[s,y,g,t])  # Partial-load energy at year-specific adjusted cost
                      +(model.Generator_Partial[s,y,g,t]*model.Generator_Start_Cost[g,y])) for t in model.periods)  # Start-up events at year-specific start cost
            Fuel_Cost_Tot += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
        return model.Total_Fuel_Cost_Act[s,g] == Fuel_Cost_Tot  # Return actualized fuel cost constraint
       
     def Total_Fuel_Cost_NonAct(model,s,g):  # Undiscounted version with time-varying price
        Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(((model.Generator_Full[s,y,g,t]*model.Generator_Nominal_Capacity_milp[g]*model.Generator_Marginal_Cost[g,y])+(model.Generator_Marginal_Cost_milp[g,y]*model.Generator_Energy_Partial[s,y,g,t])+(model.Generator_Partial[s,y,g,t]*model.Generator_Start_Cost[g,y])) for t in model.periods)  # Full-load blocks + partial-load energy + start-up cost, all at year-specific prices
            Fuel_Cost_Tot += Num  # Accumulate undiscounted yearly fuel cost
        return model.Total_Fuel_Cost_NonAct[s,g] == Fuel_Cost_Tot  # Return non-actualized fuel cost constraint
    elif Generator_Partial_Load == 0 and Fuel_Specific_Cost_Calculation == 0:  # No partial load, constant fuel price (simplest MILP fuel cost)
     def Total_Fuel_Cost_Act(model,s,g):  # Actualized fuel cost for generator type g across the project horizon
        Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(model.Generator_Energy_Total[s,y,g,t]*model.Generator_Marginal_Cost_1[g] for t in model.periods)  # Total energy × constant marginal cost
            Fuel_Cost_Tot += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
        return model.Total_Fuel_Cost_Act[s,g] == Fuel_Cost_Tot  # Return actualized fuel cost constraint
       
     def Total_Fuel_Cost_NonAct(model,s,g):  # Non-actualized (undiscounted) mirror of Total_Fuel_Cost_Act
        Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(model.Generator_Energy_Total[s,y,g,t]*model.Generator_Marginal_Cost_1[g] for t in model.periods)  # Total energy x constant marginal cost
            Fuel_Cost_Tot += Num  # Accumulate undiscounted yearly fuel cost
        return model.Total_Fuel_Cost_NonAct[s,g] == Fuel_Cost_Tot  # Return non-actualized fuel cost constraint
    elif Generator_Partial_Load == 0 and Fuel_Specific_Cost_Calculation == 1:  # No partial load, time-varying fuel price
     def Total_Fuel_Cost_Act(model,s,g):  # Actualized fuel cost for generator type g across the project horizon
        Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(model.Generator_Energy_Total[s,y,g,t]*model.Generator_Marginal_Cost[g,y] for t in model.periods)  # Total energy × year-specific marginal cost
            Fuel_Cost_Tot += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
        return model.Total_Fuel_Cost_Act[s,g] == Fuel_Cost_Tot  # Return actualized fuel cost constraint
       
     def Total_Fuel_Cost_NonAct(model,s,g):  # Non-actualized (undiscounted) mirror of Total_Fuel_Cost_Act
        Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(model.Generator_Energy_Total[s,y,g,t]*model.Generator_Marginal_Cost[g,y] for t in model.periods)  # Total energy x year-specific marginal cost
            Fuel_Cost_Tot += Num  # Accumulate undiscounted yearly fuel cost
        return model.Total_Fuel_Cost_NonAct[s,g] == Fuel_Cost_Tot  # Return non-actualized fuel cost constraint

    def Total_Electricity_Cost_Act(model,s):  # Actualized cost of electricity purchased from the national grid (MILP Greenfield)
        Electricity_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly electricity cost
        for y in range(1, model.Years +1):  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur grid purchase cost once the connection year has been reached
                Num = sum(model.Energy_From_Grid[s,y,t]*model.Grid_Availability[s,y,t]*model.Grid_Purchased_El_Price  for t in model.periods)  # Wh × USD/kWh ÷ 1000 → USD
                Electricity_Cost_Tot += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
        return model.Total_Electricity_Cost_Act[s] == Electricity_Cost_Tot  # Return actualized electricity cost constraint
       
    def Total_Electricity_Cost_NonAct(model,s):   # Non-actualized (undiscounted) mirror of Total_Electricity_Cost_Act
        Electricity_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly electricity cost
        for y in range(1, model.Years +1):  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur grid purchase cost once the connection year has been reached
                Electricity_Cost_Tot += sum(model.Energy_From_Grid[s,y,t]*model.Grid_Availability[s,y,t]*model.Grid_Purchased_El_Price  for t in model.periods)  # Accumulate undiscounted yearly electricity cost
        return model.Total_Electricity_Cost_NonAct[s] == Electricity_Cost_Tot  # Return non-actualized electricity cost constraint
    
    
    def Total_Revenues_NonAct(model, s):   # Undiscounted revenue from electricity exported to the grid
        Revenues_Yearly = [0 for _ in model.years]  # Per-year export revenue, zero before grid connection
        for y in range(1, model.Years + 1):  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only count grid export revenue once the connection year has been reached
                Revenues_Yearly[y - 1] = sum(model.Energy_To_Grid[s, y, t] * model.Grid_Availability[s, y, t] * model.Grid_Sold_El_Price  for t in model.periods)  # Yearly revenue = grid exports (masked by availability) x sale price / 1000
        return model.Total_Revenues_NonAct[s] == sum(Revenues_Yearly[y - 1] for y in model.years)  # Sum undiscounted yearly revenues

    def Total_Revenues_Act(model, s):   # Actualized (discounted) revenue from electricity exported to the grid
        Revenues_Yearly = [0 for _ in model.years]  # Per-year export revenue, zero before grid connection
        for y in range(1, model.Years + 1):  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only count grid export revenue once the connection year has been reached
                Revenues_Yearly[y - 1] = sum(model.Energy_To_Grid[s, y, t] * model.Grid_Availability[s, y, t] * model.Grid_Sold_El_Price  for t in model.periods)  # Yearly revenue = grid exports (masked by availability) x sale price / 1000
        return model.Total_Revenues_Act[s] == sum(Revenues_Yearly[y - 1] / ((1 + model.Discount_Rate) ** y) for y in model.years)  # Discount each year's export revenue to year 0 and sum
    
    def Battery_Replacement_Cost_Act(model,s):  # Actualized cost of replacing battery cells as they cycle (charge/discharge wear)
        Battery_cost_in = [0 for y in model.years]  # Per-year cost attributable to charging throughput
        Battery_cost_out = [0 for y in model.years]  # Per-year cost attributable to discharging throughput
        Battery_Yearly_cost = [0 for y in model.years]      # Per-year total replacement cost (charge + discharge)
        for y in range(1,model.Years+1):      # Loop over every project year
            Battery_cost_in[y-1] = sum(model.Battery_Inflow[s,y,t]*model.Unitary_Battery_Replacement_Cost for t in model.periods)  # Charging energy x unitary replacement cost, summed over the year
            Battery_cost_out[y-1] = sum(model.Battery_Outflow[s,y,t]*model.Unitary_Battery_Replacement_Cost for t in model.periods)  # Discharging energy x unitary replacement cost, summed over the year
            Battery_Yearly_cost[y-1] = Battery_cost_out[y-1] + Battery_cost_in[y-1]  # Total yearly replacement cost = charge + discharge wear
        return model.Battery_Replacement_Cost_Act[s] == sum(Battery_Yearly_cost[y-1]/((1+model.Discount_Rate)**y) for y in model.years)   # Discount each year's replacement cost to year 0 and sum
        
    def Battery_Replacement_Cost_NonAct(model,s):  # Non-actualized (undiscounted) mirror of Battery_Replacement_Cost_Act
        Battery_cost_in = [0 for y in model.years]  # Per-year cost attributable to charging throughput
        Battery_cost_out = [0 for y in model.years]  # Per-year cost attributable to discharging throughput
        Battery_Yearly_cost = [0 for y in model.years]      # Per-year total replacement cost (charge + discharge)
        for y in range(1,model.Years+1):      # Loop over every project year
            Battery_cost_in[y-1] = sum(model.Battery_Inflow[s,y,t]*model.Unitary_Battery_Replacement_Cost for t in model.periods)  # Charging energy x unitary replacement cost, summed over the year
            Battery_cost_out[y-1] = sum(model.Battery_Outflow[s,y,t]*model.Unitary_Battery_Replacement_Cost for t in model.periods)  # Discharging energy x unitary replacement cost, summed over the year
            Battery_Yearly_cost[y-1] = Battery_cost_out[y-1] + Battery_cost_in[y-1]  # Total yearly replacement cost = charge + discharge wear
        return model.Battery_Replacement_Cost_NonAct[s] == sum(Battery_Yearly_cost[y-1] for y in model.years)   # Sum undiscounted yearly replacement cost
    
    
    "Salvage Value"
    def _Salvage_Value_Raw(model):     # Residual (undepreciated) value of all assets at project end, discounted to year 0 -- NOT floored at zero, see Salvage_Value_Upper_Raw/Flag below
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration  # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur  # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]  # Will map each project year to its (year, step_index) tuple
        for y in model.years:      # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])          # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))            # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]  # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[s_dur*i + s_dur]  # Upgrade decision occurs at the end of each step

        if model.Steps_Number == 1:      # Single investment step: only the initial build can have residual value
            SV_Ren_1 = sum(model.RES_Units_milp[1,r]*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]-model.Years)/model.RES_Lifetime[r] /   # Residual value of step-1 RES capacity (no pre-existing capacity to credit)
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)          # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_2 = 0   # No second-step RES capacity exists yet
            SV_Ren_3 = 0    # No later-step RES capacity exists yet
            SV_Gen_1 = sum((model.Generator_Units[1,g]*model.Generator_Nominal_Capacity_milp[g])*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]-model.Years)/model.Generator_Lifetime[g] /   # Residual value of step-1 generator capacity (no pre-existing capacity to credit)
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)  # Discount generator residual value to year 0, summed over generator types
            SV_Gen_2 = 0  # No second-step generator capacity exists yet
            SV_Gen_3 = 0  # No later-step generator capacity exists yet
            SV_Grid = model.Grid_Distance*model.Grid_Connection_Cost*model.Grid_Connection / ((1 + model.Discount_Rate)**(model.Years - model.Year_Grid_Connection))   # Discounted residual value of the grid connection investment
        if model.Steps_Number == 2:      # Two investment steps: initial build + one upgrade can have residual value
            yt_last_up = upgrade_years_list[1]         # Year of the (only) capacity upgrade
            SV_Ren_1 = sum(model.RES_Units_milp[1,r]*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]-model.Years)/model.RES_Lifetime[r] /   # Residual value of step-1 RES capacity (no pre-existing capacity to credit)
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)          # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_2 = sum((model.RES_Units_milp[2,r]-model.RES_Units_milp[1,r])*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]+(yt_last_up-1)-model.Years)/model.RES_Lifetime[r] /   # Residual value of the step-2 RES capacity increment
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)          # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_3 = 0  # No later-step RES capacity exists yet
            SV_Gen_1 = sum((model.Generator_Units[1,g]*model.Generator_Nominal_Capacity_milp[g])*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]-model.Years)/model.Generator_Lifetime[g] /   # Residual value of step-1 generator capacity (no pre-existing capacity to credit)
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)          # Discount generator residual value to year 0, summed over generator types
            SV_Gen_2 = sum((model.Generator_Units[2,g]-model.Generator_Units[1,g])*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]+(yt_last_up-1)-model.Years)/model.Generator_Lifetime[g] /   # Residual value of the step-2 generator capacity increment
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)  # Discount generator residual value to year 0, summed over generator types
            SV_Gen_3 = 0  # No later-step generator capacity exists yet
            SV_Grid = model.Grid_Distance*model.Grid_Connection_Cost*model.Grid_Connection / ((1 + model.Discount_Rate)**(model.Years - model.Year_Grid_Connection))   # Discounted residual value of the grid connection investment
            
        if model.Steps_Number > 2:  # Three or more steps: initial build + last upgrade + all intermediate upgrades
            tup_list_2 = [[] for i in range(len(model.steps)-2)]  # (year, step) pairs for upgrades before the last one
            for i in range(len(model.steps) - 2):  # Build the intermediate-upgrade list (excludes the final upgrade)
                tup_list_2[i] = yu_tuples_list[s_dur*i + s_dur]  # Intermediate upgrade occurs at the end of each step
            yt_last_up = upgrade_years_list[-1]  # Year of the final capacity upgrade
            ut_last_up = tup_list[-1][1]      # Step index of the final upgrade
            ut_seclast_up = tup_list[-2][1]  # Step index of the second-to-last upgrade
    
            SV_Ren_1 = sum(model.RES_Units_milp[1,r]*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]-model.Years)/model.RES_Lifetime[r] /   # Residual value of step-1 RES capacity (no pre-existing capacity to credit)
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)      # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_2 = sum((model.RES_Units_milp[ut_last_up,r] - model.RES_Units_milp[ut_seclast_up,r])*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]+(yt_last_up-1)-model.Years)/model.RES_Lifetime[r] /   # Residual value of the final RES capacity increment
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)  # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_3 = sum(sum((model.RES_Units_milp[ut,r] - model.RES_Units_milp[ut-1,r])*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]+(yt-1)-model.Years)/model.RES_Lifetime[r] /   # Residual value of all intermediate RES capacity increments
                            ((1+model.Discount_Rate)**model.Years) for (yt,ut) in tup_list_2) for r in model.renewable_sources)  # Sum intermediate-upgrade RES residual value, discounted to year 0
            SV_Gen_1 = sum((model.Generator_Units[1,g]*model.Generator_Nominal_Capacity_milp[g])*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]-model.Years)/model.Generator_Lifetime[g] /   # Residual value of step-1 generator capacity (no pre-existing capacity to credit)
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)  # Discount generator residual value to year 0, summed over generator types
            SV_Gen_2 = sum((model.Generator_Units[ut_last_up,g] - model.Generator_Units[ut_seclast_up,g])*model.Generator_Nominal_Capacity_milp[g]*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]+(yt_last_up-1)-model.Years)/model.Generator_Lifetime[g] /   # Residual value of the final generator capacity increment
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)  # Discount generator residual value to year 0, summed over generator types
            SV_Gen_3 = sum(sum((model.Generator_Units[ut,g] - model.Generator_Units[ut-1,g])*model.Generator_Nominal_Capacity_milp[g]*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]+(yt-1)-model.Years)/model.Generator_Lifetime[g] /   # Residual value of all intermediate generator capacity increments
                            ((1+model.Discount_Rate)**model.Years) for (yt,ut) in tup_list_2) for g in model.generator_types)  # Sum intermediate-upgrade generator residual value, discounted to year 0
            SV_Grid = model.Grid_Distance*model.Grid_Connection_Cost*model.Grid_Connection / ((1 + model.Discount_Rate)**(model.Years - model.Year_Grid_Connection))   # Discounted residual value of the grid connection investment
            
        if model.Model_Components == 0 or model.Model_Components == 2:  # System includes a generator: credit RES + generator + grid salvage value
            return SV_Ren_1 + SV_Gen_1 + SV_Ren_2 + SV_Gen_2 + SV_Ren_3 + SV_Gen_3 + SV_Grid   # Sum residual value of all RES/generator vintages plus grid connection
        if model.Model_Components == 1:  # RES + battery only (no generator)
            return SV_Ren_1 + SV_Ren_2 + SV_Ren_3 + SV_Grid   # No generator: sum residual value of RES vintages plus grid connection

    def Salvage_Value_Upper_Raw(model):   # Floor-at-zero linearization (2026-08-28): Salvage_Value <= raw formula, but only enforced when Salvage_Positive_Flag=1
        BIG_M_SALVAGE = 2.0e6  # tightened 2026-08-29: anchored to Generator_Max_Units's hard cap (9400kW x $160.58/kW = ~$1.51M for this project's designs) plus headroom -- still ~230x the only real negative-salvage magnitude observed (-$8,538). RES has no formal capacity ceiling in this model, so this isn't a rigorous proof-bound, just a much tighter one than the earlier 5.0e6/2.0e7 -- an oversized Big-M here visibly degrades barrier/crossover conditioning (RHS/matrix range balloons relative to the rest of the model's coefficients)
        return model.Salvage_Value <= Constraints_Greenfield_Milp._Salvage_Value_Raw(model) + BIG_M_SALVAGE * (1 - model.Salvage_Positive_Flag)

    def Salvage_Value_Upper_Flag(model):   # Floor-at-zero linearization: Salvage_Value <= 0, forced whenever Salvage_Positive_Flag=0 (i.e. whenever the raw formula is negative)
        BIG_M_SALVAGE = 2.0e6
        return model.Salvage_Value <= BIG_M_SALVAGE * model.Salvage_Positive_Flag

    def Energy_balance(model,s,yt,ut,t): # Energy balance
        Foo = []  # Index list for RES energy production in this period
        for r in model.renewable_sources:  # Loop over all renewable source types
            Foo.append((s,yt,r,t))      # Collect index for this scenario/year/source/time
        Total_Renewable_Energy = sum(model.RES_Energy_Production[j] for j in Foo)      # Sum of all RES output in this period
        foo=[]  # Index list for generator energy production in this period
        for g in model.generator_types:  # Loop over all generator types
            foo.append((s,yt,g,t))      # Collect index for this scenario/year/generator/time
        Total_Generator_Energy = sum(model.Generator_Energy_Total[i] for i in foo)  # Sum of all generator output in this period (MILP total = full + partial)
        if model.Grid_Connection == 1:  # Grid-connected: allow energy exchange with the grid
            En_From_Grid = model.Energy_From_Grid[s,yt,t]*model.Grid_Availability[s,yt,t]  # Grid imports masked by outage availability
            if model.Grid_Connection_Type == 0:  # Bidirectional connection: exports earn revenue
                En_To_Grid = model.Energy_To_Grid[s,yt,t]*model.Grid_Availability[s,yt,t]  # Grid exports masked by outage availability (only if export allowed)
            else: En_To_Grid = 0  # Export not allowed for this connection type
        else:  # Off-grid: no grid imports or exports
            En_From_Grid = 0  # Off-grid: no imports
            En_To_Grid = 0  # Off-grid: no exports
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            return model.Energy_Demand[s,yt,t] == (Total_Renewable_Energy   # Full system: demand = RES + generator + grid import/export + battery +/- lost load/curtailment
                                                   + Total_Generator_Energy  # Add generator output
                                                   + En_From_Grid  # Add grid imports
                                                   - En_To_Grid   # Subtract grid exports
                                                   + model.Battery_Outflow[s,yt,t]  # Add battery discharge
                                                   - model.Battery_Inflow[s,yt,t]   # Subtract battery charge
                                                   + model.Lost_Load[s,yt,t]    # Add unserved (lost load) energy
                                                   - model.Energy_Curtailment[s,yt,t] )       # Subtract curtailed (wasted) energy
        if model.Model_Components == 1:  # RES + battery only (no generator)
            return model.Energy_Demand[s,yt,t] == (Total_Renewable_Energy   # Full system: demand = RES + generator + grid import/export + battery +/- lost load/curtailment
                                                   + En_From_Grid  # Add grid imports
                                                   - En_To_Grid   # Subtract grid exports
                                                   + model.Battery_Outflow[s,yt,t]  # Add battery discharge
                                                   - model.Battery_Inflow[s,yt,t]   # Subtract battery charge
                                                   + model.Lost_Load[s,yt,t]    # Add unserved (lost load) energy
                                                   - model.Energy_Curtailment[s,yt,t] )       # Subtract curtailed (wasted) energy
        if model.Model_Components == 2:  # RES + generator only (no battery)
            return model.Energy_Demand[s,yt,t] == (Total_Renewable_Energy   # Full system: demand = RES + generator + grid import/export + battery +/- lost load/curtailment
                                                   + Total_Generator_Energy  # Add generator output
                                                   + En_From_Grid  # Add grid imports
                                                   - En_To_Grid   # Subtract grid exports
                                                   + model.Lost_Load[s,yt,t]    # Add unserved (lost load) energy
                                                   - model.Energy_Curtailment[s,yt,t] )       # Subtract curtailed (wasted) energy
        
    "Renewable Energy Sources constraints"
    def Renewable_Energy(model,s,yt,ut,r,t): # Energy output of the RES
        return model.RES_Energy_Production[s,yt,r,t] == model.RES_Unit_Energy_Production[s,r,t]*model.RES_Inverter_Efficiency[r]*model.RES_Units_milp[ut,r]  # Unit yield x inverter efficiency x number of installed units
    
    def Renewable_Energy_Penetration(model,ut):      # Enforce minimum renewable energy penetration target for step ut
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration  # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur  # Start year of step i+1
        yu_tuples_list = [0 for i in model.years]  # Will map each project year to its (year, step_index) tuple
        if model.Steps_Number == 1:  # Single investment step: only the initial build can have residual value
            for y in model.years:          # Loop over every project year
                yu_tuples_list[y-1] = (y, 1)  # Single-step case: every year maps to step 1
        else:      # Multiple investment steps: determine which step each year belongs to
            for y in model.years:          # Loop over every project year
                for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                    if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                        yu_tuples_list[y-1] = (y, model.steps[i+1])              # Record (year, step) pair
                    elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                        yu_tuples_list[y-1] = (y, len(model.steps))         # Record (year, final step) pair
        years_list = []  # Project years that belong to investment step ut
        for i in yu_tuples_list:  # Loop over all (year, step) pairs
            if i[1]==ut:  # Keep only the years belonging to the current step ut
                years_list += [i[0]]              # Record this year
        Foo=[]  # Index list accumulator
        for s in range(1, model.Scenarios + 1):  # Loop over all scenarios
            for y in years_list:  # Loop over years belonging to step ut
                for g in range(1, model.Generator_Types+1):  # Loop over all generator types
                    for t in range(1,model.Periods+1):  # Loop over all time periods
                        Foo.append((s,y,g,t))                          # Collect index for this scenario/year/generator/time
        foo=[]  # Index list for generator energy production in this period
        for s in range(1, model.Scenarios + 1):  # Loop over all scenarios
            for y in years_list:  # Loop over years belonging to step ut
                for r in range(1, model.RES_Sources+1):  # Loop over all RES sources
                    for t in range(1,model.Periods+1):  # Loop over all time periods
                        foo.append((s,y,r,t))          # Collect index for this scenario/year/source/time
        goo=[]  # Index list for grid import in this period
        for s in range(1, model.Scenarios + 1):  # Loop over all scenarios
            for y in years_list:  # Loop over years belonging to step ut
                    for t in range(1,model.Periods+1):  # Loop over all time periods
                        goo.append((s,y,t))  # Collect index for this scenario/year/time
                        
        E_gen = sum(model.Generator_Energy_Total[s,y,g,t]*model.Scenario_Weight[s]  # Probability-weighted generator energy over step ut (MILP total)
                    for s,y,g,t in Foo)  # Sum over the collected scenario/year/generator/time indices
        E_ren = sum(model.RES_Energy_Production[s,y,r,t]*model.Scenario_Weight[s]  # Probability-weighted RES energy over step ut
                    for s,y,r,t in foo)  # Sum over the collected scenario/year/source/time indices
        if model.Grid_Connection == 1:  # Grid-connected: include grid imports in the penetration calculation
            E_From_Grid = sum(model.Energy_From_Grid[s,y,t]*model.Grid_Availability[s,y,t]*model.Scenario_Weight[s]  # Probability-weighted grid imports over step ut
                    for s,y,t in goo)  # Sum over the collected scenario/year/time indices
        else: E_From_Grid = 0  # Off-grid: no imports to include in penetration calculation
 
        return  (1 - model.Renewable_Penetration)*E_ren >= model.Renewable_Penetration*(E_gen + E_From_Grid)     # RES share of total energy supplied must meet the penetration target
    
    def Renewables_Min_Step_Units(model,yt,ut,r):  # RES unit count must not decrease between investment steps (no decommissioning)
        if ut > 1:  # After the first step: value must not decrease from the previous step
            return model.RES_Units_milp[ut,r] >= model.RES_Units_milp[ut-1,r]  # RES units must not decrease from the previous step
        elif ut == 1:  # First step: no prior step to compare against
            return model.RES_Units_milp[ut,r] == model.RES_Units_milp[ut,r]  # First step: trivially satisfied (no previous step)
        
    "Maximum land use constraint for Renewables"
    def Renewables_Max_Land_Use(model,ut):  # Total land footprint of RES units must not exceed available area -- signature fixed 2026-09-04, see Constraints_Brownfield's own Renewables_Max_Land_Use comment for the full explanation (same bug, same fix, same discovery route).
        return  (sum((model.RES_Units_milp[ut,r]*model.RES_Nominal_Capacity[r]*(model.RES_Specific_Area[r])) for r in model.renewable_sources)) <= model.Renewables_Total_Area  # Total land footprint of installed RES units must not exceed available area
    
    "Battery Energy Storage constraints"
    def State_of_Charge(model,s,yt,ut,t): # State of Charge of the battery
        if t==1 and yt==1: # The state of charge (State_Of_Charge) for the period 0 is equal to the Battery size.
            return model.Battery_SOC[s,yt,t] == model.Battery_Units[ut]*model.Battery_Nominal_Capacity_milp*model.Battery_Initial_SOC - model.Battery_Outflow[s,yt,t]/model.Battery_Discharge_Battery_Efficiency + model.Battery_Inflow[s,yt,t]*model.Battery_Charge_Battery_Efficiency  # Initial SOC = starting fraction of installed capacity, adjusted for first-period flows
        if t==1 and yt!=1:  # First period of a year after the first project year
            return model.Battery_SOC[s,yt,t] == model.Battery_SOC[s,yt-1,model.Periods] - model.Battery_Outflow[s,yt,t]/model.Battery_Discharge_Battery_Efficiency + model.Battery_Inflow[s,yt,t]*model.Battery_Charge_Battery_Efficiency  # Carry over SOC from the last period of the previous year
        else:    # Any period after the first: carry over SOC from the previous period
            return model.Battery_SOC[s,yt,t] == model.Battery_SOC[s,yt,t-1] - model.Battery_Outflow[s,yt,t]/model.Battery_Discharge_Battery_Efficiency + model.Battery_Inflow[s,yt,t]*model.Battery_Charge_Battery_Efficiency      # Carry over SOC from the previous period within the same year
    
    def Maximum_Charge(model,s,yt,ut,t): # Maximun state of charge of the Battery
        return model.Battery_SOC[s,yt,t] <= model.Battery_Units[ut]*model.Battery_Nominal_Capacity_milp  # SOC cannot exceed installed battery capacity
    
    def Minimum_Charge(model,s,yt,ut,t): # Minimun state of charge
        return model.Battery_SOC[s,yt,t] >= model.Battery_Units[ut]*model.Battery_Nominal_Capacity_milp*(1-model.Battery_Depth_of_Discharge)  # SOC cannot fall below the minimum allowed by the depth of discharge
    
    def Max_Power_Battery_Charge(model,ut):   # Max charge power = capacity / min charge time (C-rate limit)
        return model.Battery_Maximum_Charge_Power[ut] == (model.Battery_Units[ut]*model.Battery_Nominal_Capacity_milp)/model.Maximum_Battery_Charge_Time  # Max charge power = installed capacity / minimum charge time
    
    def Max_Power_Battery_Discharge(model,ut):  # Max discharge power = capacity / min discharge time (C-rate limit)
        return model.Battery_Maximum_Discharge_Power[ut] == (model.Battery_Units[ut]*model.Battery_Nominal_Capacity_milp)/model.Maximum_Battery_Discharge_Time  # Max discharge power = installed capacity / minimum discharge time
    
    def Max_Bat_in(model,s,yt,ut,t): # Minimun flow of energy for the charge fase
        return model.Battery_Inflow[s,yt,t] <= model.Battery_Maximum_Charge_Power[ut]*model.Delta_Time  # Cap charging power at the maximum charge rate
    
    def Max_Bat_out(model,s,yt,ut,t): # Minimum flow of energy for the discharge fase
        return model.Battery_Outflow[s,yt,t] <= model.Energy_Demand[s,yt,t]  # Discharge cannot exceed current demand
        
    def Battery_Min_Capacity(model,ut):      # Battery must meet the minimum autonomy requirement
        return   model.Battery_Units[ut]*model.Battery_Nominal_Capacity_milp >= model.Battery_Min_Capacity[ut]  # Installed battery capacity must meet the minimum autonomy requirement
    
    def Battery_Min_Step_Capacity(model,yt,ut):      # Battery capacity must not shrink between investment steps
        if ut > 1:  # After the first step: value must not decrease from the previous step
            return model.Battery_Units[ut] >= model.Battery_Units[ut-1]  # Battery unit count must not decrease from the previous step
        elif ut == 1:  # First step: no prior step to compare against
            return model.Battery_Units[ut] == model.Battery_Units[ut]  # First step: trivially satisfied (no previous step)
    
    # POWER CAP -- registered only in the linearised forms, where it carries the
    # C-rate limit that the bilinear constraint below otherwise carries implicitly.
    # Var <= Var, so linear. Omitting these when moving off the bilinear form would
    # silently delete the C-rate limit and let the battery empty in a single hour.
    def Max_Bat_flow_out_milp(model,s,yt,ut,t):
        return model.Battery_Outflow[s,yt,t] <= model.Battery_Maximum_Discharge_Power[ut]*model.Delta_Time

    def Max_Bat_flow_in_milp(model,s,yt,ut,t):
        return model.Battery_Inflow[s,yt,t] <= model.Battery_Maximum_Charge_Power[ut]*model.Delta_Time

    if Single_Flow_Linear:  # MGPY_BESS_FORM=linear -- mode gate only, cap handled above
        def Battery_Single_Flow_Discharge(model,s,yt,ut,t):
            # M = Energy_Demand is EXACT, not a guess: Max_Bat_out already caps outflow
            # at demand, so z=1 imposes nothing new and z=0 forces outflow to zero.
            # Borrowed validity -- if Max_Bat_out is ever removed, revisit this.
            return model.Battery_Outflow[s,yt,t] <= model.Energy_Demand[s,yt,t]*model.Single_Flow_BESS[s,yt,t]

        def Battery_Single_Flow_Charge(model,s,yt,ut,t):
            return model.Battery_Inflow[s,yt,t] <= model.Battery_Max_Inflow*(1-model.Single_Flow_BESS[s,yt,t])
    else:  # historical bilinear form (default) -- one constraint doing cap AND gate
        def Battery_Single_Flow_Discharge(model,s,yt,ut,t):  # Binary flag prevents simultaneous charge and discharge
            return   model.Battery_Outflow[s,yt,t] <= model.Single_Flow_BESS[s,yt,t]*model.Battery_Maximum_Discharge_Power[ut]*model.Delta_Time  # Discharge allowed only when the discharge-mode binary is active

        def Battery_Single_Flow_Charge(model,s,yt,ut,t):  # Charge only allowed when not discharging (complementary binary)
            return   model.Battery_Inflow[s,yt,t] <= (1-model.Single_Flow_BESS[s,yt,t])*model.Battery_Maximum_Charge_Power[ut]*model.Delta_Time  # Charge allowed only when not in discharge mode

    "Diesel generator constraints"
    if Generator_Partial_Load:  # Brownfield MILP with partial-load modelling enabled
     def Minimum_Generator_Energy_Partial(model,s,yt,ut,g,t):  # Partial output >= min stable load (binary active flag x min fraction)
        return model.Generator_Energy_Partial[s,yt,g,t] >= (model.Generator_Nominal_Capacity_milp[g]*model.Generator_Min_output[g]*model.Delta_Time)*model.Generator_Partial[s,yt,g,t]  # Enforce minimum stable operating point when running in partial-load mode
    
     def Maximum_Generator_Energy_Partial(model,s,yt,ut,g,t):  # Partial output <= rated capacity of one partially-loaded unit
        return model.Generator_Energy_Partial[s,yt,g,t] <= model.Generator_Nominal_Capacity_milp[g]*model.Generator_Partial[s,yt,g,t]*model.Delta_Time  # Cap partial-load output at one unit's rated capacity
    
     def Maximum_Generator_Energy_Total_1(model,s,yt,ut,g,t):  # Total output <= all installed integer units at full rating
        return model.Generator_Energy_Total[s,yt,g,t] <= model.Generator_Nominal_Capacity_milp[g]*model.Generator_Units[ut,g]*model.Delta_Time  # Cap total output at installed integer capacity
    
     def Maximum_Generator_Energy_Total_2(model,s,yt,ut,g,t):  # Total output cannot exceed demand
        return model.Generator_Energy_Total[s,yt,g,t] <= model.Energy_Demand[s,yt,t]*model.Delta_Time  # Generator cannot supply more than the current demand
    
     def Generator_Energy_Total(model,s,yt,ut,g,t):  # Total = full-load committed blocks + partial-load energy
        return model.Generator_Energy_Total[s,yt,g,t] == (model.Generator_Full[s,yt,g,t]*model.Generator_Nominal_Capacity_milp[g]) + model.Generator_Energy_Partial[s,yt,g,t]  # Sum full-load block output and partial-load output
    
     def Generator_Units_Total(model,s,yt,ut,g,t):  # Committed units (full + at most one partial) <= installed units -- the rest may sit idle
        return model.Generator_Units[ut,g] >= model.Generator_Full[s,yt,g,t] + model.Generator_Partial[s,yt,g,t]  # FIX (2026-08-22): was `==`, which forced ALL installed units into full-or-partial commitment every hour regardless of demand -- with Generator_Nominal_Capacity_milp granularity of a few kW, "installed units" is in the tens/hundreds, so the old equality forced near-total genset output at every hour and made the model infeasible the instant demand ever fell below that floor. `>=` restores the standard unit-commitment reading: committed units cannot exceed what's installed, but idle units are allowed.
    else:  # No partial load - generators operate at full capacity or off
     def Maximum_Generator_Energy_Total_1(model,s,yt,ut,g,t):  # Total output <= installed integer capacity
        return model.Generator_Energy_Total[s,yt,g,t] <= model.Generator_Nominal_Capacity_milp[g]*model.Generator_Units[ut,g]*model.Delta_Time  # Cap total output at installed integer capacity
    
     def Maximum_Generator_Energy_Total_2(model,s,yt,ut,g,t):  # Total output cannot exceed demand
        return model.Generator_Energy_Total[s,yt,g,t] <= model.Energy_Demand[s,yt,t]*model.Delta_Time  # Generator cannot supply more than the current demand
    
    def Generator_Min_Step_Capacity(model,yt,ut,g):  # Unit count must not fall between steps (no decommissioning)
        if ut > 1:  # After the first step: value must not decrease from the previous step
            return model.Generator_Units[ut,g] >= model.Generator_Units[ut-1,g]  # Generator unit count must not decrease from the previous step
        elif ut ==1:  # First step: no prior step to compare against
            return model.Generator_Units[ut,g] == model.Generator_Units[ut,g]  # First step: trivially satisfied (no previous step)

    def Generator_Max_Units(model,yt,ut,g):  # ADDED (2026-08-23): opt-in ceiling on installed generator capacity, tightens the Generator_Units LP relaxation -- see Model_Creation.py's Generator_Max_Capacity for the full rationale. No-op (10**9 kW) unless MGPY_MAX_GENERATOR_KW is set.
        return model.Generator_Units[ut,g]*model.Generator_Nominal_Capacity_milp[g] <= model.Generator_Max_Capacity[g]

    "Lost load constraints"
    def Maximum_Lost_Load(model,s,yt): # Maximum admittable lost load
        # Algebraically identical to the original form, which was written as
        #   Lost_Load_Fraction >= sum(Lost_Load) / sum(Energy_Demand)
        # but multiplied through by sum(Energy_Demand) instead of dividing by it.
        # Total annual demand is a constant, so moving it to the right-hand side
        # turns it into an RHS value rather than a matrix coefficient.
        #
        # WHY THIS MATTERS (measured 2026-07-26, not inferred): the divided form put
        # a coefficient of 1/sum(Energy_Demand) on every Lost_Load variable. With
        # demand in Wh that is ~9.9e-09 -- the SMALLEST coefficient anywhere in the
        # model, and the reason the logged matrix range reached [1e-08, 3e+06], a
        # ratio of 3.2e14 against Gurobi's guidance of below 1e9. It came from just
        # 30 rows out of 3.15 million. In this form the Lost_Load coefficients are
        # 1.0, and the worst remaining coefficient is 4.19e-06 in
        # BatteryReplacementCostAct -- roughly a 420x narrowing of the range.
        # Denominator is a sum of demands and is always strictly positive, so the
        # inequality direction is preserved.
        return sum(model.Lost_Load[s,yt,t] for t in model.periods) <= model.Lost_Load_Fraction * sum(model.Energy_Demand[s,yt,t] for t in model.periods)  # Lost load must not exceed the allowed fraction of total demand
    
    "Emission constraints"
    def RES_emission(model): #LCA emissions of RES
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration   # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]    # Will map each project year to its (year, step_index) tuple
        for y in model.years:        # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])            # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))              # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]    # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[model.Step_Duration*i + model.Step_Duration]    # Upgrade decision occurs at the end of each step
            
        return model.RES_emission == sum(model.RES_unit_CO2_emission[r]*model.RES_Units_milp[1,r]*model.RES_Nominal_Capacity[r] for r in model.renewable_sources)+sum(sum(model.RES_unit_CO2_emission[r]*(model.RES_Units_milp[ut,r]-model.RES_Units_milp[ut-1,r])*model.RES_Nominal_Capacity[r] for (yt,ut) in tup_list) for r in model.renewable_sources)  # LCA emissions from step-1 RES installation (no pre-existing capacity to credit)

    def GEN_emission(model): #LCA emissions of generator
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration   # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]    # Will map each project year to its (year, step_index) tuple
        for y in model.years:        # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])            # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))              # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]    # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[model.Step_Duration*i + model.Step_Duration]  # Upgrade decision occurs at the end of each step
            
        return model.GEN_emission == sum((model.Generator_Units[1,g]*model.Generator_Nominal_Capacity_milp[g])*model.GEN_unit_CO2_emission[g] for g in model.generator_types)+sum(sum(((model.Generator_Units[ut,g]-model.Generator_Units[ut-1,g])*model.Generator_Nominal_Capacity_milp[g])*model.GEN_unit_CO2_emission[g] for (yt,ut) in tup_list) for g in model.generator_types)  # LCA emissions from step-1 generator installation (no pre-existing capacity to credit)
    
    def FUEL_emission(model,s,yt,ut,g,t): #Emissions from fuel consumption
        return model.FUEL_emission[s,yt,g,t] == model.Generator_Energy_Total[s,yt,g,t]/model.Fuel_LHV[g]/model.Generator_Efficiency[g]*model.FUEL_unit_CO2_emission[g]   # Emissions = energy / (fuel LHV x generator efficiency) x fuel emission factor
    
    def GRID_emission(model, s, y, t):  # Emissions from grid electricity imports, using the national grid CO2 factor
        if y >= model.Year_Grid_Connection:  # Only count grid emissions once the connection year has been reached
            return model.GRID_emission[s,y,t] == model.Energy_From_Grid[s,y,t] * model.National_Grid_Specific_CO2_emissions  # Import energy x national grid CO2 factor / 1000
        else:  # Before the connection year: no grid emissions
            return model.GRID_emission[s, y, t] == 0  # No grid emissions before the connection year
     
    def BESS_emission(model): #LCA emissions of battery
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration   # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]    # Will map each project year to its (year, step_index) tuple
        for y in model.years:        # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])            # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))              # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]    # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[model.Step_Duration*i + model.Step_Duration]  # Upgrade decision occurs at the end of each step
            
        return model.BESS_emission == (model.Battery_Units[1]*model.Battery_Nominal_Capacity_milp)*model.BESS_unit_CO2_emission+sum(((model.Battery_Units[ut]-model.Battery_Units[ut-1])*model.Battery_Nominal_Capacity_milp)*model.BESS_unit_CO2_emission for (yt,ut) in tup_list)  # LCA emissions from step-1 battery installation
        
    def Scenario_FUEL_emission(model,s):   # Total fuel-burn CO2 emissions for scenario s, summed over years/generators/periods
        return model.Scenario_FUEL_emission[s] == sum(sum(sum(model.Generator_Energy_Total[s,y,g,t]/model.Fuel_LHV[g]/model.Generator_Efficiency[g]*model.FUEL_unit_CO2_emission[g]  for t in model.periods) for y in model.years) for g in model.generator_types)   # Sum fuel emissions over all periods, years and generator types
    
    def Scenario_GRID_emission(model, s):  # Total grid-import CO2 emissions for scenario s, summed over years/periods
        Total_Grid_Emission = 0  # Accumulator for total grid emissions across all years
        for y in range(1, model.Years + 1):  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only count grid emissions once the connection year has been reached
                Total_Grid_Emission += sum(model.Energy_From_Grid[s, y, t] * model.National_Grid_Specific_CO2_emissions  for t in model.periods)  # Yearly grid emissions = imports x national grid CO2 factor / 1000
        return model.Scenario_GRID_emission[s] == Total_Grid_Emission  # Return total scenario grid emissions constraint
    
    "Grid constraints"  
    def Maximum_Power_From_Grid(model,s,y,t):  # Cap imports from grid; zero before connection year or during outages
        if y < model.Year_Grid_Connection or model.Grid_Availability[s, y, t] == 0:  # Grid not yet connected or currently unavailable
            return model.Energy_From_Grid[s,y,t] == 0  # No imports possible
        else:  # Grid connected and available: cap imports at the max grid power
            return model.Energy_From_Grid[s,y,t] <= model.Maximum_Grid_Power   # Cap imports at maximum grid power (kW to W)
    
    def Maximum_Power_To_Grid(model,s,y,t):  # Cap exports to grid; zero if not allowed or grid unavailable
        if y < model.Year_Grid_Connection or model.Grid_Connection_Type == 1 or model.Grid_Availability[s, y, t] == 0:  # Export not allowed (import-only) or grid unavailable/not yet connected
            return model.Energy_To_Grid[s, y, t] == 0  # No exports possible
        elif model.Grid_Connection_Type == 0:  # Bidirectional connection: exports are allowed
            return model.Energy_To_Grid[s, y, t] <= model.Maximum_Grid_Power  # Cap exports at maximum grid power (kW to W)
        
    def Single_Flow_Energy_To_Grid(model,s,yt,ut,t):  # Binary Single_Flow_Grid prevents exporting while importing
        return model.Energy_To_Grid[s,yt,t] <= model.Single_Flow_Grid[s,yt,t]*model.Large_Constant  # Export allowed only when the export-mode binary is active
        
    def Single_Flow_Energy_From_Grid(model,s,yt,ut,t):  # Complementary binary: import only when not exporting
        return model.Energy_From_Grid[s,yt,t] <= (1-model.Single_Flow_Grid[s,yt,t])*model.Large_Constant  # Import allowed only when the export-mode binary is inactive
   
     
#%% 

# --- Brownfield ---
    
class Constraints_Brownfield_Milp():  # MILP formulation — brownfield; integer unit sizing with pre-existing assets
    current_directory = os.path.dirname(os.path.abspath(__file__))  # Folder containing this module
    inputs_directory = os.path.join(current_directory, '..', 'Inputs')  # Sibling Inputs folder
    data_file_path = params_path.PARAMS_PATH  # Resolved once via MGPY_PARAMS; see params_path.py
    Data_import = open(data_file_path).readlines()  # Read the parameters file at class-definition time
    for i in range(len(Data_import)):  # Scan every line of the parameters file
     if "param: Generator_Partial_Load" in Data_import[i]:       # Locate partial-load flag
        Generator_Partial_Load = int((re.findall('\d+',Data_import[i])[0]))  # 1 = enable partial-load constraints
     if "param: Fuel_Specific_Cost_Calculation" in Data_import[i]:       # Locate fuel-cost flag
            Fuel_Specific_Cost_Calculation = int((re.findall('\d+',Data_import[i])[0]))  # 0 = constant price, 1 = time-varying
    "Objective function"
    def Net_Present_Cost_Obj(model):   # Pyomo objective wrapper for total Net Present Cost
        return (sum(model.Scenario_Net_Present_Cost[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected NPC = probability-weighted sum across scenarios
    
    def CO2_emission_Obj(model):  # Pyomo objective wrapper for total CO2 emissions
        return (sum(model.Scenario_CO2_emission[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected CO2 emissions = probability-weighted sum across scenarios
    
    def Total_Variable_Cost_Obj(model):  # Pyomo objective wrapper for total variable cost
        return (sum(model.Total_Scenario_Variable_Cost_NonAct[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected non-actualized variable cost = probability-weighted sum across scenarios
    
    "Net Present Cost"
    def Net_Present_Cost(model):     # Defines the scalar Net Present Cost as expected value over all scenarios
        return model.Net_Present_Cost == (sum(model.Scenario_Net_Present_Cost[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected NPC = probability-weighted sum across scenarios
    
    # def Scenario_Net_Present_Cost(model,s): 
    #     foo = []
    #     for g in range(1,model.Generator_Types+1):
    #             foo.append((s,g))            
    #     Fuel_Cost = sum(model.Total_Fuel_Cost_Act[s,g] for s,g in foo)    
    #     return model.Scenario_Net_Present_Cost[s] == (model.Investment_Cost + model.Operation_Maintenance_Cost_Act + model.Battery_Replacement_Cost_Act[s] 
    #             + model.Scenario_Lost_Load_Cost_Act[s] + Fuel_Cost - model.Salvage_Value)   
    
    def Scenario_Net_Present_Cost(model,s):   # NPC for scenario s: CAPEX + variable OPEX - salvage value
        return model.Scenario_Net_Present_Cost[s] == (model.Investment_Cost + model.Total_Scenario_Variable_Cost_Act[s] - model.Salvage_Value)     # NPC = CAPEX + actualized variable cost - salvage value
    
    def Total_Variable_Cost(model):  # Defines the scalar total variable cost (non-actualized)
        return model.Total_Variable_Cost == (sum(model.Total_Scenario_Variable_Cost_NonAct[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected non-actualized variable cost = probability-weighted sum across scenarios
    
    def CO2_emission(model):  # Defines the scalar CO2 emission as expected value over all scenarios
        return model.CO2_emission == (sum(model.Scenario_CO2_emission[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Expected CO2 emissions = probability-weighted sum across scenarios
    
    def Scenario_CO2_emission(model,s):  # Total lifecycle + operational CO2 for scenario s; components depend on active model parts
        if model.Grid_Connection == 1:  # Grid-connected: include grid import emissions
            if model.Model_Components == 0:  # Full system: RES + generator + battery
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.GEN_emission + model.BESS_emission + model.Scenario_FUEL_emission[s] + model.Scenario_GRID_emission[s])  # Full system with grid: RES + generator + battery + fuel + grid import emissions
            if model.Model_Components == 1:  # RES + battery only (no generator)
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.BESS_emission + model.Scenario_GRID_emission[s])  # Grid-connected, no generator: RES + battery + grid import emissions
            if model.Model_Components == 2:  # RES + generator only (no battery)
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.GEN_emission + model.Scenario_FUEL_emission[s] + model.Scenario_GRID_emission[s])  # Grid-connected, no battery: RES + generator + fuel + grid import emissions
        else:  # Off-grid: no grid-related emissions to add
            if model.Model_Components == 0:  # Full system: RES + generator + battery
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.GEN_emission + model.BESS_emission + model.Scenario_FUEL_emission[s])  # Off-grid full system: RES + generator + battery + fuel emissions
            if model.Model_Components == 1:  # RES + battery only (no generator)
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.BESS_emission)  # Off-grid, no generator: RES + battery emissions only
            if model.Model_Components == 2:  # RES + generator only (no battery)
                return model.Scenario_CO2_emission[s] ==  (model.RES_emission + model.GEN_emission + model.Scenario_FUEL_emission[s])  # Off-grid, no battery: RES + generator + fuel emissions
    
    "Investment cost"
    def Investment_Cost(model):    # Total discounted CAPEX for RES + generator + battery + grid connection
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration   # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]    # Will map each project year to its (year, step_index) tuple
        for y in model.years:        # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])            # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))              # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]    # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[model.Step_Duration*i + model.Step_Duration]   # Upgrade decision occurs at the end of each step
          
        Inv_Ren = sum(((model.RES_Units_milp[1,r]*model.RES_Nominal_Capacity[r]-model.RES_capacity[r])*model.RES_Specific_Investment_Cost[r])  # Step-1 RES CAPEX: integer units x size minus pre-existing capacity credit
                        + sum((((model.RES_Units_milp[ut,r] - model.RES_Units_milp[ut-1,r])*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r]))/((1+model.Discount_Rate)**(yt-1))  # NPV of incremental integer RES units
                        for (yt,ut) in tup_list) for r in model.renewable_sources)    # Sum incremental RES investment NPV over all upgrade points
        Inv_Gen = sum((((model.Generator_Units[1,g]*model.Generator_Nominal_Capacity_milp[g])-model.Generator_capacity[g])*model.Generator_Specific_Investment_Cost[g])  # Step-1 generator CAPEX minus pre-existing capacity credit
                        + sum((((model.Generator_Units[ut,g] - model.Generator_Units[ut-1,g])*model.Generator_Nominal_Capacity_milp[g]*model.Generator_Specific_Investment_Cost[g]))/((1+model.Discount_Rate)**(yt-1))  # NPV of incremental integer generator units
                        for (yt,ut) in tup_list) for g in model.generator_types)  # Sum incremental generator investment NPV over all upgrade points
        Inv_Bat = ((((model.Battery_Units[1]*model.Battery_Nominal_Capacity_milp)-model.Battery_capacity)*model.Battery_Specific_Investment_Cost)  # Step-1 battery CAPEX: integer units x unit capacity minus pre-existing
                        + sum((((model.Battery_Units[ut] - model.Battery_Units[ut-1])*model.Battery_Nominal_Capacity_milp*model.Battery_Specific_Investment_Cost))/((1+model.Discount_Rate)**(yt-1))  # NPV of incremental integer battery units
                        for (yt,ut) in tup_list))  # Sum incremental battery investment NPV over all upgrade points
        Inv_Grid = 0  # Accumulator for NPV of grid connection investment
        for y in model.years:  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur the grid connection cost once the connection year has been reached
                Inv_Grid += (model.Grid_Connection_Cost * model.Grid_Distance)/((1+model.Discount_Rate)**(y-1))   # Discount grid connection cost to year 0 and accumulate
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            investment_cost = Inv_Ren + Inv_Gen + Inv_Bat  # Full system: RES + generator + battery CAPEX
            if model.Grid_Connection: investment_cost += Inv_Grid  # Add grid connection cost if enabled
            return model.Investment_Cost == investment_cost   # Return the assembled CAPEX constraint
        if model.Model_Components == 1:  # RES + battery only (no generator)
            investment_cost = Inv_Ren + Inv_Bat  # RES + battery CAPEX (no generator)
            if model.Grid_Connection: investment_cost += Inv_Grid  # Add grid connection cost if enabled
            return model.Investment_Cost == investment_cost        # Return the assembled CAPEX constraint
        if model.Model_Components == 2:  # RES + generator only (no battery)
            investment_cost = Inv_Ren + Inv_Gen  # RES + generator CAPEX (no battery)
            if model.Grid_Connection: investment_cost += Inv_Grid  # Add grid connection cost if enabled
            return model.Investment_Cost == investment_cost    # Return the assembled CAPEX constraint
    
    def Investment_Cost_Limit(model):  # Enforce a user-defined cap on total CAPEX
        return model.Investment_Cost <= model.Investment_Cost_Limit  # Total CAPEX must not exceed the limit
       
    "Fixed O&M costs"
    def Operation_Maintenance_Cost_Act(model):  # Actualized (discounted) fixed O&M cost of RES + generator + battery + grid
        OyM_Ren = sum(sum((model.RES_Units_milp[ut,r]*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r]*model.RES_Specific_OM_Cost[r])/((  # NPV of RES fixed O&M cost (integer units), discounted per year it is incurred
                        1+model.Discount_Rate)**yt)for (yt,ut) in model.years_steps)for r in model.renewable_sources)  # Sum discounted RES O&M cost over all years/steps
        OyM_Gen = sum(sum(((model.Generator_Units[ut,g]*model.Generator_Nominal_Capacity_milp[g])*model.Generator_Specific_Investment_Cost[g]*model.Generator_Specific_OM_Cost[g])/((  # NPV of generator fixed O&M cost (integer units), discounted per year it is incurred
                        1+model.Discount_Rate)**yt)for (yt,ut) in model.years_steps)for g in model.generator_types)  # Sum discounted generator O&M cost over all years/steps
        OyM_Bat = sum((model.Battery_Units[ut]*model.Battery_Nominal_Capacity_milp*model.Battery_Specific_Investment_Cost*model.Battery_Specific_OM_Cost)/((  # NPV of battery fixed O&M cost (integer units), discounted per year it is incurred
                        1+model.Discount_Rate)**yt)for (yt,ut) in model.years_steps)  # Sum discounted battery O&M cost over all years/steps
        OyM_Grid = 0  # Accumulator for grid connection O&M cost
        for y in model.years:  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur grid O&M cost once the connection year has been reached
                OyM_Grid += (model.Grid_Connection_Cost * model.Grid_Distance * model.Grid_Maintenance_Cost)/((1+model.Discount_Rate)**(y))   # Discounted grid maintenance cost, only once grid is connected
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            cost = OyM_Ren + OyM_Gen + OyM_Bat  # Full system: RES + generator + battery O&M
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_Act == cost  # Return the assembled actualized O&M constraint
        if model.Model_Components == 1:  # RES + battery only (no generator)
            cost = OyM_Ren + OyM_Bat  # RES + battery O&M (no generator)
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_Act == cost  # Return the assembled actualized O&M constraint
        if model.Model_Components == 2:  # RES + generator only (no battery)
            cost = OyM_Ren + OyM_Gen  # RES + generator O&M (no battery)
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_Act == cost   # Return the assembled actualized O&M constraint

    
    def Operation_Maintenance_Cost_NonAct(model):  # Non-actualized (undiscounted) fixed O&M cost, mirrors the Act version
        OyM_Ren = sum(sum((model.RES_Units_milp[ut,r]*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r]*model.RES_Specific_OM_Cost[r])  # Undiscounted RES fixed O&M cost (integer units)
                        for (yt,ut) in model.years_steps)for r in model.renewable_sources)  # Sum undiscounted RES O&M cost over all years/steps
        OyM_Gen = sum(sum(((model.Generator_Units[ut,g]*model.Generator_Nominal_Capacity_milp[g])*model.Generator_Specific_Investment_Cost[g]*model.Generator_Specific_OM_Cost[g])  # Undiscounted generator fixed O&M cost (integer units)
                        for (yt,ut) in model.years_steps)for g in model.generator_types)  # Sum undiscounted generator O&M cost over all years/steps
        OyM_Bat = sum((model.Battery_Units[ut]*model.Battery_Nominal_Capacity_milp*model.Battery_Specific_Investment_Cost*model.Battery_Specific_OM_Cost)  # Undiscounted battery fixed O&M cost (integer units)
                        for (yt,ut) in model.years_steps)  # Sum undiscounted battery O&M cost over all years/steps
        OyM_Grid = 0  # Accumulator for grid connection O&M cost
        for y in model.years:  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur grid O&M cost once the connection year has been reached
                OyM_Grid += (model.Grid_Connection_Cost * model.Grid_Distance * model.Grid_Maintenance_Cost)  # Undiscounted grid maintenance cost, only once grid is connected
        
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            cost = OyM_Ren + OyM_Gen + OyM_Bat  # Full system: RES + generator + battery O&M
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_NonAct == cost   # Return the assembled non-actualized O&M constraint
        if model.Model_Components == 1:  # RES + battery only (no generator)
            cost = OyM_Ren + OyM_Bat  # RES + battery O&M (no generator)
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_NonAct == cost   # Return the assembled non-actualized O&M constraint
        if model.Model_Components == 2:  # RES + generator only (no battery)
            cost = OyM_Ren + OyM_Gen  # RES + generator O&M (no battery)
            if model.Grid_Connection: cost += OyM_Grid  # Add grid O&M cost if enabled
            return model.Operation_Maintenance_Cost_NonAct == cost    # Return the assembled non-actualized O&M constraint

    
    "Variable costs"
    def Total_Variable_Cost_Act(model):  # Scalar actualized total variable cost = expected value over all scenarios
        return model.Total_Variable_Cost_Act == (sum(model.Total_Scenario_Variable_Cost_Act[s]*model.Scenario_Weight[s] for s in model.scenarios))  # Probability-weighted sum across scenarios
    
    def Scenario_Variable_Cost_Act(model, s):  # Actualized variable cost for scenario s: O&M + battery replacement + lost load + fuel + net electricity cost
        foo = []  # Index list accumulator
        for g in range(1,model.Generator_Types+1):  # Loop over all generator types for this scenario
                foo.append((s,g))     # Collect (scenario, generator) index pair
        Fuel_Cost = sum(model.Total_Fuel_Cost_Act[s,g] for s,g in foo)     # Total actualized fuel cost across all generator types
        if model.Grid_Connection == 1:  # Grid-connected: include electricity purchase cost (and sale revenue if allowed)
            Electricity_Cost = model.Total_Electricity_Cost_Act[s]   # Actualized cost of electricity purchased from the grid
            if model.Grid_Connection_Type == 0:  # Bidirectional connection: exports earn revenue
                Electricity_Revenues = model.Total_Revenues_Act[s]  # Actualized revenue from electricity sold to the grid
            else: Electricity_Revenues = 0  # Import-only connection: no export revenue
        else:  # Off-grid: no electricity purchase or export cost
            Electricity_Cost = 0  # Off-grid: no electricity purchase cost
            Electricity_Revenues = 0  # Off-grid: no electricity sale revenue
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            return model.Total_Scenario_Variable_Cost_Act[s] == model.Operation_Maintenance_Cost_Act + model.Battery_Replacement_Cost_Act[s] + model.Scenario_Lost_Load_Cost_Act[s] + Fuel_Cost + Electricity_Cost - Electricity_Revenues  # Full system: O&M + battery replacement + lost load + fuel + net electricity
        if model.Model_Components == 1:  # RES + battery only (no generator)
            return model.Total_Scenario_Variable_Cost_Act[s] == model.Operation_Maintenance_Cost_Act + model.Battery_Replacement_Cost_Act[s] + model.Scenario_Lost_Load_Cost_Act[s] + Electricity_Cost - Electricity_Revenues  # No generator: O&M + battery replacement + lost load + net electricity
        if model.Model_Components == 2:  # RES + generator only (no battery)
            return model.Total_Scenario_Variable_Cost_Act[s] == model.Operation_Maintenance_Cost_Act + model.Scenario_Lost_Load_Cost_Act[s] + Fuel_Cost + Electricity_Cost - Electricity_Revenues  # No battery: O&M + lost load + fuel + net electricity
    
    def Scenario_Variable_Cost_NonAct(model, s):   # Non-actualized (undiscounted) mirror of Scenario_Variable_Cost_Act
        foo = []  # Index list accumulator
        for g in range(1,model.Generator_Types+1):  # Loop over all generator types for this scenario
            foo.append((s,g))     # Collect (scenario, generator) index pair
        Fuel_Cost = sum(model.Total_Fuel_Cost_NonAct[s,g] for s,g in foo)    # Total undiscounted fuel cost across all generator types
        if model.Grid_Connection == 1:  # Grid-connected: include electricity purchase cost (and sale revenue if allowed)
            Electricity_Cost = model.Total_Electricity_Cost_NonAct[s]   # Undiscounted cost of electricity purchased from the grid
            if model.Grid_Connection_Type == 0:  # Bidirectional connection: exports earn revenue
                Electricity_Revenues = model.Total_Revenues_NonAct[s]  # Undiscounted revenue from electricity sold to the grid
            else: Electricity_Revenues = 0  # Import-only connection: no export revenue
        else:  # Off-grid: no electricity purchase or export cost
            Electricity_Cost = 0  # Off-grid: no electricity purchase cost
            Electricity_Revenues = 0  # Off-grid: no electricity sale revenue
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            return model.Total_Scenario_Variable_Cost_NonAct[s] == model.Operation_Maintenance_Cost_NonAct + model.Battery_Replacement_Cost_NonAct[s] + model.Scenario_Lost_Load_Cost_NonAct[s] + Fuel_Cost + Electricity_Cost - Electricity_Revenues  # Full system: O&M + battery replacement + lost load + fuel + net electricity
        if model.Model_Components == 1:  # RES + battery only (no generator)
            return model.Total_Scenario_Variable_Cost_NonAct[s] == model.Operation_Maintenance_Cost_NonAct + model.Battery_Replacement_Cost_NonAct[s] + model.Scenario_Lost_Load_Cost_NonAct[s] + Electricity_Cost - Electricity_Revenues  # No generator: O&M + battery replacement + lost load + net electricity
        if model.Model_Components == 2:  # RES + generator only (no battery)
            return model.Total_Scenario_Variable_Cost_NonAct[s] == model.Operation_Maintenance_Cost_NonAct + model.Scenario_Lost_Load_Cost_NonAct[s] + Fuel_Cost + Electricity_Cost - Electricity_Revenues  # No battery: O&M + lost load + fuel + net electricity
    
    def Scenario_Lost_Load_Cost_Act(model,s):      # Actualized cost of unserved (lost load) energy for scenario s
        Cost_Lost_Load = 0           # Accumulator for discounted yearly lost-load cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(model.Lost_Load[s,y,t]*model.Lost_Load_Specific_Cost for t in model.periods)  # Yearly lost-load cost = unserved energy x penalty price
            Cost_Lost_Load += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
        return  model.Scenario_Lost_Load_Cost_Act[s] == Cost_Lost_Load  # Return actualized lost-load cost constraint
    
    def Scenario_Lost_Load_Cost_NonAct(model,s):  # Non-actualized (undiscounted) mirror of Scenario_Lost_Load_Cost_Act
        Cost_Lost_Load = 0           # Accumulator for discounted yearly lost-load cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(model.Lost_Load[s,y,t]*model.Lost_Load_Specific_Cost for t in model.periods)  # Yearly lost-load cost = unserved energy x penalty price
            Cost_Lost_Load += Num  # Accumulate undiscounted yearly lost-load cost
        return  model.Scenario_Lost_Load_Cost_NonAct[s] == Cost_Lost_Load  # Return non-actualized lost-load cost constraint

    if Generator_Partial_Load == 1 and Fuel_Specific_Cost_Calculation == 0:  # Partial-load MILP with constant fuel price
     "Partial Load Effect"
     def Total_Fuel_Cost_Act(model,s,g):  # Actualized fuel cost for generator type g across the project horizon
        Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(((model.Generator_Full[s,y,g,t]*model.Generator_Nominal_Capacity_milp[g]*model.Generator_Marginal_Cost_1[g])+(model.Generator_Marginal_Cost_milp_1[g]*model.Generator_Energy_Partial[s,y,g,t])+(model.Generator_Partial[s,y,g,t]*model.Generator_Start_Cost_1[g])) for t in model.periods)  # Full-load blocks + partial-load energy + start-up cost, all at constant prices
            Fuel_Cost_Tot += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
        return model.Total_Fuel_Cost_Act[s,g] == Fuel_Cost_Tot  # Return actualized fuel cost constraint
       
     def Total_Fuel_Cost_NonAct(model,s,g):  # Non-actualized (undiscounted) mirror of Total_Fuel_Cost_Act
        Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(((model.Generator_Full[s,y,g,t]*model.Generator_Nominal_Capacity_milp[g]*model.Generator_Marginal_Cost_1[g])+(model.Generator_Marginal_Cost_milp_1[g]*model.Generator_Energy_Partial[s,y,g,t])+(model.Generator_Partial[s,y,g,t]*model.Generator_Start_Cost_1[g])) for t in model.periods)  # Full-load blocks + partial-load energy + start-up cost, all at constant prices
            Fuel_Cost_Tot += Num  # Accumulate undiscounted yearly fuel cost
        return model.Total_Fuel_Cost_NonAct[s,g] == Fuel_Cost_Tot  # Return non-actualized fuel cost constraint
    elif Generator_Partial_Load == 1 and Fuel_Specific_Cost_Calculation == 1:  # Partial-load MILP with time-varying fuel price
     def Total_Fuel_Cost_Act(model,s,g):  # Actualized fuel cost for generator type g across the project horizon
        Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(((model.Generator_Full[s,y,g,t]*model.Generator_Nominal_Capacity_milp[g]*model.Generator_Marginal_Cost[g,y])+(model.Generator_Marginal_Cost_milp[g,y]*model.Generator_Energy_Partial[s,y,g,t])+(model.Generator_Partial[s,y,g,t]*model.Generator_Start_Cost[g,y])) for t in model.periods)  # Full-load blocks + partial-load energy + start-up cost, all at year-specific prices
            Fuel_Cost_Tot += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
        return model.Total_Fuel_Cost_Act[s,g] == Fuel_Cost_Tot  # Return actualized fuel cost constraint
       
     def Total_Fuel_Cost_NonAct(model,s,g):  # Non-actualized (undiscounted) mirror of Total_Fuel_Cost_Act
        Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(((model.Generator_Full[s,y,g,t]*model.Generator_Nominal_Capacity_milp[g]*model.Generator_Marginal_Cost[g,y])+(model.Generator_Marginal_Cost_milp[g,y]*model.Generator_Energy_Partial[s,y,g,t])+(model.Generator_Partial[s,y,g,t]*model.Generator_Start_Cost[g,y])) for t in model.periods)  # Full-load blocks + partial-load energy + start-up cost, all at year-specific prices
            Fuel_Cost_Tot += Num  # Accumulate undiscounted yearly fuel cost
        return model.Total_Fuel_Cost_NonAct[s,g] == Fuel_Cost_Tot  # Return non-actualized fuel cost constraint
    elif Generator_Partial_Load == 0 and Fuel_Specific_Cost_Calculation == 0:  # No partial load, constant fuel price (simplest MILP fuel cost)
     def Total_Fuel_Cost_Act(model,s,g):  # Actualized fuel cost for generator type g across the project horizon
        Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(model.Generator_Energy_Total[s,y,g,t]*model.Generator_Marginal_Cost_1[g] for t in model.periods)  # Total energy x constant marginal cost
            Fuel_Cost_Tot += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
        return model.Total_Fuel_Cost_Act[s,g] == Fuel_Cost_Tot  # Return actualized fuel cost constraint
       
     def Total_Fuel_Cost_NonAct(model,s,g):  # Non-actualized (undiscounted) mirror of Total_Fuel_Cost_Act
        Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(model.Generator_Energy_Total[s,y,g,t]*model.Generator_Marginal_Cost_1[g] for t in model.periods)  # Total energy x constant marginal cost
            Fuel_Cost_Tot += Num  # Accumulate undiscounted yearly fuel cost
        return model.Total_Fuel_Cost_NonAct[s,g] == Fuel_Cost_Tot  # Return non-actualized fuel cost constraint
    elif Generator_Partial_Load == 0 and Fuel_Specific_Cost_Calculation == 1:  # No partial load, time-varying fuel price
     def Total_Fuel_Cost_Act(model,s,g):  # Actualized fuel cost for generator type g across the project horizon
        Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(model.Generator_Energy_Total[s,y,g,t]*model.Generator_Marginal_Cost[g,y] for t in model.periods)  # Total energy x year-specific marginal cost
            Fuel_Cost_Tot += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
        return model.Total_Fuel_Cost_Act[s,g] == Fuel_Cost_Tot  # Return actualized fuel cost constraint
       
     def Total_Fuel_Cost_NonAct(model,s,g):  # Non-actualized (undiscounted) mirror of Total_Fuel_Cost_Act
        Fuel_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly fuel cost
        for y in range(1, model.Years +1):  # Loop over every project year
            Num = sum(model.Generator_Energy_Total[s,y,g,t]*model.Generator_Marginal_Cost[g,y] for t in model.periods)  # Total energy x year-specific marginal cost
            Fuel_Cost_Tot += Num  # Accumulate undiscounted yearly fuel cost
        return model.Total_Fuel_Cost_NonAct[s,g] == Fuel_Cost_Tot  # Return non-actualized fuel cost constraint
   
    def Total_Electricity_Cost_Act(model,s):   # Actualized cost of electricity purchased from the national grid
        Electricity_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly electricity cost
        for y in range(1, model.Years +1):  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur grid purchase cost once the connection year has been reached
                Num = sum(model.Energy_From_Grid[s,y,t]*model.Grid_Availability[s,y,t]*model.Grid_Purchased_El_Price  for t in model.periods)  # Yearly cost = grid imports (masked by availability) x purchase price / 1000
                Electricity_Cost_Tot += Num/((1+model.Discount_Rate)**y)  # Discount to year 0 and accumulate
        return model.Total_Electricity_Cost_Act[s] == Electricity_Cost_Tot  # Return actualized electricity cost constraint
       
    def Total_Electricity_Cost_NonAct(model,s):   # Non-actualized (undiscounted) mirror of Total_Electricity_Cost_Act
        Electricity_Cost_Tot = 0  # Accumulator for discounted (or undiscounted) yearly electricity cost
        for y in range(1, model.Years +1):  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only incur grid purchase cost once the connection year has been reached
                Electricity_Cost_Tot += sum(model.Energy_From_Grid[s,y,t]*model.Grid_Availability[s,y,t]*model.Grid_Purchased_El_Price  for t in model.periods)  # Accumulate undiscounted yearly electricity cost
        return model.Total_Electricity_Cost_NonAct[s] == Electricity_Cost_Tot  # Return non-actualized electricity cost constraint
    
    def Total_Revenues_NonAct(model, s):   # Undiscounted revenue from electricity exported to the grid
        Revenues_Yearly = [0 for _ in model.years]  # Per-year export revenue, zero before grid connection
        for y in range(1, model.Years + 1):  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only count grid export revenue once the connection year has been reached
                Revenues_Yearly[y - 1] = sum(model.Energy_To_Grid[s, y, t] * model.Grid_Availability[s, y, t] * model.Grid_Sold_El_Price  for t in model.periods)  # Yearly revenue = grid exports (masked by availability) x sale price / 1000
        return model.Total_Revenues_NonAct[s] == sum(Revenues_Yearly[y - 1] for y in model.years)  # Sum undiscounted yearly revenues

    def Total_Revenues_Act(model, s):   # Actualized (discounted) revenue from electricity exported to the grid
        Revenues_Yearly = [0 for _ in model.years]  # Per-year export revenue, zero before grid connection
        for y in range(1, model.Years + 1):  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only count grid export revenue once the connection year has been reached
                Revenues_Yearly[y - 1] = sum(model.Energy_To_Grid[s, y, t] * model.Grid_Availability[s, y, t] * model.Grid_Sold_El_Price  for t in model.periods)  # Yearly revenue = grid exports (masked by availability) x sale price / 1000
        return model.Total_Revenues_Act[s] == sum(Revenues_Yearly[y - 1] / ((1 + model.Discount_Rate) ** y) for y in model.years)  # Discount each year's export revenue to year 0 and sum
    
    def Battery_Replacement_Cost_Act(model,s):  # Actualized cost of replacing battery cells as they cycle (charge/discharge wear)
        Battery_cost_in = [0 for y in model.years]  # Per-year cost attributable to charging throughput
        Battery_cost_out = [0 for y in model.years]  # Per-year cost attributable to discharging throughput
        Battery_Yearly_cost = [0 for y in model.years]      # Per-year total replacement cost (charge + discharge)
        for y in range(1,model.Years+1):      # Loop over every project year
            Battery_cost_in[y-1] = sum(model.Battery_Inflow[s,y,t]*model.Unitary_Battery_Replacement_Cost for t in model.periods)  # Charging energy x unitary replacement cost, summed over the year
            Battery_cost_out[y-1] = sum(model.Battery_Outflow[s,y,t]*model.Unitary_Battery_Replacement_Cost for t in model.periods)  # Discharging energy x unitary replacement cost, summed over the year
            Battery_Yearly_cost[y-1] = Battery_cost_in[y-1] + Battery_cost_out[y-1]  # Total yearly replacement cost = charge + discharge wear
        return model.Battery_Replacement_Cost_Act[s] == sum(Battery_Yearly_cost[y-1]/((1+model.Discount_Rate)**y) for y in model.years)   # Discount each year's replacement cost to year 0 and sum
        
    def Battery_Replacement_Cost_NonAct(model,s):  # Non-actualized (undiscounted) mirror of Battery_Replacement_Cost_Act
        Battery_cost_in = [0 for y in model.years]  # Per-year cost attributable to charging throughput
        Battery_cost_out = [0 for y in model.years]  # Per-year cost attributable to discharging throughput
        Battery_Yearly_cost = [0 for y in model.years]      # Per-year total replacement cost (charge + discharge)
        for y in range(1,model.Years+1):      # Loop over every project year
            Battery_cost_in[y-1] = sum(model.Battery_Inflow[s,y,t]*model.Unitary_Battery_Replacement_Cost for t in model.periods)  # Charging energy x unitary replacement cost, summed over the year
            Battery_cost_out[y-1] = sum(model.Battery_Outflow[s,y,t]*model.Unitary_Battery_Replacement_Cost for t in model.periods)  # Discharging energy x unitary replacement cost, summed over the year
            Battery_Yearly_cost[y-1] = Battery_cost_in[y-1] + Battery_cost_out[y-1]  # Total yearly replacement cost = charge + discharge wear
        return model.Battery_Replacement_Cost_NonAct[s] == sum(Battery_Yearly_cost[y-1] for y in model.years)   # Sum undiscounted yearly replacement cost
    
    "Salvage Value"
    def _Salvage_Value_Raw(model):     # Residual (undepreciated) value of all assets at project end, discounted to year 0 -- NOT floored at zero, see Salvage_Value_Upper_Raw/Flag below
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration  # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur  # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]  # Will map each project year to its (year, step_index) tuple
        for y in model.years:      # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])          # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))            # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]  # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[s_dur*i + s_dur]  # Upgrade decision occurs at the end of each step

        if model.Steps_Number == 1:      # Single investment step: only the initial build can have residual value
            SV_Ren_1 = sum(((model.RES_Units_milp[1,r]*model.RES_Nominal_Capacity[r])-model.RES_capacity[r])*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]-model.Years)/model.RES_Lifetime[r] /   # Residual value of step-1 RES units, net of credit for pre-existing capacity
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)+sum(model.RES_capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]-model.RES_years[r]-model.Years)/model.RES_Lifetime[r] /   # Plus residual value of the pre-existing RES capacity itself
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)          # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_2 = 0   # No second-step RES capacity exists yet
            SV_Ren_3 = 0      # No later-step RES capacity exists yet
            SV_Gen_1 = sum(((model.Generator_Units[1,g]*model.Generator_Nominal_Capacity_milp[g])-model.Generator_capacity[g])*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]-model.Years)/model.Generator_Lifetime[g] /   # Residual value of step-1 generator capacity, net of credit for pre-existing capacity
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)+sum(model.Generator_capacity[g]*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]-model.GEN_years[g]-model.Years)/model.Generator_Lifetime[g] /   # Plus residual value of the pre-existing generator capacity itself
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)          # Discount generator residual value to year 0, summed over generator types
            SV_Gen_2 = 0  # No second-step generator capacity exists yet
            SV_Gen_3 = 0  # No later-step generator capacity exists yet
            SV_Grid = model.Grid_Distance*model.Grid_Connection_Cost*model.Grid_Connection / ((1 + model.Discount_Rate)**(model.Years - model.Year_Grid_Connection))   # Discounted residual value of the grid connection investment
    
        if model.Steps_Number == 2:      # Two investment steps: initial build + one upgrade can have residual value
            yt_last_up = upgrade_years_list[1]         # Year of the (only) capacity upgrade
            SV_Ren_1 = sum(((model.RES_Units_milp[1,r]*model.RES_Nominal_Capacity[r])-model.RES_capacity[r])*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]-model.Years)/model.RES_Lifetime[r] /   # Residual value of step-1 RES units, net of credit for pre-existing capacity
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)+sum(model.RES_capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]-model.RES_years[r]-model.Years)/model.RES_Lifetime[r] /   # Plus residual value of the pre-existing RES capacity itself
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)  # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_2 = sum((model.RES_Units_milp[2,r]-model.RES_Units_milp[1,r])*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]+(yt_last_up-1)-model.Years)/model.RES_Lifetime[r] /   # Residual value of the step-2 RES capacity increment
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)          # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_3 = 0      # No later-step RES capacity exists yet
            SV_Gen_1 = sum(((model.Generator_Units[1,g]*model.Generator_Nominal_Capacity_milp[g])-model.Generator_capacity[g])*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]-model.Years)/model.Generator_Lifetime[g] /   # Residual value of step-1 generator capacity, net of credit for pre-existing capacity
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)+sum(model.Generator_capacity[g]*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]-model.GEN_years[g]-model.Years)/model.Generator_Lifetime[g] /   # Plus residual value of the pre-existing generator capacity itself
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)          # Discount generator residual value to year 0, summed over generator types
            SV_Gen_2 = sum((model.Generator_Units[2,g]-model.Generator_Units[1,g])*model.Generator_Nominal_Capacity_milp[g]*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]+(yt_last_up-1)-model.Years)/model.Generator_Lifetime[g] /   # Residual value of the step-2 generator capacity increment
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)  # Discount generator residual value to year 0, summed over generator types
            SV_Gen_3 = 0  # No later-step generator capacity exists yet
            SV_Grid = model.Grid_Distance*model.Grid_Connection_Cost*model.Grid_Connection / ((1 + model.Discount_Rate)**(model.Years - model.Year_Grid_Connection))   # Discounted residual value of the grid connection investment
            
        if model.Steps_Number > 2:  # Three or more steps: initial build + last upgrade + all intermediate upgrades
            tup_list_2 = [[] for i in range(len(model.steps)-2)]  # (year, step) pairs for upgrades before the last one
            for i in range(len(model.steps) - 2):  # Build the intermediate-upgrade list (excludes the final upgrade)
                tup_list_2[i] = yu_tuples_list[s_dur*i + s_dur]  # Intermediate upgrade occurs at the end of each step
            yt_last_up = upgrade_years_list[-1]  # Year of the final capacity upgrade
            ut_last_up = tup_list[-1][1]      # Step index of the final upgrade
            ut_seclast_up = tup_list[-2][1]  # Step index of the second-to-last upgrade
    
            SV_Ren_1 = sum(((model.RES_Units_milp[1,r]*model.RES_Nominal_Capacity[r])-model.RES_capacity[r])*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]-model.Years)/model.RES_Lifetime[r] /   # Residual value of step-1 RES units, net of credit for pre-existing capacity
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)+sum(model.RES_capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]-model.RES_years[r]-model.Years)/model.RES_Lifetime[r] /   # Plus residual value of the pre-existing RES capacity itself
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)      # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_2 = sum((model.RES_Units_milp[ut_last_up,r] - model.RES_Units_milp[ut_seclast_up,r])*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]+(yt_last_up-1)-model.Years)/model.RES_Lifetime[r] /   # Residual value of the final RES capacity increment
                            ((1 + model.Discount_Rate)**(model.Years)) for r in model.renewable_sources)  # Discount RES residual value to year 0, summed over renewable sources
            SV_Ren_3 = sum(sum((model.RES_Units_milp[ut,r] - model.RES_Units_milp[ut-1,r])*model.RES_Nominal_Capacity[r]*model.RES_Specific_Investment_Cost[r] * (model.RES_Lifetime[r]+(yt-1)-model.Years)/model.RES_Lifetime[r] /   # Residual value of all intermediate RES capacity increments
                            ((1+model.Discount_Rate)**model.Years) for (yt,ut) in tup_list_2) for r in model.renewable_sources)          # Sum intermediate-upgrade RES residual value, discounted to year 0
            SV_Gen_1 = sum(((model.Generator_Units[1,g]*model.Generator_Nominal_Capacity_milp[g])-model.Generator_capacity[g])*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]-model.Years)/model.Generator_Lifetime[g] /   # Residual value of step-1 generator capacity, net of credit for pre-existing capacity
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)+sum(model.Generator_capacity[g]*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]-model.GEN_years[g]-model.Years)/model.Generator_Lifetime[g] /   # Plus residual value of the pre-existing generator capacity itself
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)  # Discount generator residual value to year 0, summed over generator types
            SV_Gen_2 = sum((model.Generator_Units[ut_last_up,g] - model.Generator_Units[ut_seclast_up,g])*model.Generator_Nominal_Capacity_milp[g]*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]+(yt_last_up-1)-model.Years)/model.Generator_Lifetime[g] /   # Residual value of the final generator capacity increment
                            ((1 + model.Discount_Rate)**(model.Years)) for g in model.generator_types)  # Discount generator residual value to year 0, summed over generator types
            SV_Gen_3 = sum(sum((model.Generator_Units[ut,g] - model.Generator_Units[ut-1,g])*model.Generator_Nominal_Capacity_milp[g]*model.Generator_Specific_Investment_Cost[g] * (model.Generator_Lifetime[g]+(yt-1)-model.Years)/model.Generator_Lifetime[g] /   # Residual value of all intermediate generator capacity increments
                            ((1+model.Discount_Rate)**model.Years) for (yt,ut) in tup_list_2) for g in model.generator_types)  # Sum intermediate-upgrade generator residual value, discounted to year 0
            SV_Grid = model.Grid_Distance*model.Grid_Connection_Cost*model.Grid_Connection / ((1 + model.Discount_Rate)**(model.Years - model.Year_Grid_Connection))   # Discounted residual value of the grid connection investment
       
        if model.Model_Components == 0 or model.Model_Components == 2:  # System includes a generator: credit RES + generator + grid salvage value
           return SV_Ren_1 + SV_Gen_1 + SV_Ren_2 + SV_Gen_2 + SV_Ren_3 + SV_Gen_3 + SV_Grid  # Sum residual value of all RES/generator vintages plus grid connection
        if model.Model_Components == 1:  # RES + battery only (no generator)
           return SV_Ren_1 + SV_Ren_2 + SV_Ren_3 + SV_Grid  # No generator: sum residual value of RES vintages plus grid connection

    def Salvage_Value_Upper_Raw(model):   # Floor-at-zero linearization (2026-08-28): Salvage_Value <= raw formula, but only enforced when Salvage_Positive_Flag=1. Root-caused via IIS on a real capexp5_6sc design (single constraint + single bound, raw -$8,538.38): a generator vintage installed early (front-loaded) with a lifetime much shorter than the project horizon drives the raw formula negative, which the NonNegativeReals domain then rejects outright instead of flooring at zero.
        BIG_M_SALVAGE = 2.0e6  # tightened 2026-08-29: anchored to Generator_Max_Units's hard cap (9400kW x $160.58/kW = ~$1.51M for this project's designs) plus headroom -- still ~230x the only real negative-salvage magnitude observed (-$8,538). RES has no formal capacity ceiling in this model, so this isn't a rigorous proof-bound, just a much tighter one than the earlier 5.0e6/2.0e7 -- an oversized Big-M here visibly degrades barrier/crossover conditioning (RHS/matrix range balloons relative to the rest of the model's coefficients)
        return model.Salvage_Value <= Constraints_Brownfield_Milp._Salvage_Value_Raw(model) + BIG_M_SALVAGE * (1 - model.Salvage_Positive_Flag)

    def Salvage_Value_Upper_Flag(model):   # Floor-at-zero linearization: Salvage_Value <= 0, forced whenever Salvage_Positive_Flag=0 (i.e. whenever the raw formula is negative)
        BIG_M_SALVAGE = 2.0e6
        return model.Salvage_Value <= BIG_M_SALVAGE * model.Salvage_Positive_Flag

    #%% Electricity balance constraints
    def BESS_Capacity(model,ut): #Minimum battery capacity 
        return model.Battery_Units[1]*model.Battery_Nominal_Capacity_milp >= model.Battery_capacity  # Installed capacity must cover the pre-existing battery
    
    def GEN_Capacity(model,ut,g): #Minimum generator capacity
        return model.Generator_Units[1,g]*model.Generator_Nominal_Capacity_milp[g] >= model.Generator_capacity[g]  # Installed capacity must cover the pre-existing generator
    
    def RES_Capacity(model,s,yt,ut,r,t): #Minimum RES energy production
        return model.RES_Energy_Production[s,yt,r,t] >= model.RES_Unit_Energy_Production[s,r,t]*model.RES_Inverter_Efficiency[r]* (model.RES_capacity[r] / model.RES_Nominal_Capacity[r])  # Minimum RES energy production must at least match pre-existing unit output
    
    def Energy_balance(model,s,yt,ut,t): # Energy balance
        Foo = []  # Index list for RES energy production in this period
        for r in model.renewable_sources:  # Loop over all renewable source types
            Foo.append((s,yt,r,t))      # Collect index for this scenario/year/source/time
        Total_Renewable_Energy = sum(model.RES_Energy_Production[j] for j in Foo)      # Sum of all RES output in this period
        foo=[]  # Index list for generator energy production in this period
        for g in model.generator_types:  # Loop over all generator types
            foo.append((s,yt,g,t))      # Collect index for this scenario/year/generator/time
        Total_Generator_Energy = sum(model.Generator_Energy_Total[i] for i in foo)    # Sum of all generator output in this period (MILP total = full + partial)
        if model.Grid_Connection == 1:  # Grid-connected: allow energy exchange with the grid
            En_From_Grid = model.Energy_From_Grid[s,yt,t]*model.Grid_Availability[s,yt,t]  # Grid imports masked by outage availability
            if model.Grid_Connection_Type == 0:  # Bidirectional connection: exports earn revenue
                En_To_Grid = model.Energy_To_Grid[s,yt,t]*model.Grid_Availability[s,yt,t]  # Grid exports masked by outage availability (only if export allowed)
            else: En_To_Grid = 0  # Export not allowed for this connection type
        else:  # Off-grid: no grid imports or exports
            En_From_Grid = 0  # Off-grid: no imports
            En_To_Grid = 0  # Off-grid: no exports
        
        if model.Model_Components == 0:  # Full system: RES + generator + battery
            return model.Energy_Demand[s,yt,t] == (Total_Renewable_Energy   # Full system: demand = RES + generator + grid import/export + battery +/- lost load/curtailment
                                                   + Total_Generator_Energy  # Add generator output
                                                   + En_From_Grid  # Add grid imports
                                                   - En_To_Grid   # Subtract grid exports
                                                   - model.Battery_Inflow[s,yt,t]   # Subtract battery charge
                                                   + model.Battery_Outflow[s,yt,t]   # Add battery discharge
                                                   + model.Lost_Load[s,yt,t]    # Add unserved (lost load) energy
                                                   - model.Energy_Curtailment[s,yt,t] )       # Subtract curtailed (wasted) energy
        if model.Model_Components == 1:  # RES + battery only (no generator)
            return model.Energy_Demand[s,yt,t] == (Total_Renewable_Energy   # Full system: demand = RES + generator + grid import/export + battery +/- lost load/curtailment
                                                   + En_From_Grid  # Add grid imports
                                                   - En_To_Grid   # Subtract grid exports
                                                   - model.Battery_Inflow[s,yt,t]   # Subtract battery charge
                                                   + model.Battery_Outflow[s,yt,t]   # Add battery discharge
                                                   + model.Lost_Load[s,yt,t]    # Add unserved (lost load) energy
                                                   - model.Energy_Curtailment[s,yt,t] )       # Subtract curtailed (wasted) energy
        if model.Model_Components == 2:  # RES + generator only (no battery)
            return model.Energy_Demand[s,yt,t] == (Total_Renewable_Energy   # Full system: demand = RES + generator + grid import/export + battery +/- lost load/curtailment
                                                   + Total_Generator_Energy  # Add generator output
                                                   + En_From_Grid  # Add grid imports
                                                   - En_To_Grid   # Subtract grid exports
                                                   + model.Lost_Load[s,yt,t]    # Add unserved (lost load) energy
                                                   - model.Energy_Curtailment[s,yt,t] )       # Subtract curtailed (wasted) energy
    
    "Renewable Energy Sources constraints"
    def Renewable_Energy(model,s,yt,ut,r,t): # Energy output of the solar panels
        return model.RES_Energy_Production[s,yt,r,t] == model.RES_Unit_Energy_Production[s,r,t]*model.RES_Inverter_Efficiency[r]*model.RES_Units_milp[ut,r]  # Unit yield x inverter efficiency x number of installed units
    
    def Renewable_Energy_Penetration(model,ut):      # Enforce minimum renewable energy penetration target for step ut
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration  # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur  # Start year of step i+1
        yu_tuples_list = [0 for i in model.years]  # Will map each project year to its (year, step_index) tuple
        if model.Steps_Number == 1:  # Single investment step: only the initial build can have residual value
            for y in model.years:          # Loop over every project year
                yu_tuples_list[y-1] = (y, 1)  # Single-step case: every year maps to step 1
        else:      # Multiple investment steps: determine which step each year belongs to
            for y in model.years:          # Loop over every project year
                for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                    if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                        yu_tuples_list[y-1] = (y, model.steps[i+1])              # Record (year, step) pair
                    elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                        yu_tuples_list[y-1] = (y, len(model.steps))         # Record (year, final step) pair
        years_list = []  # Project years that belong to investment step ut
        for i in yu_tuples_list:  # Loop over all (year, step) pairs
            if i[1]==ut:  # Keep only the years belonging to the current step ut
                years_list += [i[0]]              # Record this year
        Foo=[]  # Index list accumulator
        for s in range(1, model.Scenarios + 1):  # Loop over all scenarios
            for y in years_list:  # Loop over years belonging to step ut
                for g in range(1, model.Generator_Types+1):  # Loop over all generator types
                    for t in range(1,model.Periods+1):  # Loop over all time periods
                        Foo.append((s,y,g,t))                          # Collect index for this scenario/year/generator/time
        foo=[]  # Index list for generator energy production in this period
        for s in range(1, model.Scenarios + 1):  # Loop over all scenarios
            for y in years_list:  # Loop over years belonging to step ut
                for r in range(1, model.RES_Sources+1):  # Loop over all RES sources
                    for t in range(1,model.Periods+1):  # Loop over all time periods
                        foo.append((s,y,r,t))          # Collect index for this scenario/year/source/time
        goo=[]  # Index list for grid import in this period
        for s in range(1, model.Scenarios + 1):  # Loop over all scenarios
            for y in years_list:  # Loop over years belonging to step ut
                    for t in range(1,model.Periods+1):  # Loop over all time periods
                        goo.append((s,y,t))  # Collect index for this scenario/year/time
                        
        E_gen = sum(model.Generator_Energy_Total[s,y,g,t]*model.Scenario_Weight[s]  # Probability-weighted generator energy over step ut (MILP total)
                    for s,y,g,t in Foo)  # Sum over the collected scenario/year/generator/time indices
        E_ren = sum(model.RES_Energy_Production[s,y,r,t]*model.Scenario_Weight[s]  # Probability-weighted RES energy over step ut
                    for s,y,r,t in foo)  # Sum over the collected scenario/year/source/time indices
        if model.Grid_Connection == 1:  # Grid-connected: include grid imports in the penetration calculation
            E_From_Grid = sum(model.Energy_From_Grid[s,y,t]*model.Grid_Availability[s,y,t]*model.Scenario_Weight[s]  # Probability-weighted grid imports over step ut
                    for s,y,t in goo)  # Sum over the collected scenario/year/time indices
        else: E_From_Grid = 0  # Off-grid: no imports to include in penetration calculation
 
        return  (1 - model.Renewable_Penetration)*E_ren >= model.Renewable_Penetration*(E_gen + E_From_Grid)     # RES share of total energy supplied must meet the penetration target
    
    def Renewables_Min_Step_Units(model,yt,ut,r):  # RES unit count must not decrease between investment steps (no decommissioning)
        if ut > 1:  # After the first step: value must not decrease from the previous step
            return model.RES_Units_milp[ut,r] >= model.RES_Units_milp[ut-1,r]  # RES units must not decrease from the previous step
        elif ut == 1:  # First step: no prior step to compare against
            return model.RES_Units_milp[ut,r] == model.RES_Units_milp[ut,r]  # First step: trivially satisfied (no previous step)
            
    "Maximum land use constraint for Renewables"
    def Renewables_Max_Land_Use(model,ut):  # Total land footprint of RES (existing + new units) must not exceed available area -- signature fixed 2026-09-04 (was (model,s,yt,ut,r,t), which never matched the Constraint(model.steps, ...) declaration in Model_Resolution.py nor this body's actual dependence on `ut` alone; crashed with "missing 4 required positional arguments" the first time Land_Use=1 was ever exercised, via the linopy Stage 2 cross-validation. Matches Constraints_Greenfield's already-correct (model,ut) signature for the same rule.
        return  (sum((model.RES_existing_area[r] + (model.RES_Units_milp[ut,r]*model.RES_Nominal_Capacity[r]*(model.RES_Specific_Area[r]))) for r in model.renewable_sources)) <= model.Renewables_Total_Area  # Land use = pre-existing footprint + new units' footprint, capped at available area
    
    
    "Battery Energy Storage constraints"
    def State_of_Charge(model,s,yt,ut,t): # State of Charge of the battery
        if t==1 and yt==1: # The state of charge (State_Of_Charge) for the period 0 is equal to the Battery size.
            return model.Battery_SOC[s,yt,t] == model.Battery_Units[ut]*model.Battery_Nominal_Capacity_milp*model.Battery_Initial_SOC - model.Battery_Outflow[s,yt,t]/model.Battery_Discharge_Battery_Efficiency + model.Battery_Inflow[s,yt,t]*model.Battery_Charge_Battery_Efficiency  # Initial SOC = starting fraction of installed capacity, adjusted for first-period flows
        if t==1 and yt!=1:  # First period of a year after the first project year
            return model.Battery_SOC[s,yt,t] == model.Battery_SOC[s,yt-1,model.Periods] - model.Battery_Outflow[s,yt,t]/model.Battery_Discharge_Battery_Efficiency + model.Battery_Inflow[s,yt,t]*model.Battery_Charge_Battery_Efficiency  # Carry over SOC from the last period of the previous year
        else:    # Any period after the first: carry over SOC from the previous period
            return model.Battery_SOC[s,yt,t] == model.Battery_SOC[s,yt,t-1] - model.Battery_Outflow[s,yt,t]/model.Battery_Discharge_Battery_Efficiency + model.Battery_Inflow[s,yt,t]*model.Battery_Charge_Battery_Efficiency      # Carry over SOC from the previous period within the same year
    
    def Maximum_Charge(model,s,yt,ut,t): # Maximun state of charge of the Battery
        return model.Battery_SOC[s,yt,t] <= model.Battery_Units[ut]*model.Battery_Nominal_Capacity_milp  # SOC cannot exceed installed battery capacity
    
    def Minimum_Charge(model,s,yt,ut,t): # Minimun state of charge
        return model.Battery_SOC[s,yt,t] >= model.Battery_Units[ut]*model.Battery_Nominal_Capacity_milp*(1-model.Battery_Depth_of_Discharge)  # SOC cannot fall below the minimum allowed by the depth of discharge
    
    def Max_Power_Battery_Charge(model,ut):   # Max charge power = capacity / min charge time (C-rate limit)
        return model.Battery_Maximum_Charge_Power[ut] == (model.Battery_Units[ut]*model.Battery_Nominal_Capacity_milp)/model.Maximum_Battery_Charge_Time  # Max charge power = installed capacity / minimum charge time
    
    def Max_Power_Battery_Discharge(model,ut):  # Max discharge power = capacity / min discharge time (C-rate limit)
        return model.Battery_Maximum_Discharge_Power[ut] == (model.Battery_Units[ut]*model.Battery_Nominal_Capacity_milp)/model.Maximum_Battery_Discharge_Time  # Max discharge power = installed capacity / minimum discharge time
    
    def Max_Bat_in(model,s,yt,ut,t): # Minimun flow of energy for the charge fase
        return model.Battery_Inflow[s,yt,t] <= model.Battery_Maximum_Charge_Power[ut]*model.Delta_Time  # Cap charging power at the maximum charge rate
    
    def Max_Bat_out(model,s,yt,ut,t): # Minimum flow of energy for the discharge fase
        return model.Battery_Outflow[s,yt,t] <= model.Energy_Demand[s,yt,t]  # Discharge cannot exceed current demand
        
    def Battery_Min_Capacity(model,ut):      # Battery must meet the minimum autonomy requirement
        return   model.Battery_Units[ut]*model.Battery_Nominal_Capacity_milp >= model.Battery_Min_Capacity[ut]  # Installed battery capacity must meet the minimum autonomy requirement
    
    def Battery_Min_Step_Capacity(model,yt,ut):      # Battery capacity must not shrink between investment steps
        if ut > 1:  # After the first step: value must not decrease from the previous step
            return model.Battery_Units[ut] >= model.Battery_Units[ut-1]  # Battery unit count must not decrease from the previous step
        elif ut == 1:  # First step: no prior step to compare against
            return model.Battery_Units[ut] == model.Battery_Units[ut]  # First step: trivially satisfied (no previous step)
        
    # See the identical block in Constraints_Greenfield_Milp for the full rationale.
    def Max_Bat_flow_out_milp(model,s,yt,ut,t):
        return model.Battery_Outflow[s,yt,t] <= model.Battery_Maximum_Discharge_Power[ut]*model.Delta_Time

    def Max_Bat_flow_in_milp(model,s,yt,ut,t):
        return model.Battery_Inflow[s,yt,t] <= model.Battery_Maximum_Charge_Power[ut]*model.Delta_Time

    if Single_Flow_Linear:  # MGPY_BESS_FORM=linear -- mode gate only, cap handled above
        def Battery_Single_Flow_Discharge(model,s,yt,ut,t):
            return model.Battery_Outflow[s,yt,t] <= model.Energy_Demand[s,yt,t]*model.Single_Flow_BESS[s,yt,t]

        def Battery_Single_Flow_Charge(model,s,yt,ut,t):
            return model.Battery_Inflow[s,yt,t] <= model.Battery_Max_Inflow*(1-model.Single_Flow_BESS[s,yt,t])
    else:  # historical bilinear form (default)
        def Battery_Single_Flow_Discharge(model,s,yt,ut,t):  # Binary flag prevents simultaneous charge and discharge
            return   model.Battery_Outflow[s,yt,t] <= model.Single_Flow_BESS[s,yt,t]*model.Battery_Maximum_Discharge_Power[ut]*model.Delta_Time  # Discharge allowed only when the discharge-mode binary is active

        def Battery_Single_Flow_Charge(model,s,yt,ut,t):  # Charge only allowed when not discharging (complementary binary)
            return   model.Battery_Inflow[s,yt,t] <= (1-model.Single_Flow_BESS[s,yt,t])*model.Battery_Maximum_Charge_Power[ut]*model.Delta_Time  # Charge allowed only when not in discharge mode

    "Diesel generator constraints"
    if Generator_Partial_Load:  # Partial-load MILP modelling enabled
     def Minimum_Generator_Energy_Partial(model,s,yt,ut,g,t):   # Partial output >= min stable load (binary active flag x min fraction)
        return model.Generator_Energy_Partial[s,yt,g,t] >= (model.Generator_Nominal_Capacity_milp[g]*model.Generator_Min_output[g]*model.Delta_Time)*model.Generator_Partial[s,yt,g,t]  # Enforce minimum stable operating point when running in partial-load mode
    
     def Maximum_Generator_Energy_Partial(model,s,yt,ut,g,t):   # Partial output <= rated capacity of one partially-loaded unit
        return model.Generator_Energy_Partial[s,yt,g,t] <= model.Generator_Nominal_Capacity_milp[g]*model.Generator_Partial[s,yt,g,t]*model.Delta_Time  # Cap partial-load output at one unit's rated capacity
    
     def Maximum_Generator_Energy_Total_1(model,s,yt,ut,g,t):  # Total output <= all installed integer units at full rating
        return model.Generator_Energy_Total[s,yt,g,t] <= model.Generator_Nominal_Capacity_milp[g]*model.Generator_Units[ut,g]*model.Delta_Time  # Cap total output at installed integer capacity
    
     def Maximum_Generator_Energy_Total_2(model,s,yt,ut,g,t):  # Total output cannot exceed demand
        return model.Generator_Energy_Total[s,yt,g,t] <= model.Energy_Demand[s,yt,t]*model.Delta_Time  # Generator cannot supply more than the current demand
    
     def Generator_Energy_Total(model,s,yt,ut,g,t):  # Total = full-load committed blocks + partial-load energy
        return model.Generator_Energy_Total[s,yt,g,t] == (model.Generator_Full[s,yt,g,t]*model.Generator_Nominal_Capacity_milp[g]) + model.Generator_Energy_Partial[s,yt,g,t]  # Sum full-load block output and partial-load output
    
     def Generator_Units_Total(model,s,yt,ut,g,t):  # Committed units (full + at most one partial) <= installed units -- the rest may sit idle
        return model.Generator_Units[ut,g] >= model.Generator_Full[s,yt,g,t] + model.Generator_Partial[s,yt,g,t]  # FIX (2026-08-22): was `==`, which forced ALL installed units into full-or-partial commitment every hour regardless of demand -- with Generator_Nominal_Capacity_milp granularity of a few kW, "installed units" is in the tens/hundreds, so the old equality forced near-total genset output at every hour and made the model infeasible the instant demand ever fell below that floor. `>=` restores the standard unit-commitment reading: committed units cannot exceed what's installed, but idle units are allowed.
    else:  # No partial-load modelling: generators run at full capacity or off
     def Maximum_Generator_Energy_Total_1(model,s,yt,ut,g,t):  # Total output <= all installed integer units at full rating
        return model.Generator_Energy_Total[s,yt,g,t] <= model.Generator_Nominal_Capacity_milp[g]*model.Generator_Units[ut,g]*model.Delta_Time  # Cap total output at installed integer capacity
    
     def Maximum_Generator_Energy_Total_2(model,s,yt,ut,g,t):   # Total output cannot exceed demand
        return model.Generator_Energy_Total[s,yt,g,t] <= model.Energy_Demand[s,yt,t]*model.Delta_Time  # Generator cannot supply more than the current demand
    
    def Generator_Min_Step_Capacity(model,yt,ut,g):  # Capacity/unit count must not shrink between investment steps
        if ut > 1:  # After the first step: value must not decrease from the previous step
            return model.Generator_Units[ut,g] >= model.Generator_Units[ut-1,g]  # Generator unit count must not decrease from the previous step
        elif ut ==1:  # First step: no prior step to compare against
            return model.Generator_Units[ut,g] == model.Generator_Units[ut,g]  # First step: trivially satisfied (no previous step)

    def Generator_Max_Units(model,yt,ut,g):  # ADDED (2026-08-23): opt-in ceiling on installed generator capacity, tightens the Generator_Units LP relaxation -- see Model_Creation.py's Generator_Max_Capacity for the full rationale. No-op (10**9 kW) unless MGPY_MAX_GENERATOR_KW is set.
        return model.Generator_Units[ut,g]*model.Generator_Nominal_Capacity_milp[g] <= model.Generator_Max_Capacity[g]

    "Lost load constraints"
    def Maximum_Lost_Load(model,s,yt): # Maximum admittable lost load
        # Algebraically identical to the original form, which was written as
        #   Lost_Load_Fraction >= sum(Lost_Load) / sum(Energy_Demand)
        # but multiplied through by sum(Energy_Demand) instead of dividing by it.
        # Total annual demand is a constant, so moving it to the right-hand side
        # turns it into an RHS value rather than a matrix coefficient.
        #
        # WHY THIS MATTERS (measured 2026-07-26, not inferred): the divided form put
        # a coefficient of 1/sum(Energy_Demand) on every Lost_Load variable. With
        # demand in Wh that is ~9.9e-09 -- the SMALLEST coefficient anywhere in the
        # model, and the reason the logged matrix range reached [1e-08, 3e+06], a
        # ratio of 3.2e14 against Gurobi's guidance of below 1e9. It came from just
        # 30 rows out of 3.15 million. In this form the Lost_Load coefficients are
        # 1.0, and the worst remaining coefficient is 4.19e-06 in
        # BatteryReplacementCostAct -- roughly a 420x narrowing of the range.
        # Denominator is a sum of demands and is always strictly positive, so the
        # inequality direction is preserved.
        return sum(model.Lost_Load[s,yt,t] for t in model.periods) <= model.Lost_Load_Fraction * sum(model.Energy_Demand[s,yt,t] for t in model.periods)  # Lost load must not exceed the allowed fraction of total demand
    
    "Emission constraints"
    def RES_emission(model): #LCA emissions of RES
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration   # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]    # Will map each project year to its (year, step_index) tuple
        for y in model.years:        # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])            # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))              # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]    # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[model.Step_Duration*i + model.Step_Duration]  # Upgrade decision occurs at the end of each step
            
        return model.RES_emission == sum(model.RES_unit_CO2_emission[r]*((model.RES_Units_milp[1,r]*model.RES_Nominal_Capacity[r]) - model.RES_capacity[r]) for r in model.renewable_sources)+sum(sum(model.RES_unit_CO2_emission[r]*(model.RES_Units_milp[ut,r]-model.RES_Units_milp[ut-1,r])*model.RES_Nominal_Capacity[r] for (yt,ut) in tup_list) for r in model.renewable_sources)  # LCA emissions net of credit for pre-existing RES capacity, plus incremental upgrades
    
    def GEN_emission(model): #LCA emissions of generator
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration   # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]    # Will map each project year to its (year, step_index) tuple
        for y in model.years:        # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])            # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))              # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]    # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[model.Step_Duration*i + model.Step_Duration]  # Upgrade decision occurs at the end of each step
            
        return model.GEN_emission == sum(((model.Generator_Units[1,g]*model.Generator_Nominal_Capacity_milp[g])-model.Generator_capacity[g])*model.GEN_unit_CO2_emission[g] for g in model.generator_types)+sum(sum(((model.Generator_Units[ut,g]-model.Generator_Units[ut-1,g])*model.Generator_Nominal_Capacity_milp[g])*model.GEN_unit_CO2_emission[g] for (yt,ut) in tup_list) for g in model.generator_types)  # LCA emissions net of credit for pre-existing generator capacity, plus incremental upgrades
    
    def FUEL_emission(model,s,yt,ut,g,t): #Emissions from fuel consumption
        return model.FUEL_emission[s,yt,g,t] == model.Generator_Energy_Total[s,yt,g,t]/model.Fuel_LHV[g]/model.Generator_Efficiency[g]*model.FUEL_unit_CO2_emission[g]   # Emissions = energy / (fuel LHV x generator efficiency) x fuel emission factor
    
    def GRID_emission(model, s, y, t):  # Emissions from grid electricity imports, using the national grid CO2 factor
        if y >= model.Year_Grid_Connection:  # Only count grid emissions once the connection year has been reached
            return model.GRID_emission[s,y,t] == model.Energy_From_Grid[s,y,t] * model.National_Grid_Specific_CO2_emissions  # Import energy x national grid CO2 factor / 1000
        else:  # Before the connection year: no grid emissions
            return model.GRID_emission[s, y, t] == 0      # No grid emissions before the connection year

    
    def BESS_emission(model): #LCA emissions of generator
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Start year of each investment step (step 1 begins at year 1)
        s_dur = model.Step_Duration   # Number of years in each investment step
        for i in range(1, len(model.steps)):   # Each subsequent step starts s_dur years after the previous
            upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur    # Start year of step i+1
        yu_tuples_list = [[] for i in model.years]    # Will map each project year to its (year, step_index) tuple
        for y in model.years:        # Loop over every project year
            for i in range(len(upgrade_years_list)-1):  # Find which investment step year y belongs to
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year y falls within investment step i+1
                    yu_tuples_list[y-1] = (y, model.steps[i+1])            # Record (year, step) pair
                elif y >= upgrade_years_list[-1]:  # Year y falls in the final investment step
                    yu_tuples_list[y-1] = (y, len(model.steps))              # Record (year, final step) pair
        tup_list = [[] for i in range(len(model.steps)-1)]    # (year, step) pairs at each capacity upgrade decision point
        for i in range(0, len(model.steps) - 1):  # Build the list of upgrade decision points
            tup_list[i] = yu_tuples_list[model.Step_Duration*i + model.Step_Duration]  # Upgrade decision occurs at the end of each step
            
        return model.BESS_emission == ((model.Battery_Units[1]*model.Battery_Nominal_Capacity_milp)-model.Battery_capacity)*model.BESS_unit_CO2_emission+sum(((model.Battery_Units[ut]-model.Battery_Units[ut-1])*model.Battery_Nominal_Capacity_milp)*model.BESS_unit_CO2_emission for (yt,ut) in tup_list)  # LCA emissions net of credit for pre-existing battery capacity, plus incremental upgrades
        
    def Scenario_FUEL_emission(model,s):   # Total fuel-burn CO2 emissions for scenario s, summed over years/generators/periods
        return model.Scenario_FUEL_emission[s] == sum(sum(sum(model.Generator_Energy_Total[s,y,g,t]/model.Fuel_LHV[g]/model.Generator_Efficiency[g]*model.FUEL_unit_CO2_emission[g]  for t in model.periods) for y in model.years) for g in model.generator_types)   # Sum fuel emissions over all periods, years and generator types
    
    def Scenario_GRID_emission(model, s):  # Total grid-import CO2 emissions for scenario s, summed over years/periods
        Total_Grid_Emission = 0  # Accumulator for total grid emissions across all years
        for y in range(1, model.Years + 1):  # Loop over every project year
            if y >= model.Year_Grid_Connection:  # Only count grid emissions once the connection year has been reached
                Total_Grid_Emission += sum(model.Energy_From_Grid[s, y, t] * model.National_Grid_Specific_CO2_emissions  for t in model.periods)  # Yearly grid emissions = imports x national grid CO2 factor / 1000
        return model.Scenario_GRID_emission[s] == Total_Grid_Emission  # Return total scenario grid emissions constraint
    
    "Grid constraints"  
    def Maximum_Power_From_Grid(model,s,y,t):  # Cap imports from grid; zero before connection year or during outages
        if y < model.Year_Grid_Connection or model.Grid_Availability[s, y, t] == 0:  # Grid not yet connected or currently unavailable
            return model.Energy_From_Grid[s,y,t] == 0  # No imports possible
        else:  # Grid connected and available: cap imports at the max grid power
            return model.Energy_From_Grid[s,y,t] <= model.Maximum_Grid_Power   # Cap imports at maximum grid power (kW to W)
    
    def Maximum_Power_To_Grid(model,s,y,t):  # Cap exports to grid; zero if not allowed or grid unavailable
        if y < model.Year_Grid_Connection or model.Grid_Connection_Type == 1 or model.Grid_Availability[s, y, t] == 0:  # Export not allowed (import-only) or grid unavailable/not yet connected
            return model.Energy_To_Grid[s, y, t] == 0  # No exports possible
        elif model.Grid_Connection_Type == 0:  # Bidirectional connection: exports are allowed
            return model.Energy_To_Grid[s, y, t] <= model.Maximum_Grid_Power  # Cap exports at maximum grid power (kW to W)
        
    def Single_Flow_Energy_To_Grid(model,s,yt,ut,t):  # Binary Single_Flow_Grid prevents exporting while importing
        return model.Energy_To_Grid[s,yt,t] <= model.Single_Flow_Grid[s,yt,t]*model.Large_Constant  # Export allowed only when the export-mode binary is active
        
    def Single_Flow_Energy_From_Grid(model,s,yt,ut,t):  # Complementary binary: import only when not exporting
        return model.Energy_From_Grid[s,yt,t] <= (1-model.Single_Flow_Grid[s,yt,t])*model.Large_Constant     # Import allowed only when the export-mode binary is inactive



