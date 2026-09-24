"""
MicroGridsPy - Multi-year capacity-expansion (MYCE)

Linear Programming framework for microgrids least-cost sizing,
able to account for time-variable load demand evolution and capacity expansion.

Authors: 
    Alessandro Onori   - Department of Energy, Politecnico di Milano
    Giulia Guidicini   - Department of Energy, Politecnico di Milano 
    Lorenzo Rinaldi    - Department of Energy, Politecnico di Milano
    Nicolò Stevanato   - Department of Energy, Politecnico di Milano / Fondazione Eni Enrico Mattei
    Francesco Lombardi - Department of Energy, Politecnico di Milano
    Emanuela Colombo   - Department of Energy, Politecnico di Milano
    
Based on the original model by:
    Sergio Balderrama  - Department of Mechanical and Aerospace Engineering, University of Liège / San Simon University, Centro Universitario de Investigacion en Energia
    Sylvain Quoilin    - Department of Mechanical Engineering Technology, KU Leuven
"""


import pandas as pd  # Import pandas for data manipulation and CSV loading
import re  # Import regex module for pattern-based text parsing
import os  # Import os module for file and directory path operations
import params_path  # Resolves which parameters file this run reads (MGPY_PARAMS)
from RE_calculation import RE_supply  # Import renewable energy supply calculation function
from Demand import demand_generation  # Import demand profile generation function
from Grid_Availability import grid_availability as grid_avail  # Import grid availability simulation function


#%% This section extracts the values of Scenarios, Periods, Years from data.dat and creates ranges for them


current_directory = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the directory containing this script
inputs_directory = os.path.join(current_directory, '..', 'Inputs')  # Build path to the Inputs folder one level up
data_file_path = params_path.PARAMS_PATH  # Model parameters file, resolved once via MGPY_PARAMS (see params_path.py)
demand_file_path = params_path.DEMAND_PATH  # Electric demand CSV, resolved once via MGPY_DEMAND (see params_path.py)
res_file_path = params_path.RES_PATH  # Renewable energy time series CSV, resolved once via MGPY_RES (see params_path.py)
fuel_file_path = os.path.join(inputs_directory, 'Fuel Specific Cost.csv')  # Full path to the fuel cost CSV file
grid_file_path = os.path.join(inputs_directory, 'Grid Availability.csv')  # Full path to the grid availability CSV file
results_directory = os.path.join(current_directory, '..', 'Results')  # Full path to the Results output directory
plot_path = os.path.join(results_directory, '..', 'Plots')  # Full path to the Plots output directory

Data_import = open(data_file_path).readlines()  # Read all lines of the parameters file into a list

Fuel_Specific_Start_Cost = []  # Initialize empty list to store per-generator fuel start costs
Fuel_Specific_Cost_Rate = []  # Initialize empty list to store per-generator annual fuel cost escalation rates

for i in range(len(Data_import)):  # Iterate over every line in the parameters file
    if "param: Scenarios" in Data_import[i]:  # Check if the line declares the number of scenarios
        n_scenarios = int((re.findall('\d+',Data_import[i])[0]))  # Extract and store the number of scenarios as integer
    if "param: Years" in Data_import[i]:  # Check if the line declares the number of years
        n_years = int((re.findall('\d+',Data_import[i])[0]))  # Extract and store the number of simulation years as integer
    if "param: Periods" in Data_import[i]:  # Check if the line declares the number of periods
        n_periods = int((re.findall('\d+',Data_import[i])[0]))  # Extract and store the number of time periods per year as integer
    if "param: Generator_Types" in Data_import[i]:  # Check if the line declares the number of generator types
        n_generators = int((re.findall('\d+',Data_import[i])[0]))  # Extract and store the number of generator types as integer
    if "param: Step_Duration" in Data_import[i]:  # Check if the line declares the investment step duration
        step_duration = int((re.findall('\d+',Data_import[i])[0]))  # Extract and store the number of years per investment step
    if "param: Min_Last_Step_Duration" in Data_import[i]:  # Check if the line declares the minimum duration of the last step
        min_last_step_duration = int((re.findall('\d+',Data_import[i])[0]))  # Extract and store minimum last-step duration in years
    if "param: Battery_Independence" in Data_import[i]:  # Check if the line declares the battery autonomy requirement
        Battery_Independence = int((re.findall('\d+',Data_import[i])[0]))  # Extract and store battery independence days as integer
    if "param: Renewable_Penetration" in Data_import[i]:  # Check if the line declares the required renewable penetration fraction
        Renewable_Penetration = float((re.findall("\d+\.\d+|\d+",Data_import[i])[0]))  # Extract and store renewable penetration as a float
    if "param: Greenfield_Investment" in Data_import[i]:  # Check if the line declares whether greenfield investment is assumed
        Greenfield_Investment = int((re.findall('\d+',Data_import[i])[0]))  # Extract greenfield investment flag (0 or 1)
    if "param: Multiobjective_Optimization" in Data_import[i]:  # Check if multi-objective optimization is enabled
        Multiobjective_Optimization = int((re.findall('\d+',Data_import[i])[0]))  # Extract multi-objective flag (0 or 1)
    if "param: Optimization_Goal" in Data_import[i]:  # Check if the line specifies the optimization objective
        Optimization_Goal = int((re.findall('\d+',Data_import[i])[0]))  # Extract optimization goal identifier (e.g., 1=cost, 2=emissions)
    if "param: MILP_Formulation" in Data_import[i]:  # Check if MILP formulation is selected
        MILP_Formulation = int((re.findall('\d+',Data_import[i])[0]))  # Extract MILP flag (0=LP, 1=MILP)
    if "param: Generator_Partial_Load" in Data_import[i]:  # Check if partial load modeling for generators is enabled
        Generator_Partial_Load = int((re.findall('\d+',Data_import[i])[0]))  # Extract partial load flag (0 or 1)
    if "param: Plot_Max_Cost" in Data_import[i]:  # Check if the line sets a maximum cost for plotting
        Plot_Max_Cost = int((re.findall('\d+',Data_import[i])[0]))  # Extract maximum cost value for plot axis scaling
    if "param: RE_Supply_Calculation" in Data_import[i]:  # Check if RE supply should be calculated internally
        RE_Supply_Calculation = int((re.findall('\d+',Data_import[i])[0]))  # Extract RE supply calculation flag (0=external, 1=internal)
    if "param: Demand_Profile_Generation" in Data_import[i]:  # Check if demand profile should be generated using archetypes
        Demand_Profile_Generation = int((re.findall('\d+',Data_import[i])[0]))  # Extract demand generation flag (0=load CSV, 1=generate)
    if "param: Fuel_Specific_Cost_Calculation" in Data_import[i]:  # Check if fuel cost should be calculated dynamically
        Fuel_Specific_Cost_Calculation = int((re.findall('\d+',Data_import[i])[0]))  # Extract fuel cost calculation flag (0=fixed, 1=dynamic)
    if "param: Fuel_Specific_Cost_Import" in Data_import[i]:  # Check if fuel cost data should be imported from a file
        Fuel_Specific_Cost_Import = int((re.findall('\d+',Data_import[i])[0]))  # Extract fuel cost import flag (0=calculate, 1=import)
    if "param: Grid_Average_Number_Outages" in Data_import[i]:  # Check if the line specifies average number of grid outages
        average_n_outages = int((re.findall('\d+',Data_import[i])[0]))  # Extract average number of grid outages per year
    if "param: Grid_Average_Outage_Duration" in Data_import[i]:  # Check if the line specifies average grid outage duration
        average_outage_duration = int((re.findall('\d+',Data_import[i])[0]))  # Extract average duration of each grid outage in hours
    if "param: Grid_Connection " in Data_import[i]:  # Check if a grid connection is present (note trailing space in param name)
        Grid_Connection = int((re.findall('\d+',Data_import[i])[0]))  # Extract grid connection flag (0=off-grid, 1=grid-connected)
    if "param: Grid_Availability_Simulation" in Data_import[i]:  # Check if grid availability should be simulated stochastically
        Grid_Availability_Simulation = int((re.findall('\d+',Data_import[i])[0]))  # Extract grid availability simulation flag (0 or 1)
    if "param: Year_Grid_Connection " in Data_import[i]:  # Check if the line specifies the year when grid connection starts
        year_grid_connection = int((re.findall('\d+',Data_import[i])[0]))  # Extract the year in which grid connection becomes available
    if "param: Model_Components " in Data_import[i]:  # Check if the line specifies which model components to include
        Model_Components = int((re.findall('\d+',Data_import[i])[0]))  # Extract model components bitmask/flag
    if "param: WACC_Calculation " in Data_import[i]:  # Check if WACC should be computed from equity/debt inputs
        WACC_Calculation = int((re.findall('\d+',Data_import[i])[0]))  # Extract WACC calculation flag (0=use fixed rate, 1=compute WACC)
    if "param: cost_of_equity" in Data_import[i]:  # Check if the line provides cost of equity
        cost_of_equity = float((re.findall("\d+\.\d+|\d+",Data_import[i])[0]))  # Extract cost of equity as a decimal fraction
    if "param: cost_of_debt" in Data_import[i]:  # Check if the line provides cost of debt
        cost_of_debt = float((re.findall("\d+\.\d+|\d+",Data_import[i])[0]))  # Extract cost of debt as a decimal fraction
    if "param: tax" in Data_import[i]:  # Check if the line provides the corporate tax rate
        tax = float((re.findall("\d+\.\d+|\d+",Data_import[i])[0]))  # Extract tax rate as a decimal fraction
    if "param: equity_share" in Data_import[i]:  # Check if the line provides the equity share of financing
        equity_share = float((re.findall("\d+\.\d+|\d+",Data_import[i])[0]))  # Extract equity share as a decimal fraction
    if "param: debt_share" in Data_import[i]:  # Check if the line provides the debt share of financing
        debt_share = float((re.findall("\d+\.\d+|\d+",Data_import[i])[0]))  # Extract debt share as a decimal fraction
    if "param: Real_Discount_Rate" in Data_import[i]:  # Check if the line provides the real discount rate
        Discount_Rate_default = float((re.findall("\d+\.\d+|\d+|\d+",Data_import[i])[0]))  # Extract default real discount rate as a float
    if "param: Fuel_Specific_Start_Cost" in Data_import[i]:  # Check if the block of fuel start costs begins here
        for j in range(n_generators):  # Iterate over each generator type
            Fuel_Specific_Start_Cost.append(float((re.findall("\d+\s+(\d+\.\d+|\d+)",Data_import[i+1+j])[0])))  # Parse and append each generator's fuel start cost
    if "param: Fuel_Specific_Cost_Rate" in Data_import[i]:  # Check if the block of fuel cost escalation rates begins here
        for j in range(n_generators):  # Iterate over each generator type
            Fuel_Specific_Cost_Rate.append(float((re.findall("\d+\s+(\d+\.\d+|\d+)",Data_import[i+1+j])[0])))  # Parse and append each generator's annual fuel cost escalation rate


scenario = [i for i in range(1,n_scenarios+1)]  # Build a list of scenario indices from 1 to n_scenarios
year = [i for i in range(1,n_years+1)]  # Build a list of year indices from 1 to n_years
period = [i for i in range(1,n_periods+1)]  # Build a list of period indices from 1 to n_periods
generator = [i for i in range(1,n_generators+1)]  # Build a list of generator indices from 1 to n_generators

#%% This section imports, generates and plots the different types of demands

if Demand_Profile_Generation:  # If endogenous demand profile generation is enabled
    Demand = demand_generation()  # Generate demand profiles using archetype-based model
    Demand.columns = Demand.columns.map(str)  # Convert column names to strings for consistency
    print("Electric demand data generated endogenously using archetypes")  # Log confirmation of endogenous generation
else:  # Otherwise load demand data from an external CSV file
    delimiters = [(',', '.'), (';', ','), (';', '.')]  # List of (delimiter, decimal) combinations to try when parsing the CSV
    loaded_successfully = False  # Flag to track whether the CSV was loaded without errors
    for delimiter, decimal in delimiters:  # Try each delimiter/decimal combination in order
        try:
            # Set index_col to 0 if the first column is the index
            Demand = pd.read_csv(demand_file_path, delimiter=delimiter, decimal=decimal, header=0, index_col=0)  # Attempt to load Demand CSV with current delimiter/decimal
            print(f"Demand data loaded exogenously using delimiter '{delimiter}' and decimal '{decimal}'")  # Log successful load
            loaded_successfully = True  # Mark loading as successful
            break  # Stop trying further combinations
        except pd.errors.ParserError:  # Catch CSV parsing errors caused by wrong delimiter/decimal
            print(f"Failed to load with delimiter '{delimiter}' and decimal '{decimal}'. Trying next combination...")  # Log failure and continue
        except FileNotFoundError:  # Catch missing file error
            print(f"File not found: {demand_file_path}. Please check the file path and try again.")  # Log file not found message
            raise  # Re-raise exception to stop execution
        except Exception as e:  # Catch any other unexpected exceptions
            print(f"An unexpected error occurred: {e}. Trying next combination...")  # Log unexpected error and continue
    
    if not loaded_successfully:  # If all delimiter combinations failed
        print("Error during import of Demand.csv: unable to automatically detect delimiter and decimal. Please try again using delimiter ';' or ',' and decimal ',' or '.'.")  # Log detailed error
        raise ValueError("Failed to load demand data with all provided delimiter and decimal combinations.")  # Raise error to halt execution

# Validate DataFrame dimensions against expected years and periods
expected_columns = n_scenarios * len(year)  # Expected number of data columns, excluding the index (n_years per scenario, concatenated for all scenarios)
expected_rows = len(period)  # Expected number of rows equals the number of time periods per year

# Validate columns
if Demand.shape[1] < expected_columns:  # Check if file has fewer columns than expected
    raise ValueError(f"Number of columns in the file ({Demand.shape[1]}) is less than the expected number of years ({expected_columns}): unable to proceed. Please check the Demand.csv file.")  # Raise error for insufficient columns
elif Demand.shape[1] > expected_columns:  # Check if file has more columns than expected
    # FATAL BY DEFAULT since 2026-07-26. This used to print a warning and silently
    # trim, which cost four wrong runs across two days: Demand.csv was swapped to
    # the 6sc/10y profile at 16:56 on 25 July (one minute before the first 10y job
    # started) and never swapped back, so every subsequent "5y" run silently used
    # the first 5 columns of the 10-YEAR demand. The runs completed and looked
    # healthy -- only the numbers were wrong. The tell was the LP relaxation moving
    # from 96,153.90 (job 58519303) to ~102,457 with identical model structure;
    # the warning itself was buried under thousands of solver log lines.
    #
    # Trimming is only ever correct when the extra columns are genuinely surplus.
    # It is NOT correct when the file belongs to a different horizon, and nothing
    # here can tell those two cases apart -- so refuse, and make the human decide.
    _message = (f"DEMAND FILE MISMATCH -- refusing to start.\n"
                f"  {demand_file_path}\n"
                f"  has {Demand.shape[1]} data columns, but this configuration expects "
                f"{expected_columns} ({n_scenarios} scenarios x {len(year)} years).\n"
                f"  This usually means Demand.csv belongs to a different planning horizon "
                f"than the parameters file.\n"
                f"  Fix: copy the matching Demand_<config>.csv over Code/Inputs/Demand.csv.\n"
                f"  To trim to the first {expected_columns} columns anyway (only correct if "
                f"the extra columns are genuinely surplus), set MGPY_ALLOW_DEMAND_TRIM=1.")
    if os.environ.get('MGPY_ALLOW_DEMAND_TRIM') == '1':  # Explicit opt-in escape hatch
        print(f"Warning: {_message}\n  Proceeding anyway because MGPY_ALLOW_DEMAND_TRIM=1.")  # Loud, but permitted
        Demand = Demand.iloc[:, :expected_columns]  # Trim DataFrame to only the expected number of columns
    else:
        raise ValueError(_message)  # Stop before the solver burns any walltime on wrong data

# Validate rows
if Demand.shape[0] < expected_rows:  # Check if file has fewer rows than expected
    raise ValueError(f"Number of rows in the file ({Demand.shape[0]}) is less than the expected number of periods ({expected_rows}): unable to proceed.Please check the Demand.csv file.")  # Raise error for insufficient rows

Electric_Energy_Demand_Series = pd.Series(dtype=float)  # Initialize an empty float Series to accumulate demand values
# Iterate over actual column names in the DataFrame

for col in Demand.columns:  # Iterate over every column in the Demand DataFrame
    dum = Demand[col].reset_index(drop=True)  # Extract column values and reset index to 0-based integers
    Electric_Energy_Demand_Series = pd.concat([Electric_Energy_Demand_Series, dum])  # Append column values to the accumulating Series

frame = [scenario, year, period]  # Define the axes for the multi-level index: scenarios, years, periods
index = pd.MultiIndex.from_product(frame, names=['scenario', 'year', 'period'])  # Create a Cartesian-product MultiIndex from scenario/year/period lists
Electric_Energy_Demand = pd.DataFrame(Electric_Energy_Demand_Series)  # Convert the concatenated Series to a DataFrame
Electric_Energy_Demand.index = index  # Assign the MultiIndex to the demand DataFrame
Electric_Energy_Demand_dict = Electric_Energy_Demand[0].to_dict()  # Pre-convert once -- Initialize_Demand is called ~500K+ times during instantiation; a plain dict lookup avoids repeated pandas MultiIndex tuple resolution, which is what made pandas 3.0.3 pathologically slow here (2.1.1 handled it fine, but this is the actual hot path regardless of version)

Electric_Energy_Demand_2 = pd.DataFrame()  # Initialize an empty DataFrame to hold demand keyed by scenario column
# Iterate over scenarios and years, assuming scenario and year are defined and match the CSV structure
for s in scenario:  # Iterate over each scenario index
    Electric_Energy_Demand_Series_2 = pd.Series()  # Initialize an empty Series to accumulate demand for this scenario
    for y in year:  # Iterate over each year within the scenario
        # Construct the column name as it appears in the CSV headers
        column_name = f'{(s-1)*len(year) + y}'  # Compute the CSV column index corresponding to this (scenario, year) pair
        if column_name in Demand.columns:  # Check that the expected column actually exists in the DataFrame
            dum_2 = Demand[column_name].dropna().reset_index(drop=True)  # Extract column, drop NaN values, and reset index
            Electric_Energy_Demand_Series_2 = pd.concat([Electric_Energy_Demand_Series_2, dum_2], ignore_index=True)  # Append this year's demand to the scenario Series
        else:  # Handle the case where the expected column is missing
            print(f"Warning: Column '{column_name}' does not exist in the Demand DataFrame")  # Warn about the missing column
    Electric_Energy_Demand_2[s] = Electric_Energy_Demand_Series_2  # Store the concatenated scenario demand as a column

# Create a RangeIndex for Electric_Energy_Demand_2
index_2 = pd.RangeIndex(1, len(year) * len(period) + 1)  # Create a 1-based integer index covering all year×period combinations
Electric_Energy_Demand_2.index = index_2  # Assign the range index to the secondary demand DataFrame

"Electric Demand"
def Initialize_Demand(model, s, y, t):
    """
    Initializes electric demand based on the scenario, year, and period.

    Parameters:
    model (object): The model for which to initialize electric demand.
    s (int): Scenario number.
    y (int): Year.
    t (int): Time period.

    Returns:
    float: The electric demand.
    """
    return float(Electric_Energy_Demand_dict[(s, y, t)])  # Look up and return the demand value for the given (scenario, year, period) key

#%% This section imports or generates the renewables and temperature time series data 

if RE_Supply_Calculation == 0:  # If external RE time series data should be loaded from a file
    delimiters = [(',', '.'), (';', ','), (';', '.')]  # List of (delimiter, decimal) combinations to attempt
    loaded_successfully = False  # Flag indicating whether the CSV was successfully loaded
    for delimiter, decimal in delimiters:  # Try each delimiter/decimal pair in sequence
        try:
            # Set index_col to 0 if the first column is the index
            Renewable_Energy = pd.read_csv(res_file_path, delimiter=delimiter, decimal=decimal, header=0, index_col=0)  # Load RE time series CSV with current format settings
            print(f"Renewables Time Series data loaded exogenously using delimiter '{delimiter}' and decimal '{decimal}'")  # Log successful load
            loaded_successfully = True  # Mark as successfully loaded
            break  # Stop trying further combinations
        except pd.errors.ParserError:  # Catch parsing failures due to incorrect delimiter or decimal
            print(f"Failed to load with delimiter '{delimiter}' and decimal '{decimal}'. Trying next combination...")  # Log and continue to next combination
        except FileNotFoundError:  # Catch missing file error
            print(f"File not found: {res_file_path}. Please check the file path and try again.")  # Log file not found
            raise  # Re-raise to halt execution
        except Exception as e:  # Catch any other unforeseen errors
            print(f"An unexpected error occurred: {e}. Trying next combination...")  # Log error and continue
    
    if not loaded_successfully:  # If all combinations failed
        print("Error during import of RES_Time_Series.csv: unable to automatically detect delimiter and decimal. Please try again using delimiter ';' or ',' and decimal ',' or '.'.")  # Log failure
        raise ValueError("Failed to load renewables data with all provided delimiter and decimal combinations.")  # Raise to halt execution
    
    plot_path = os.path.join(results_directory, 'Renewables Availability.png')  # Set plot path for renewables availability figure
else:  # If RE supply should be computed internally from NASA POWER data
    Renewable_Energy = RE_supply()  # Generate renewable energy time series using the RE_supply function
    Renewable_Energy = Renewable_Energy.set_index(pd.Index(range(1, n_periods+1)), inplace=False)  # Assign 1-based period index to the generated RE DataFrame
    print("Renewables Time Series data generated endogenously using NASA POWER")  # Log confirmation of endogenous generation

Renewable_Energy_array = Renewable_Energy.to_numpy()  # Pre-convert once -- Initialize_RES_Energy is called over a million times during instantiation; raw numpy indexing avoids repeated pandas .iloc overhead, the same category of hot-path cost that made pandas 3.0.3 pathologically slow here

# Zero out near-noise capacity-factor values (e.g. dawn/dusk irradiance) before they
# reach Gurobi: a solar panel at <0.5% of rated output isn't a meaningful dispatch
# choice, and Gurobi drops exact zeros from the matrix, so this trims tiny coefficients
# without changing dispatch. No-op on the current default dataset (RES_Time_Series.csv's
# nonzero values already range 0.00294-0.287) -- kept defensively for other datasets.
Renewable_Energy_array[Renewable_Energy_array < 0.005] = 0.0

def Initialize_RES_Energy(model, s, r, t):
    """
    Initializes renewable energy supply based on the specified scenario, resource, and time period.

    Parameters:
    model (object): The model for which the renewable energy supply is initialized.
    scenario (int): The scenario number.
    resource (int): The resource index.
    time_period (int): The time period index.

    Returns:
    float: The amount of renewable energy supplied.
    """
    column = (s - 1) * model.RES_Sources + r  # Compute the column index in the RE DataFrame for this (scenario, resource) pair
    return float(Renewable_Energy_array[t - 1, column - 1])   # Return the RE generation value at the specified period and column (1-based to 0-based conversion)

#%% This section defines the number of investment steps as well as assigns each year to its corresponding step

def Initialize_Upgrades_Number(model):
    """
    Calculates the number of upgrades for the given model.

    Parameters:
    model (object): The model for which to calculate upgrades.

    Returns:
    int: Number of upgrades.
    """
    if n_years % step_duration == 0:  # Check if years divide evenly into steps with no remainder
        n_upgrades = n_years/step_duration  # Compute exact number of equal-length investment steps
        return n_upgrades  # Return the number of upgrade steps
    
    else:  # Handle case where years do not divide evenly into steps
        n_upgrades = 1  # Start with one step (the first step always exists)
        for y in  range(1, n_years + 1):  # Iterate over every year in the planning horizon
            if y % step_duration == 0 and n_years - y > min_last_step_duration:  # Add a new step if this year is a step boundary and remaining years exceed the minimum last step duration
                n_upgrades += 1  # Increment the step counter
        return int(n_upgrades)  # Return the total number of investment steps as an integer

def Initialize_YearUpgrade_Tuples(model):
    """
    Initializes year-upgrade tuples for the model.

    Parameters:
    model (object): The model for which to initialize year-upgrade tuples.

    Returns:
    list: List of year-upgrade tuples.
    """
    upgrade_years_list = [1 for i in range(len(model.steps))]  # Initialize start-year list with 1 for every step
    s_dur = model.Step_Duration  # Store step duration for concise reference
    for i in range(1, len(model.steps)):  # Iterate over steps after the first
        upgrade_years_list[i] = upgrade_years_list[i-1] + s_dur  # Set each step's start year by adding step duration to the previous
    yu_tuples_list = [0 for i in model.years]  # Initialize the output list with placeholders for each year
    if model.Steps_Number == 1:  # Special case: only one investment step covers all years
        for y in model.years:  # Iterate over all years
            yu_tuples_list[y-1] = (y, 1)  # Assign each year to step 1
    else:  # General case: multiple investment steps
        for y in model.years:  # Iterate over all years
            for i in range(len(upgrade_years_list)-1):  # Check each intermediate step boundary
                if y >= upgrade_years_list[i] and y < upgrade_years_list[i+1]:  # Year falls within this step's range
                    yu_tuples_list[y-1] = (y, model.steps[i+1])  # Assign year to the corresponding step index
                elif y >= upgrade_years_list[-1]:  # Year falls in or beyond the last step
                    yu_tuples_list[y-1] = (y, len(model.steps))  # Assign year to the final step
    print('\nTime horizon (year,investment-step): ' + str(yu_tuples_list))  # Print the completed year-to-step mapping
    return yu_tuples_list  # Return the list of (year, step) tuples


#%% This section initializes economic parameters related to the project

def Initialize_Discount_Rate(model):
    """
    Calculates and initializes the discount rate for the model.

    Parameters:
    model (object): The model for which to calculate the discount rate.

    Returns:
    float: The calculated discount rate.
    """
    if WACC_Calculation:  # If WACC should be computed from financing structure
        if equity_share == 0:  # Edge case: pure debt financing (no equity)
            discount_rate = cost_of_debt*(1-tax)  # After-tax cost of debt equals the discount rate
        else:
           # Definition of Leverage (L): risk perceived by investors, or viceversa as the attractiveness of the investment to external debtors.
           L = debt_share/equity_share  # Compute leverage ratio as debt-to-equity
           discount_rate = cost_of_debt*(1-tax)*L/(1+L) + cost_of_equity*1/(1+L)  # Compute WACC as weighted sum of after-tax debt cost and equity cost
           print("Weighted Average Cost of Capital calculation completed")  # Log WACC calculation success
    else:  # Use the pre-specified fixed discount rate from the parameters file
        discount_rate = Discount_Rate_default  # Assign default real discount rate
    return discount_rate  # Return the computed or default discount rate

################################################################## ELECTRICITY PRODUCTION ##########################################################################

#%% This section initializes parameters related to the battery bank

def Initialize_Battery_Unit_Repl_Cost(model):
    """
    Initializes the unit replacement cost of the battery based on the model parameters.

    Parameters:
    model (object): The model containing parameters related to battery cost and performance.

    Returns:
    float: The calculated unit replacement cost of the battery.
    """
    Unitary_Battery_Cost = model.Battery_Specific_Investment_Cost - model.Battery_Specific_Electronic_Investment_Cost  # Compute the cell-only battery cost by subtracting the electronics cost
    return Unitary_Battery_Cost/(model.Battery_Cycles*2*(model.Battery_Depth_of_Discharge))  # Divide cost by total energy throughput over lifetime (cycles × 2 × DoD) to get unit replacement cost per kWh
      

def Initialize_Battery_Minimum_Capacity(model,ut): 
    """
    Calculates the minimum capacity required for the battery bank based on the model's demand profile and battery independence criteria.

    Parameters:
    model (object): The model containing parameters and configurations for battery and demand.
    ut (int): The current upgrade step in the model.

    Returns:
    float: The minimum required battery capacity to ensure the desired level of battery independence.
    """
    if model.Battery_Independence == 0:  # If no battery autonomy requirement is set
        return 0  # No minimum capacity constraint; return zero
    else:  # Battery autonomy is required
        Periods = model.Battery_Independence*24  # Convert autonomy days to periods (assuming hourly resolution)
        Len =  int(model.Periods*model.Years/Periods)  # Compute the number of grouping windows over the full time horizon
        Grouper = 1  # Initialize group counter for period grouping
        index = 1  # Initialize row index for iterating over Electric_Energy_Demand_2
        for i in range(1, Len+1):  # Iterate over each demand-grouping window
            for j in range(1,Periods+1):  # Iterate over each period within the window
                Electric_Energy_Demand_2.loc[index, 'Grouper'] = Grouper  # Assign group label to this period's row
                index += 1  # Advance to the next row
            Grouper += 1  # Move to the next group
    
        upgrade_years_list = [1 for i in range(len(model.steps))]  # Initialize list of upgrade start years (all start at year 1)
        
        for u in range(1, len(model.steps)):  # Iterate over steps after the first
            upgrade_years_list[u] =upgrade_years_list[u-1] + model.Step_Duration  # Set each step's start year cumulatively
        if model.Steps_Number ==1:  # Single-step case: use the full demand dataset
            Electric_Energy_Demand_Upgrade = Electric_Energy_Demand_2  # Assign full demand DataFrame to upgrade slice
        else:  # Multi-step case: slice demand to the relevant upgrade window
            if ut==1:  # First upgrade step
                start = 0  # Start from the beginning of the dataset
                Electric_Energy_Demand_Upgrade = Electric_Energy_Demand_2.loc[start : model.Periods*(upgrade_years_list[ut]-1), :]  # Slice demand up to the end of the first step
            elif ut == len(model.steps):  # Last upgrade step
                start = model.Periods*(upgrade_years_list[ut-1] -1)+1  # Start from the beginning of the last step
                Electric_Energy_Demand_Upgrade = Electric_Energy_Demand_2.loc[start :, :]  # Slice demand from the last step start to the end
            else:  # Intermediate upgrade step
                start = model.Periods*(upgrade_years_list[ut-1] -1)+1  # Start from the beginning of this step
                Electric_Energy_Demand_Upgrade = Electric_Energy_Demand_2.loc[start : model.Periods*(upgrade_years_list[ut]-1), :]  # Slice demand to the end of this step
        
        Period_Energy = Electric_Energy_Demand_Upgrade.groupby(['Grouper']).sum()  # Sum demand within each autonomy window group
        Period_Average_Energy = Period_Energy.mean()  # Compute the average energy across all groups for each scenario
        Available_Energy = sum(Period_Average_Energy[s]*model.Scenario_Weight[s] for s in model.scenarios)  # Compute weighted average energy across scenarios using scenario probabilities
        
        return Available_Energy/(model.Battery_Depth_of_Discharge)  # Return minimum battery capacity as average energy divided by depth-of-discharge


#%% This section initializes parameters related to generators and fuels

def Initialize_Fuel_Specific_Cost(model, g, y):
    """
    Initializes the specific cost of fuel for a generator type and a specific year. The function supports
    importing data from a file or calculating the cost based on a rate of increase.

    Parameters:
    model (object): The model for which the fuel cost is being initialized.
    g (int): The generator type index.
    y (int): The year index.

    Returns:
    float: The specific fuel cost for the given generator type and year.
    """
    if Fuel_Specific_Cost_Calculation == 1 and Fuel_Specific_Cost_Import == 1:  # If dynamic fuel cost is enabled and data should be imported from a file
        delimiters = [(',', '.'), (';', ','), (';', '.')]  # List of (delimiter, decimal) combinations to try
        loaded_successfully = False  # Flag tracking whether the file was loaded successfully
        for delimiter, decimal in delimiters:  # Try each combination in order
            try:
                # Set index_col to 0 if the first column is the index
                fuel_cost_data = pd.read_csv(res_file_path, delimiter=delimiter, decimal=decimal, header=0)  # Attempt to load fuel cost CSV
                print(f"Diesel Fuel prices loaded exogenously using delimiter '{delimiter}' and decimal '{decimal}'")  # Log successful load
                loaded_successfully = True  # Mark as successfully loaded
                break  # Stop trying further combinations
            except pd.errors.ParserError:  # Catch CSV parsing failures
                print(f"Failed to load with delimiter '{delimiter}' and decimal '{decimal}'. Trying next combination...")  # Log and continue
            except FileNotFoundError:  # Catch missing file error
                print(f"File not found: {res_file_path}. Please check the file path and try again.")  # Log file not found
                raise  # Re-raise to halt execution
            except Exception as e:  # Catch other unexpected errors
                print(f"An unexpected error occurred: {e}. Trying next combination...")  # Log and continue
    
        if not loaded_successfully:  # If all combinations failed
            print("Error during import of Fuel_Costs.csv: unable to automatically detect delimiter and decimal. Please try again using delimiter ';' or ',' and decimal ',' or '.'.")  # Log detailed error
            raise ValueError("Failed to load fuel cost data with all provided delimiter and decimal combinations.")  # Raise to halt execution

        # Create a dictionary for fuel costs
        fuel_cost_dict = {(int(gen_type), int(year)): fuel_cost_data.at[year, gen_type]
                      for gen_type in fuel_cost_data.columns
                      for year in fuel_cost_data.index}  # Build a (generator, year) → cost lookup dictionary from the imported file
        return fuel_cost_dict[(g, y)]  # Return the fuel cost for the requested generator and year
    elif Fuel_Specific_Cost_Calculation == 1 and Fuel_Specific_Cost_Import == 0:  # If dynamic fuel cost is enabled but should be calculated from start cost and escalation rate
        years = range(1, n_years+1)  # Create a range of year indices
        fuel_cost_dict = {}  # Initialize dictionary to store computed fuel costs
        for gen_type in range(n_generators):  # Iterate over each generator type index
            previous_cost = Fuel_Specific_Start_Cost[gen_type]  # Set initial cost from parsed parameters
            for year in years:  # Iterate over each year
                if year == 1:  # First year uses the base cost directly
                    cost = previous_cost  # No escalation applied in the first year
                else: cost = previous_cost * (1 + Fuel_Specific_Cost_Rate[gen_type])  # Apply annual escalation rate to the previous year's cost
                fuel_cost_dict[(gen_type + 1, year)] = cost  # Store computed cost with 1-based generator index
                previous_cost = cost  # Update previous cost for the next year's calculation
        return fuel_cost_dict[(g, y)]  # Return the computed fuel cost for the requested generator and year
        
def Initialize_Fuel_Specific_Cost_1(model, g):
    """
    Initializes the starting specific cost of fuel for a generator type. This function is used when
    the fuel cost is fixed and not subject to annual variation.

    Parameters:
    model (object): The model for which the fuel cost is being initialized.
    g (int): The generator type index.

    Returns:
    float: The starting specific fuel cost for the given generator type.
    """
    return model.Fuel_Specific_Start_Cost[g]  # Return the fixed start fuel cost for the specified generator type

def Initialize_Generator_Marginal_Cost(model,g,y):
    """
    Initializes the marginal cost of operation for a generator type considering fuel cost, heating value of the fuel,
    and generator efficiency. This function is applicable when the fuel cost varies annually.

    Parameters:
    model (object): The model for which the marginal cost is being initialized.
    g (int): The generator type index.
    y (int): The year index.

    Returns:
    float: The marginal cost of operation for the specified generator type and year.
    """
    if Fuel_Specific_Cost_Calculation == 1:  # Only compute if dynamic fuel cost mode is active
        return model.Fuel_Specific_Cost[g,y]/(model.Fuel_LHV[g]*model.Generator_Efficiency[g])  # Divide fuel cost by (LHV × efficiency) to get cost per unit of electrical output
    else: None  # Return None when dynamic fuel cost mode is not used

def Initialize_Generator_Marginal_Cost_1(model,g):
    """
    Calculates the marginal cost of operation for a generator type with a fixed fuel cost,
    taking into account the heating value of the fuel and generator efficiency.

    Parameters:
    model (object): The model for which the marginal cost is being initialized.
    g (int): The generator type index.

    Returns:
    float: The marginal cost of operation for the specified generator type.
    """
    if Fuel_Specific_Cost_Calculation == 0:  # Only compute if fixed (non-dynamic) fuel cost mode is active
        return model.Fuel_Specific_Cost_1[g]/(model.Fuel_LHV[g]*model.Generator_Efficiency[g])  # Divide fixed fuel cost by (LHV × efficiency) to get cost per unit of electrical output
    else: None  # Return None when fixed fuel cost mode is not used

def Initialize_Generator_Start_Cost(model,g,y):
    """
    Initializes the start-up cost of a generator for a given year. This cost is based on the generator's marginal cost,
    nominal capacity, and a part-load operation parameter. Applicable in dynamic fuel cost scenarios and MILP formulation.

    Parameters:
    model (object): The model for which the start-up cost is being initialized.
    g (int): The generator type index.
    y (int): The year index.

    Returns:
    float: The start-up cost for the specified generator type and year.
    """
    if Fuel_Specific_Cost_Calculation == 1 and MILP_Formulation == 1:  # Only compute when dynamic fuel cost and MILP formulation are both enabled
        return model.Generator_Marginal_Cost[g,y]*model.Generator_Nominal_Capacity_milp[g]*model.Generator_pgen[g]  # Compute start-up cost as marginal cost × nominal capacity × part-load fraction
    else: None  # Return None when conditions for start-up cost calculation are not met

def Initialize_Generator_Start_Cost_1(model,g):
    """
    Initializes the start-up cost of a generator in a scenario with fixed fuel cost and considering MILP formulation.
    This cost is derived from the generator's marginal cost, nominal capacity, and part-load operation parameter.

    Parameters:
    model (object): The model for which the start-up cost is being initialized.
    g (int): The generator type index.

    Returns:
    float: The start-up cost for the specified generator type.
    """
    if Fuel_Specific_Cost_Calculation == 0 and MILP_Formulation == 1 and Generator_Partial_Load == 1:  # Only compute when fixed fuel cost, MILP, and partial load are all active
        return model.Generator_Marginal_Cost_1[g]*model.Generator_Nominal_Capacity_milp[g]*model.Generator_pgen[g]  # Compute start-up cost using fixed marginal cost × nominal capacity × part-load fraction
    else: None  # Return None when conditions are not met

def Initialize_Generator_Marginal_Cost_milp(model,g,y):
    """
    Calculates the marginal cost for a generator under MILP formulation for dynamic fuel cost scenarios. 
    This cost accounts for the generator's operational characteristics, including nominal capacity and start-up costs.

    Parameters:
    model (object): The model for which the marginal cost is being initialized.
    g (int): The generator type index.
    y (int): The year index.

    Returns:
    float: The marginal cost for the specified generator type and year under MILP formulation.
    """
    if Fuel_Specific_Cost_Calculation == 1 and MILP_Formulation == 1:  # Only compute when dynamic fuel cost and MILP are both active
        return ((model.Generator_Marginal_Cost[g,y]*model.Generator_Nominal_Capacity_milp[g])-model.Generator_Start_Cost[g,y])/model.Generator_Nominal_Capacity_milp[g]  # Subtract start-up cost from total cost and normalise by nominal capacity to get MILP marginal cost
    else: None  # Return None when conditions are not met

def Initialize_Generator_Marginal_Cost_milp_1(model,g):
    """
    Determines the marginal cost for a generator in scenarios with fixed fuel costs and MILP formulation. 
    This cost calculation considers the generator's nominal capacity and start-up cost.

    Parameters:
    model (object): The model for which the marginal cost is being initialized.
    g (int): The generator type index.

    Returns:
    float: The marginal cost for the specified generator type under MILP formulation with fixed fuel costs.
    """
    if Fuel_Specific_Cost_Calculation == 0 and MILP_Formulation == 1 and Generator_Partial_Load == 1:  # Only compute when fixed fuel cost, MILP, and partial load are all active
        return ((model.Generator_Marginal_Cost_1[g]*model.Generator_Nominal_Capacity_milp[g])-model.Generator_Start_Cost_1[g])/model.Generator_Nominal_Capacity_milp[g]  # Subtract fixed start-up cost from total and normalise by nominal capacity for MILP marginal cost
    else: None  # Return None when conditions are not met
    
#%% This section initializes parameters related to grid connection

# Reading grid availability data
if Grid_Connection == 1:  # Only process grid data if a grid connection is specified
    if Grid_Availability_Simulation:  # If stochastic grid availability simulation is enabled
        grid_avail(average_n_outages, average_outage_duration, n_years, year_grid_connection, n_scenarios, n_periods)  # Simulate grid outages and write availability data to file
        availability = pd.read_csv(grid_file_path, delimiter=';', header=0)  # Load the freshly simulated grid availability CSV
    else:  # Use pre-existing deterministic grid availability data
        availability = pd.read_csv(grid_file_path, delimiter=';', header=0)  # Load grid availability CSV from file

    # Create grid_availability Series
    grid_availability_Series = pd.Series()  # Initialize empty Series for accumulating grid availability values
    for i in range(1, n_years * n_scenarios + 1):  # Iterate over each (year, scenario) column in the availability file
        dum = availability[str(i)]  # Extract the column for this (year, scenario) index
        grid_availability_Series = pd.concat([s for s in [grid_availability_Series, dum] if not s.empty])  # Append non-empty availability data to the series

    grid_availability = pd.DataFrame(grid_availability_Series)  # Convert the concatenated Series into a DataFrame

    # Create a MultiIndex
    frame = [scenario, year, period]  # Define axes for the multi-level index
    index = pd.MultiIndex.from_product(frame, names=['scenario', 'year', 'period'])  # Build a Cartesian-product MultiIndex for (scenario, year, period)
    grid_availability.index = index  # Assign the MultiIndex to the grid availability DataFrame

    # Normalize the column name to 0 for consistency
    if grid_availability.columns[0] != 0:  # Check if the first column name is not already 0
        grid_availability.columns = [0]  # Rename the column to 0 for consistent downstream access

    # Create grid_availability_2 DataFrame
    grid_availability_2 = pd.DataFrame()  # Initialize an empty DataFrame for scenario-keyed availability
    for s in scenario:  # Iterate over each scenario
        grid_availability_Series_2 = pd.Series()  # Initialize empty Series to accumulate availability for this scenario
        for y in year:  # Iterate over each year within the scenario
            if Grid_Connection:  # If grid connection is active, use the string column index
                dum_2 = availability[str((s - 1) * n_years + y)]  # Compute and retrieve the string column name for this (scenario, year)
            else:  # Fallback for no grid connection (handles edge cases)
                dum_2 = availability[(s - 1) * n_years + y]  # Retrieve column by integer index when no grid connection
            grid_availability_Series_2 = pd.concat([s for s in [grid_availability_Series_2, dum_2] if not s.empty])  # Append non-empty availability data
        grid_availability_2[s] = grid_availability_Series_2  # Store this scenario's availability as a named column

    # Create a RangeIndex
    index_2 = pd.RangeIndex(1, n_years * n_periods + 1)  # Create 1-based integer index spanning all year×period combinations
    grid_availability_2.index = index_2  # Assign range index to the scenario-keyed availability DataFrame

def Initialize_Grid_Availability(model, s, y, t): 
    """
    Initializes the grid availability based on the specified scenario, year, and time period.

    Parameters:
    model (object): The model for which the grid availability is being initialized.
    s (int): The scenario number.
    y (int): The year.
    t (int): The time period.

    Returns:
    float: The grid availability for the specified scenario, year, and time period.
    """
    if Grid_Connection:  # Only look up availability if there is a grid connection
        try:
            return float(grid_availability[list(grid_availability.columns)[0]][(s, y, t)])  # Retrieve and return grid availability for the (scenario, year, period) key
        except KeyError:  # Handle missing keys gracefully (e.g., period outside grid connection range)
            return 0  # Return 0 (grid unavailable) when the key is not found
    else:  # No grid connection at all
        return 0  # Always return 0 availability when there is no grid connection

def Initialize_National_Grid_Inv_Cost(model):
    """
    Calculates the initial investment cost of connecting to the national grid,
    considering the distance to the grid, specific connection cost, and discount rate.

    Parameters:
    model (object): The model for which the grid investment cost is being initialized.

    Returns:
    float: The total investment cost for connecting to the national grid.
    """
    if Grid_Connection: return model.Grid_Distance*model.Grid_Connection_Cost* model.Grid_Connection/((1+model.Discount_Rate)**(model.Year_Grid_Connection-1))  # Compute NPV of grid connection investment cost discounted to the year of connection
    else: 0  # Return nothing (implicitly None) when there is no grid connection
    
def Initialize_National_Grid_OM_Cost(model):
    """
    Calculates the operation and maintenance cost for the national grid connection,
    accounting for the specific connection cost, grid maintenance cost, and discount rate over the project lifetime.

    Parameters:
    model (object): The model for which the grid O&M cost is being initialized.

    Returns:
    float: The total operation and maintenance cost for the national grid connection.
    """
    Grid_OM_Cost = (model.Grid_Connection_Cost * model.Grid_Connection * model.Grid_Distance) * model.Grid_Maintenance_Cost  # Compute annual O&M cost as connection cost × maintenance rate
    Grid_Fixed_Cost = pd.DataFrame()  # Initialize empty DataFrame to accumulate discounted fixed costs
    g_fc = 0  # Initialize cumulative discounted O&M cost accumulator

    for y in model.year:  # Assuming 'year' is a member of 'model'
        if y < model.Year_Grid_Connection[None]:  # Before grid connection year, no O&M cost is incurred
            g_fc += (0) / ((1 + model.Discount_Rate) ** (y))  # Add zero cost for pre-connection years (discounted)
        else:  # From the grid connection year onward, O&M cost is active
            g_fc += (Grid_OM_Cost) / ((1 + model.Discount_Rate) ** (y))  # Discount annual O&M cost to present value and accumulate

    grid_fc = pd.DataFrame({'Total': g_fc}, index=pd.MultiIndex.from_tuples([("Fixed cost", "National Grid", "-", "kUSD")]))  # Wrap accumulated cost in a labelled DataFrame with a descriptive MultiIndex
    grid_fc.index.names = ['Cost item', 'Component', 'Scenario', 'Unit']  # Assign meaningful names to the MultiIndex levels

    Grid_Fixed_Cost = pd.concat([Grid_Fixed_Cost, grid_fc], axis=1).fillna(0)  # Append the grid fixed cost to the accumulating DataFrame and fill missing values with 0
    Grid_Fixed_Cost = Grid_Fixed_Cost.T.groupby(level=[0], sort=False).sum().T  # Transpose, group by cost item, sum, then transpose back to aggregate costs
    if Grid_Connection: return Grid_Fixed_Cost.iloc[0]['Total']  # Return the total discounted O&M cost if grid-connected
    else: 0  # Return nothing (implicitly None) when there is no grid connection
