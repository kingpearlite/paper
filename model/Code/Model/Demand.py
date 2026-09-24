import re, time, pandas as pd, numpy as np  # re for text parsing, pandas/numpy for data handling
import os  # os for filesystem paths
import params_path  # Resolves which parameters file this run reads (MGPY_PARAMS)

def data_import(data_demand):  # Parses the raw lines of Parameters.dat and extracts demand-related settings
    for value in data_demand:  # Iterate over each line of the .dat file
        if "param: lat" in value:  # Locate the latitude parameter line
            lat = (value[value.index('=')+1:value.index(';')])  # Extract text between '=' and ';'
            numbers = re.compile('-?\d+')  # Regex to match an (optionally negative) integer
            lat = list(map(int, numbers.findall(lat)))[0]  # Parse the latitude value as an integer
            if  10 <= lat <=20:
                F = 'F1'  # Climate zone F1: latitude between 10 and 20
            elif -10 <= lat < 10:
                F = 'F2'  # Climate zone F2: latitude between -10 and 10
            elif -20 <= lat < -10:
                F = 'F3'  # Climate zone F3: latitude between -20 and -10
            elif -30<= lat < -20:
                F = 'F4'  # Climate zone F4: latitude between -30 and -20
            elif lat < -30:
                F = 'F5'  # Climate zone F5: latitude below -30
        if "param: cooling_period" in value:
            cooling_period = value[value.index('=')+1:value.index(';')].replace(' ','').replace("'","")  # Cooling period label (e.g. season/method)
        if "param: h_tier1" in value:
            h_tier1 = float(value[value.index('=')+1:value.index(';')].replace(' ','').replace("'",""))  # % of households in wealth tier 1
        if "param: h_tier2" in value:
            h_tier2 = float(value[value.index('=')+1:value.index(';')].replace(' ','').replace("'",""))  # % of households in wealth tier 2
        if "param: h_tier3" in value:
            h_tier3 = float(value[value.index('=')+1:value.index(';')].replace(' ','').replace("'",""))  # % of households in wealth tier 3
        if "param: h_tier4" in value:
            h_tier4 = float(value[value.index('=')+1:value.index(';')].replace(' ','').replace("'",""))  # % of households in wealth tier 4
        if "param: h_tier5" in value:
            h_tier5 = float(value[value.index('=')+1:value.index(';')].replace(' ','').replace("'",""))  # % of households in wealth tier 5
        if "param: schools" in value:
            num_schools = float(value[value.index('=')+1:value.index(';')].replace(' ','').replace("'",""))  # Number of schools
        if "param: hospital_1" in value:
            num_hosp_1 = float(value[value.index('=')+1:value.index(';')].replace(' ','').replace("'",""))  # Number of hospitals of tier 1
        if "param: hospital_2" in value:
            num_hosp_2 = float(value[value.index('=')+1:value.index(';')].replace(' ','').replace("'",""))  # Number of hospitals of tier 2
        if "param: hospital_3" in value:
            num_hosp_3 = float(value[value.index('=')+1:value.index(';')].replace(' ','').replace("'",""))  # Number of hospitals of tier 3
        if "param: hospital_4" in value:
            num_hosp_4 = float(value[value.index('=')+1:value.index(';')].replace(' ','').replace("'",""))  # Number of hospitals of tier 4
        if "param: hospital_5" in value:
            num_hosp_5 = float(value[value.index('=')+1:value.index(';')].replace(' ','').replace("'",""))  # Number of hospitals of tier 5
        if "param: demand_growth" in value:
            demand_growth = float(value[value.index('=')+1:value.index(';')].replace(' ','').replace("'",""))       # Yearly demand growth rate (%)
        if "param: Years" in value:
            years = int(value[value.index('=')+1:value.index(';')].replace(' ','').replace("'",""))  # Number of years to project the demand over
        if "param: Periods" in value:
            periods = int(value[value.index('=')+1:value.index(';')].replace(' ','').replace("'",""))  # Number of time periods per year (temporal resolution)
    
    return F, cooling_period, [h_tier1, h_tier2, h_tier3, h_tier4, h_tier5], [num_hosp_1, num_hosp_2, num_hosp_3, num_hosp_4, num_hosp_5,num_schools], demand_growth, years, periods  # Return all parsed parameters

#%% Calculates the load demand given as input the latitude, cooling period and number of households for each wealth tier and number of services (schools and hospitals)

def aggregate_load(load_data, periods):  # Downsamples an hourly load series into the requested number of periods
    # Number of hours in the dataset
    total_hours = len(load_data)  # Total number of hourly rows in the input series

    # Calculate aggregation factor
    agg_factor = total_hours // periods  # How many consecutive hours are summed into one period

    # Aggregate data
    aggregated_load = load_data.groupby(load_data.index // agg_factor).sum()  # Group every agg_factor rows together and sum them

    return aggregated_load

def demand_calculation():  # Builds the full multi-year household + services load demand profile
    
    current_directory = os.path.dirname(os.path.abspath(__file__))  # Directory containing this script
    inputs_directory = os.path.join(current_directory, '..', 'Inputs')  # Path to the Inputs folder
    data_file_path = params_path.PARAMS_PATH  # Resolved once via MGPY_PARAMS; see params_path.py
    data_demand = open(data_file_path).readlines()  # Read all lines of the parameters file
    
    num_h_tier = []
    
    F, cooling_period, num_h_tier, num_services, demand_growth, years, periods = data_import(data_demand)  # Parse all needed parameters
    
    class household:  # Represents a group of households of a given wealth tier
      def __init__(self, zone, wealth, cooling, number):
        self.zone = zone  # Climate zone (F1-F5)
        self.wealth = wealth  # Wealth tier index
        self.cooling = cooling  # Cooling period/method label
        self.number = number  # Percentage/number of households in this tier
    
      def load_demand(self, h_load):
          load = self.number/100 * h_load  # Scale archetype load by the share of households in this tier
          return load
      
    class service:  # Represents a group of service facilities (hospitals/schools)
        def __init__(self, structure, number):
            self.structure = structure  # Structure/tier index
            self.number = number  # Number of facilities of this type
        
        def load_demand(self, h_load):
            load = self.number * h_load  # Scale archetype load by the number of facilities
            return load
        
    households = []
    load_households = []  
    demand_archetypes_path = os.path.join(current_directory, '..', 'Demand_archetypes')  # Folder with load archetype Excel files

    for ii in range(1, len(num_h_tier) + 1):  # Loop over each household wealth tier
        households.append(household(F, ii, cooling_period, num_h_tier[ii - 1]))  # Create a household object for this tier
        h_load_name = households[ii - 1].cooling + '_' + F + '_Tier-' + str(ii)  # Build archetype filename from cooling, zone and tier
        file_path = os.path.join(demand_archetypes_path, h_load_name + ".xlsx")
        h_load = pd.DataFrame(pd.read_excel(file_path, skiprows=0, usecols="B"))  # Read hourly archetype load from column B
        h_load = aggregate_load(h_load, periods)  # Aggregate to the model's time resolution
        load_households.append(household.load_demand(households[ii - 1], h_load))  # Scale by number of households and store
        
    load_households = pd.concat([sum(load_households)]*years, axis = 1, ignore_index = True)  # Sum all tiers, then repeat the profile for every year
            
    #%% Load demand of services    
    services = []
    load_tot_services = []
    for ii in range(1, len(num_services) + 1):  # Loop over each service type (hospitals tiers 1-5, then schools)
        services.append(service(ii, num_services[ii-1]))
        if ii < 6:
            service_load_name = "HOSPITAL_Tier-" + str(ii)  # Hospital archetype file for this tier
        else: 
            service_load_name = "SCHOOL"  # School archetype file
        file_path = os.path.join(demand_archetypes_path, service_load_name + ".xlsx")
        service_load = pd.DataFrame(pd.read_excel(file_path, skiprows=0, usecols="B"))  # Read hourly archetype load from column B
        service_load = aggregate_load(service_load, periods)  # Aggregate service load data
        load_tot_services.append(services[ii-1].load_demand(service_load))  # Scale by number of facilities and store
    
    load_tot_services = pd.concat([sum(load_tot_services)]*years, axis = 1, ignore_index = True)  # Sum all service types, then repeat for every year
    
    # Total load demand (households + services)   
    
    load_total = load_tot_services + load_households  # Combine household and service demand for each year
    for column in load_total:  # Iterate over year columns (0-indexed)
        if column == 0:
            continue  # First year keeps the base demand as-is
        else: 
            load_total[column] = load_total[column-1]*(1+demand_growth/100)    # yearly demand growth 

    return load_total, years  # Return the full multi-year load demand and number of years
    
    #%% Export results to excel
def excel_export(load,years):  # Writes the computed load demand DataFrame to Demand.csv
    # Setting new column names based on the number of years
    load = load.set_axis(np.arange(1, years+1), axis=1)  # Rename columns to year numbers 1..years
    
    current_directory = os.path.dirname(os.path.abspath(__file__))
    inputs_directory = os.path.join(current_directory, '..', 'Inputs')
    demand_file_path = os.path.join(inputs_directory, 'Demand.csv')  # Output path for the demand CSV
    # Exporting the DataFrame to a CSV file
    load.to_csv(demand_file_path, sep=';', decimal=',', index=True)  # Save using ';' delimiter and ',' decimal separator


#%% Calculates and export the load demand  time series of households and services for 20 years to Demand.xlsx

def demand_generation():  # Entry point: computes demand, exports it, and reports timing
    start = time.time()  # Start timer
        
    print("Load demand calculation started, please remember to close Demand.xlsx... \n")
    load_tot, years = demand_calculation()  # Compute the multi-year load demand
    excel_export(load_tot,years)  # Export the result to Demand.csv
    
    
    end = time.time()  # End timer
    elapsed = end - start  # Elapsed computation time in seconds
    load_tot = load_tot.set_axis(np.arange(1,years+1), axis=1)  # Rename columns to year numbers for the returned DataFrame
    print('\n\nLoad demand calculation completed (overall time: ',round(elapsed,0),'s,', round(elapsed/60,1),' m)\n')
    return load_tot

if __name__ == "__Demand__":  # Guard that never triggers when run as a script (module name would be "__main__")
    demand_calculation()
    demand_generation()

