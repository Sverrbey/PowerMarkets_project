# -*- coding: utf-8 -*-
import pandas as pd
from src.Problem_2.Task_2 import DCOPF_model
from src.Problem_2.Task_3 import DCOPF_model_multiple_generators
from src.Problem_2.Task_4 import DCOPF_model_multiple_generators_and_loads, DCOPF_model_multiple_generators_and_loads_SW
from src.Problem_2.Task_5 import DCOPF_model_multiple_gens_and_loads_emissions

def main():
    """ Methodology to run the project problems
    """

    N = [1,2,3] # Nodes
    S_base = 1000 # Base power in MVA

    ## Task 2-2: Creating the optimization model
    # Extracting the Generator data
    G_data      = pd.read_csv("data/Problem 2.2 - Base case/Generator Data.csv").set_index("Generator", drop=True) 
    G           = G_data.index.tolist()
    PGmax       = dict(zip(G_data.index, G_data["Capacity [pu]"]))
    C           = dict(zip(G_data.index, G_data["Marginal cost [NOK/puh]"]))
    # Extracting the Load data
    D_data      = pd.read_csv("data/Problem 2.2 - Base case/Load data.csv").set_index("Load unit", drop=True)
    D           = D_data.index.tolist()
    demands     = dict(zip(D_data.index, D_data["Demand [pu]"]))
    # Extracting the Transmission line data
    L_data     = pd.read_csv("data/Problem 2.2 - Base case/Transmission line data.csv").set_index("Line", drop=True)
    L          = L_data.index.tolist()
    linecap    = dict(zip(L_data.index, L_data["Capacity [pu]"]))
    susceptance = dict(zip(L_data.index, L_data["Susceptance [pu]"]))

    # 2.2a)
    #DCOPF_model(N, L, D, G, PGmax, C, demands, linecap, susceptance, S_base)
    
    # 2.2c) 
    C[3] = 1000000  # [NOK/pu.h]
    #DCOPF_model(N, L, D, G, PGmax, C, demands, linecap, susceptance, S_base)

    ## Task 2-3: Extending the model: Multiple generators
    # Extracting the Generator data
    G_data      = pd.read_csv("data/Problem 2.3 - Generators/Generator Data.csv").set_index("Generator", drop=True)
    G           = G_data.index.tolist()
    PGmax       = dict(zip(G_data.index, G_data["Capacity [pu]"]))
    C           = dict(zip(G_data.index, G_data["Marginal cost [NOK/puh]"]))
    location_g    = dict(zip(G_data.index, G_data["Location"]))

    # 2.3 
    #DCOPF_model_multiple_generators(N, L, D, G, PGmax, C, demands, linecap, susceptance, location_g, S_base)
    
    
    ## Task 2-4: Extending the model: Multiple loads
    # Extracting the Load data
    D_data      = pd.read_csv("data/Problem 2.4 - Loads/Load data.csv").set_index("Load unit", drop=True)
    D           = D_data.index.tolist()
    demands     = dict(zip(D_data.index, D_data["Demand [pu]"]))
    location_d = dict(zip(D_data.index, D_data["Location"]))
    U = dict(zip(D_data.index, D_data["Marginal cost [NOK/puh]"]))
    
    # 2.4a) Inelastic loads
    #DCOPF_model_multiple_generators_and_loads(N, L, D, G, PGmax, C, demands, U, linecap, susceptance, location_g, location_d, S_base)

    # 2.4c) Elastic loads
    #DCOPF_model_multiple_generators_and_loads_SW(N, L, D, G, PGmax, C, demands, U, linecap, susceptance, location_g, location_d, S_base)


    ## Task 2-5: Extending the model: Environmental constraints
    # Extracting the Generator data
    G_data      = pd.read_csv("data/Problem 2.5 - Environmental/Generator Data.csv").set_index("Generator", drop=True)
    G           = G_data.index.tolist()
    PGmax       = dict(zip(G_data.index, G_data["Capacity [pu]"]))
    C           = dict(zip(G_data.index, G_data["Marginal cost [NOK/puh]"]))
    location_g    = dict(zip(G_data.index, G_data["Location"]))
    emissions = dict(zip(G_data.index, G_data["CO2 emission [kg/puh]"]))

    # 2.5 B) CES constraint
    DCOPF_model_multiple_gens_and_loads_emissions(N, L, D, G, PGmax, C, demands, U, linecap, susceptance, location_g, location_d, emissions, S_base)
    


if __name__ == "__main__":
    main()