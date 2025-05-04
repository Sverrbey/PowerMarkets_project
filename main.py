# -*- coding: utf-8 -*-
"""
    Course project for TET4185 Power Markets
    (c) Bastian Øie, May 2025
        Sverre Beyer, May 2025
        Aurora Vinslid, May 2025
"""

import pandas as pd
import sys

from src.Problem_2.Task_2 import DCOPF_model
from src.Problem_2.Task_3 import DCOPF_model_multiple_generators
from src.Problem_2.Task_4 import DCOPF_model_multiple_generators_and_loads, DCOPF_model_multiple_generators_and_loads_SW
from src.Problem_2.Task_5 import DCOPF_model_multiple_gens_and_loads_emissions_CES, DCOPF_model_multiple_gens_and_loads_emissions_cap_and_trade

from src.Problem_3.OPF_pyomo import OPF_model, Read_Excel, Create_matrices
from src.Problem_3.OPF_pyomo_CO2 import OPF_model_CO2

def main():
    """ 
    The following code can be used to run the different tasks in the project. See README.md for more information.
    """
    N = [1,2,3] # Nodes
    S_base = 1000 # Base power in MVA

    ### Task 2: ###

    ## Task 2.2: Creating the optimization model
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

    # Task 2.2a)
    #DCOPF_model(N, L, D, G, PGmax, C, demands, linecap, susceptance, S_base)
    
    # Task 2.2c) 
    C[3] = 1000000  # [NOK/pu.h]
    #DCOPF_model(N, L, D, G, PGmax, C, demands, linecap, susceptance, S_base)

    ## Task 2.3: Extending the model: Multiple generators
    # Extracting the Generator data
    G_data      = pd.read_csv("data/Problem 2.3 - Generators/Generator Data.csv").set_index("Generator", drop=True)
    G           = G_data.index.tolist()
    PGmax       = dict(zip(G_data.index, G_data["Capacity [pu]"]))
    C           = dict(zip(G_data.index, G_data["Marginal cost [NOK/puh]"]))
    location_g    = dict(zip(G_data.index, G_data["Location"]))

    # Task 2.3 
    #DCOPF_model_multiple_generators(N, L, D, G, PGmax, C, demands, linecap, susceptance, location_g, S_base)
    
    ## Task 2.4: Extending the model: Multiple loads
    # Extracting the Load data
    D_data      = pd.read_csv("data/Problem 2.4 - Loads/Load data.csv").set_index("Load unit", drop=True)
    D           = D_data.index.tolist()
    demands     = dict(zip(D_data.index, D_data["Demand [pu]"]))
    location_d = dict(zip(D_data.index, D_data["Location"]))
    U = dict(zip(D_data.index, D_data["Marginal cost [NOK/puh]"]))
    
    # 2.4a) Inelastic loads
    #DCOPF_model_multiple_generators_and_loads(N, L, D, G, PGmax, C, demands, U, linecap, susceptance, location_g, location_d, S_base)

    # 2.4c) Objective is maximizing Social Welfare + Elastic and inelastic loads
    #DCOPF_model_multiple_generators_and_loads_SW(N, L, D, G, PGmax, C, demands, U, linecap, susceptance, location_g, location_d, S_base)


    ## Task 2.5: Extending the model: Environmental constraints
    # Extracting the Generator data
    G_data      = pd.read_csv("data/Problem 2.5 - Environmental/Generator Data.csv").set_index("Generator", drop=True)
    G           = G_data.index.tolist()
    PGmax       = dict(zip(G_data.index, G_data["Capacity [pu]"]))
    C           = dict(zip(G_data.index, G_data["Marginal cost [NOK/puh]"]))
    location_g    = dict(zip(G_data.index, G_data["Location"]))
    emissions = dict(zip(G_data.index, G_data["CO2 emission [kg/puh]"]))

    # Task 2.5 B) CES constraint
    #DCOPF_model_multiple_gens_and_loads_emissions_CES(N, L, D, G, PGmax, C, demands, U, linecap, susceptance, location_g, location_d, emissions, S_base)
    
    # Task 2.5 b) Cap-and-trade constraint
    #DCOPF_model_multiple_gens_and_loads_emissions_cap_and_trade(N, L, D, G, PGmax, C, demands, U, linecap, susceptance, location_g, location_d, emissions, S_base)

    ###  Task 3: ###
    # Task 3.2 - Analyzing a wet-year scenario
    Data = Read_Excel("data/Nordic_wet.xlsx")    #Master dictionary, created from input data dictionary
            
    #Create matrices for the lines and cables
    Data = Create_matrices(Data)

    # Task 3.2
    #OPF_model(Data)    # DCOPF
    Data["DCFlow"] = 0
    #OPF_model(Data)    # ATC

    ## Task 3.3 - Dry-year scenario comparison
    Data = Read_Excel("data/Nordic_dry.xlsx")
    Data = Create_matrices(Data)

    #Task 3.3
    #OPF_model(Data)    # DCOPF
    Data["DCFlow"] = 0
    #OPF_model(Data)    # ATC

    ## Task 3.4 Phasing out baseload produciton
    Data = Read_Excel("data/Nordic_wet.xlsx")    #Master dictionary, created from input data dictionary
    Data['Nodes']['GENCAP'][8] -= 8400           # Phasing out baseload production at SE3
    #Create matrices for the lines and cables
    Data = Create_matrices(Data)

    #Task 3.4
    #OPF_model(Data)    # DCOPF
    Data["DCFlow"] = 0
    #OPF_model(Data)    # ATC


    ## Task 3.5 - Emission trading system
    Data = Read_Excel("data/Nordic_wet.xlsx")    #Master dictionary, created from input data dictionary
    
    #Carbon intensity
    Data['Nodes']['EmissionValue'] = {
        1:0.0340,
        2:0.0240,
        3:0.0260,
        4:0.0550,
        5:0.0260,
        6:0.0393,
        7:0.0393,
        8:0.0393,
        9:0.0393,
        10:0.0590,
        11:0.0570,
        12:0.0910
    }
    Data["Emissions_cost"] = 65  #[EUR/tonne CO2eq]
    Data = Create_matrices(Data)
    #Task 3.5
    #OPF_model_CO2(Data) #DCOPF

if __name__ == "__main__":
    main()