# -*- coding: utf-8 -*-
import pandas as pd
from src.Problem_2.Task_2 import DCOPF_model_task_2 
from src.Problem_2.Task_3 import DCOPF_model_task_3

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
    suseptance = dict(zip(L_data.index, L_data["Susceptance [pu]"]))

    # 2.2a)
    DCOPF_model_task_2(N, L, D, G, PGmax, C, demands, linecap, suseptance, S_base)
    
    # 2.2c)
    C[3] = 1000000
    #DCOPF_model_task_2(N, L, D, G, PGmax, C, demands, linecap, suseptance, S_base)

    ## Task 2-3: Creating the optimization model
    # Extracting the Generator data
    G_data      = pd.read_csv("data/Problem 2.3 - Generators/Generator Data.csv").set_index("Generator", drop=True)
    G           = G_data.index.tolist()
    PGmax       = dict(zip(G_data.index, G_data["Capacity [pu]"]))
    C           = dict(zip(G_data.index, G_data["Marginal cost [NOK/puh]"]))
    location    = dict(zip(G_data.index, G_data["Location"]))

    #DCOPF_model_task_3(N, L, D, G, PGmax, C, demands, linecap, suseptance, location, S_base)
    
if __name__ == "__main__":
    main()