import numpy as np
import sys
import time
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

def Main():

    """ Main function that set up, execute, and store results """


def Read_Excel(name):
    """
    Reads input excel file and reads the data into dataframes.
    Separates between each sheet, and stores into one dictionary
    """

    generators_df = pd.read_excel(name, sheet_name="Generators")
    demands_df = pd.read_excel(name, sheet_name="Demands")

def Preparing_Data(generators_df, demands_df):
    
    """
    Setting up matrices for the problems:
        - B-matrix  -> Admittance matrix for the lines. Used in DCOPF
        - DC-matrix -> Bus incidence matrix for the DC cables. Used in both DCOPF and ATC
        - X-matrix  -> Bus incidence matrix for AC cables. Used in ATC
    """
    generators = generators_df["Generators"].tolist()
    PGmax = dict(zip(generators_df["Generator"], generators_df["PGmax"]))
    C = dict(zip(generators_df["Generator"], generators_df["Cost"]))

    demands = demands_df["Demand"].tolist()
    L = dict(zip(demands_df["Demand"], demands_df["L"]))
    U = dict(zip(demands_df["DEmand"], demands_df["U"]))


def DCOPF_model(Data):
    
    """
    Set up the optimization model, run it and store the data in a .xlsx file
    """

    model = pyo.ConcreteModel()

    # Define sets for generators and demands
    model.g = pyo.Set(initialize=generators)
    model.d = pyo.Set(initialize=demands)

    # Define parameters using the dictionaries
    model.PGmax = pyo.Param(model.g, initialize=PGmax)
    model.C = pyo.Param(model.g, initialize=C)
    model.L = pyo.Param(model.d, initialize=L)
    model.U = pyo.Param(model.d, initialize=U)


    # Defining Varaibles
    model.p_G = pyo.Var(model.g, within=pyo.NonNegativeReals)   # Generation per generator
    model.p_D = pyo.Var(model.d, within=pyo.NonNegativeReals)   # Demand per demand point


    # We maximize the social welfare using the following objective function
    def objective_rule(model):
        return  sum(model.U[d] * model.p_D[d] for d in model.d) - \
                sum(model.C[g] * model.p_G[g] for g in model.g)
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Generator capacity constriant
    def generator_capacity_rule(model, g):
        return  model.p_G[g] <= model.PGmax[g]
    model.generator_capacity = pyo.Constraint(model.g, rule=generator_capacity_rule)

    # Demand capacity constraint 
    def demand_capacity_rule(model, d):
        return  model.p_D[d] <= model.L[d]
    model.demand_capacity = pyo.Constraint(model.d, rule=demand_capacity_rule)

    # Power balance constraint
    def balance_rule(model):
        return  sum(model.p_D[d] for d in model.d) == sum(model.p_G[g] for g in model.g)
    model.balance = pyo.Constraint(rule=balance_rule)

    solver = SolverFactory("gurobi")    # Ensure Gurobi is installed and licenced
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT) # For dual variables
    results = solver.solve(model, tee=True)

    print(f"{'='*10} Optimal Solution {'='*10}")
    print("Social Welfare (SW):", model.objective())
    for g in model.g:
        print(f"Generation by {g}: {model.p_G[g].value} MW")
    for d in model.d:
        print(f"Demand by {d}: {model.p_D[d].value} MW")

    # Market clearing price from the power balance constraint
    if model.balance in model.dual:
        print("Market Clearing Price of Electricity:", model.dual[model.balance])
    else:
        print("No dual value found for the balance constraint.")

    # Extract and display dual values for generator capacity constraints 
    print("Dual values for generator capacity constraints:")
    for g in model.g:
        dual_value = model.dual.get(model.generator_capacity[g])
        if dual_value is not None:
            print(f"Generator {g}: {dual_value}")
        else:
            print(f"Generator {g}: No dual value found.")

    #Extract and display dual values for demand capacity constraints
    print("Dual values for demand capacity constraints:")
    for d in model. d:
        dual_value = model.dual.get(model.demand_capacity[d])
        if dual_value is not None:
            print(f"Demand {d}: {dual_value}")
        else:
            print("Demand {d}: No dual value found.")

 
def Store_model_data(model,Data):
    
    """
    Stores the results from the optimization model run into an excel file
    """

#Main()