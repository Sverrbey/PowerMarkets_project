import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd

## Exercise 3
def ImportData(production_data_filename, consumption_data_filename):
    p_data = pd.read_csv(production_data_filename).set_index('Producer', drop=True)
    c_data = pd.read_csv(consumption_data_filename).set_index('Consumer', drop=True)
    return p_data, c_data
p_data, c_data = ImportData('Assignment_3/production_data.csv','Assignment_3/consumption_data.csv')  # Fill these in! Make sure it's in the correct order as defined in the function :)


generators = p_data.index.tolist()
PGmax = dict(zip(p_data.index, p_data["Capacity"]))
C = dict(zip(p_data.index, p_data["Cost"]))

demands = c_data.index.tolist()
L = dict(zip(c_data.index, c_data["Capacity"])) # Load demand
U = dict(zip(c_data.index, c_data["Cost"])) # Utility cost


# Now make your model here: (and feel free to delete the 'print' tests!)
def DCOPF_model(generators, PGmax, C, demands, L, U):
    
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


DCOPF_model(generators, PGmax, C, demands, L, U)