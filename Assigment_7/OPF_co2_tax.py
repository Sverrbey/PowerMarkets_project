
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

def main():

    
    producer = pd.read_csv('Assigment_7/data/supply.txt', sep= ',', header=0, names=['Technology','ID','Pg','Cg','Area'], index_col='Technology')
    consumer = pd.read_csv('Assigment_7/data/demand.txt', sep= ',', header=0, names=['Company', 'ID', 'Pd', 'Ud', 'Area'], index_col='ID')
    emission = pd.read_csv('Assigment_7/data/emission.txt', sep= ',', header=0, names=['Technology','Emission_intensity'], index_col='Technology')
    

    generators = producer.index.tolist()
    PGmax = dict(zip(producer.index, producer["Pg"]))
    C = dict(zip(producer.index, producer["Cg"]))
    E = dict(zip(emission.index, emission['Emission_intensity'])) # Emission intensity
    prod = dict(zip(producer.index, producer["Area"]))
    
    demands = consumer.index.tolist()
    L = dict(zip(consumer.index, consumer["Pd"])) # Load demand
    U = dict(zip(consumer.index, consumer["Ud"])) # Utility cost
    dema = dict(zip(consumer.index, consumer["Area"]))

    DCOPF_model(generators, PGmax, C, demands, L, U, E, prod, dema)

    return

# Now make your model here: (and feel free to delete the 'print' tests!)
def DCOPF_model(generators, PGmax, C, demands, L, U, E, prod, dema):
    
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
    model.E = pyo.Param(model.g, initialize=E) # Added emission intensity parameter
    model.L = pyo.Param(model.d, initialize=L)
    model.U = pyo.Param(model.d, initialize=U)
    model.prod = pyo.Param(model.g, initialize=prod) # Added area parameter
    model.dema = pyo.Param(model.d, initialize=dema) # Added area parameter

    # Transmission capacity between areas A and B
    model.transmission_capacity = 40
    model.transmission_capacity = pyo.Param(initialize=model.transmission_capacity)

    # Defining Varaibles
    model.p_G = pyo.Var(model.g, within=pyo.NonNegativeReals)   # Generation per generator
    model.p_D = pyo.Var(model.d, within=pyo.NonNegativeReals)   # Demand per demand point
    model.flow_AB = pyo.Var(within=pyo.Reals)  # Power flow between areas A and B


    # Define areas
    area_A_generators = [g for g in generators if prod[g] == 'A']  # Generators in Area A
    area_A_consumers  = [d for d in demands if dema[d] == 'A']     # Consumers in Area A
    area_B_generators = [g for g in generators if prod[g] == 'B']  # Generators in Area B
    area_B_consumers  = [d for d in demands if dema[d] == 'B']     # Consumers in Area B

    # Objective function: Maximize social welfare for both areas
    def objective_rule(model):
        # Social welfare = Utility from demand - Cost of generation
        welfare_A = sum(model.U[d] * model.p_D[d] for d in area_A_consumers) - \
                    sum(model.C[g] * model.p_G[g] for g in area_A_generators)
        welfare_B = sum(model.U[d] * model.p_D[d] for d in area_B_consumers) - \
                    sum(model.C[g] * model.p_G[g] for g in area_B_generators)
        return welfare_A + welfare_B
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Generator capacity constriant
    def generator_capacity_rule(model, g):
        return  model.p_G[g] <= model.PGmax[g]
    model.generator_capacity = pyo.Constraint(model.g, rule=generator_capacity_rule)

    def emission_constraint_rule(model, g):
        return model.E[g]*1e-3 * model.p_G[g] <= 200
    model.emission_constraint = pyo.Constraint(model.g, rule=emission_constraint_rule)


    # Demand capacity constraint 
    def demand_capacity_rule(model, d):
        return  model.p_D[d] <= model.L[d]
    model.demand_capacity = pyo.Constraint(model.d, rule=demand_capacity_rule)

    # Power balance constraint
    def balance_rule_A(model):
        # Power balance for Area A and Area B
        balance_A = sum(model.p_G[g] for g in area_A_generators) - sum(model.p_D[d] for d in area_A_consumers) - model.flow_AB
        # Both areas must balance independently
        return balance_A == 0
    model.balance_A = pyo.Constraint(rule=balance_rule_A)

    # Power balance constraint
    def balance_rule_B(model):
        # Power balance for Area A and Area B
        balance_B = sum(model.p_G[g] for g in area_B_generators) - sum(model.p_D[d] for d in area_B_consumers) + model.flow_AB
        # Both areas must balance independently
        return balance_B == 0
    model.balance_B = pyo.Constraint(rule=balance_rule_B)

    # Transmission capacity constraint
    def transmission_constraint_rule_lower(model):
        return -model.transmission_capacity <= model.flow_AB
    model.transmission_constraint_lower = pyo.Constraint(rule=transmission_constraint_rule_lower)
    
    def transmission_constraint_rule_upper(model):
        return model.flow_AB <= model.transmission_capacity
    model.transmission_constraint_upper = pyo.Constraint(rule=transmission_constraint_rule_upper)
    
    # Power flow equation
    def flow_AB_rule(model):
        # Flow from Area A to Area B is the net generation in Area A minus the net consumption in Area A
        return model.flow_AB == (
            sum(model.p_G[g] for g in area_A_generators) - sum(model.p_D[d] for d in area_A_consumers)
        )
    model.flow_AB_equation = pyo.Constraint(rule=flow_AB_rule)

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
    if model.balance_A in model.dual:
        print("Market Clearing Price of Electricity in area A:", model.dual[model.balance_A])
    else:
        print("No dual value found for the balance constraint.")

    # Market clearing price from the power balance constraint
    if model.balance_B in model.dual:
        print("Market Clearing Price of Electricity in area B:", model.dual[model.balance_A])
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

    #Extract and display dual values for total emission constraints
    print("Dual values for emission capacity constraints:")
    for g in model. g:
        dual_value = model.dual.get(model.emission_constraint[g])
        if dual_value is not None:
            print(f"Emission {g}: {dual_value}")
        else:
            print("Emission {g}: No dual value found.")

    print("C_new: Dual values for generator capacity constraints:")
    for g in model.g:
        c_old = model.dual.get(model.generator_capacity[g])
        carbon_tax = model.dual.get(model.emission_constraint[g])
        c_new = c_old + model.E[g]*carbon_tax
        if dual_value is not None:
            print(f"Generator {g}: {c_new}")
        else:
            print(f"Generator {g}: No dual value found.")
main()


