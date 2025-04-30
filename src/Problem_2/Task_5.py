import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import math

def DCOPF_model_multiple_gens_and_loads_emissions(N, L, D, G, PGmax, C, demands, U, linecap, susceptance, location_g, location_d, emissions, S_base):
    model = pyo.ConcreteModel()

    # Define sets
    model.n = pyo.Set(initialize=N)   # Nodes
    model.l = pyo.Set(initialize=L)   # Lines
    model.d = pyo.Set(initialize=D)   # Laster
    model.g = pyo.Set(initialize=G)   # Generatorer

    model.PGmax = pyo.Param(model.g, initialize=PGmax,  within=pyo.NonNegativeReals)
    model.C = pyo.Param(model.g, initialize=C, within=pyo.NonNegativeReals)
    model.Demands = pyo.Param(model.d, initialize=demands,  within=pyo.NonNegativeReals)
    model.U = pyo.Param(model.d, initialize=U)
    model.location_g = pyo.Param(model.g, initialize=location_g, within=pyo.Reals)
    model.location_d = pyo.Param(model.d, initialize=location_d, within=pyo.Reals)

    model.linecap = pyo.Param(model.l, initialize=linecap, within=pyo.NonNegativeReals)
    model.suseptance = pyo.Param(model.l, initialize=susceptance)
    model.emissions = pyo.Param(model.g, initialize=emissions)

    # Defining Varaibles
    model.p_G = pyo.Var(model.g, within=pyo.NonNegativeReals)   # Generation per generator
    model.delta = pyo.Var(model.n)                              # Spenningsvinkel
    model.p_D = pyo.Var(model.d, within=pyo.NonNegativeReals)   # Demand per demand point
    model.flow = pyo.Var(model.l, within=pyo.Reals)             # Flow in lines
    model.E = pyo.Var(model.g, within=pyo.NonNegativeReals)     # Emission per generator
    
    def emissions_rule(model, g):
        return model.E[g] == model.p_G[g] * model.emissions[g]
    model.emissions_rule = pyo.Constraint(model.g, rule=emissions_rule)

    def emissions_limit_rule(model, g):
        return model.E[g] <= 100000000
    model.emissions_limit = pyo.Constraint(model.g, rule=emissions_limit_rule)

    def CES(model, g):
        return model.p_G[g] >= 0.2 * sum(model.p_D[d] for d in model.d) if (model.emissions[g] == 0) else pyo.Constraint.Skip  
    model.CES = pyo.Constraint(model.g, rule=CES)   

    def objective_rule(model):
        utility = sum(model.U[d] * model.p_D[d] for d in model.d if not math.isnan(model.U[d]))
        cost_prod = - sum(model.C[g] * model.p_G[g] for g in model.g)
        return   utility + cost_prod
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    # Generator capacity constriant
    def generator_capacity_rule(model, g):
        return  model.p_G[g] <= model.PGmax[g]
    model.generator_capacity = pyo.Constraint(model.g, rule=generator_capacity_rule)

    ## Task 2-4b and 2-4c
    # Demand constraints
    def demand_rule(model, d):
        return model.p_D[d] <= model.Demands[d]
    model.demand = pyo.Constraint(model.d, rule=demand_rule)

    def power_balance(model, n):
        flows = {
            1: model.flow[1] + model.flow[2],
            2: -model.flow[1] + model.flow[3],
            3: -model.flow[2] - model.flow[3]
        }
        ## Task 2-4b and 2-4c
        return sum(model.p_G[g] for g in model.g if model.location_g[g] == n) - sum(model.p_D[d] for d in model.d if model.location_d[d] ==n) == flows[n]
    model.balance = pyo.Constraint(model.n, rule=power_balance)

    # line capacity constraints
    def line_capacity_rule(model, l):
        return model.flow[l] <= model.linecap[l]
    model.line_capacity = pyo.Constraint(model.l, rule=line_capacity_rule)

    def line_capacity_lower_rule(model, l):
        return -model.linecap[l] <= model.flow[l]
    model.line_capacity_lower = pyo.Constraint(model.l, rule=line_capacity_lower_rule)

    def flow_rule(model, l):
        if l == 1:
            return model.flow[l] == model.suseptance[l] * (model.delta[1] - model.delta[2])
        elif l == 3:
            return model.flow[l] == model.suseptance[l] * (model.delta[2] - model.delta[3])
        elif l == 2:
            return model.flow[l] == model.suseptance[l] * (model.delta[1] - model.delta[3])
    model.flow_rule = pyo.Constraint(model.l, rule=flow_rule)

    # Inelastic Load constraint
    def inelastic_load(model, d):
        if math.isnan(model.U[d]):
            return model.p_D[d] == model.Demands[d]
        else:
            return pyo.Constraint.Skip
    model.inelastic_load = pyo.Constraint(model.d,rule=inelastic_load)

    # Reference angle constraint
    def ref_angle(model):
        return model.delta[1] == 0
    model.ref_angle = pyo.Constraint(rule=ref_angle)

    solver = SolverFactory("gurobi")    # Ensure Gurobi is installed and licenced
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT) # For dual variables
    results = solver.solve(model, tee=True)

    print(f"{'='*10} Optimal Solution {'='*10}")
    print(f"Social Welfare: {model.objective():.2f} NOK")
    
    # Extract and display the optimal generation for each generator
    print("\nOptimal generation [MW]:")
    for g in model.g:
        print(f"P_{g}^(gen): {S_base * model.p_G[g].value:.2f}")
    
    # Extract and display the optimal load for each load
    print("\nOptimal Load [MW]:")
    for d in model.d:
        print(f"P_{d}^(d): {S_base * model.p_D[d].value:.2f}")
    
    # Extract and display voltage angles for each node
    print("\nVoltage angles [rad]:") 
    for n in model.n:
        print(f"Delta {n}: {model.delta[n].value:.4f}")

    # Nodal price calculation
    print("\nNodal prices [NOK/MWh]:")
    for n in model.n:
        dual_value = model.dual.get(model.balance[n])
        if dual_value is not None:
            print(f"Electricity Price at node {n}: {dual_value/S_base:.2f}")
        else:
            print(f"Node {g}: No dual value found.")
    

    # Extract and display dual values for generator capacity constraints 
    print("\nDual values for generator capacity constraints [NOK/MWh]:")
    for g in model.g:
        dual_value = model.dual.get(model.generator_capacity[g])
        if dual_value is not None:
            print(f"Generator {g}: {dual_value/S_base:.2f}")
        else:
            print(f"Generator {g}: No dual value found.")

    # Extract and display dual values for line capacity constraints
    print("\nDual values for line capacity constraints [NOK/MWh]:")
    for l in model.l:
        dual_value = model.dual.get(model.line_capacity[l])
        if dual_value is not None:
            print(f"Line {l}: {dual_value/S_base:.2f}")
        else:
            print(f"Line {l}: No dual value found.")

    # Extract and display power flow in lines
    print("\nFlow in lines [MW]:")
    for l in model.l:
        print(f"Line {l}: {pyo.value(S_base*model.flow[l]):.2f}")

    # Extract and display voltage angles
    print("\nVoltage angles [rad]:")
    for n in model.n:
        print(f"Node {n}: {pyo.value(model.delta[n]):.4f}")
