import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

def main():
    # Generator data
    gens = [
        {'Gen': 'Gen 1_1', 'Capacity': 300,  'MarginalCost': 200, 'Location': 1},
        {'Gen': 'Gen 1_2', 'Capacity': 400,  'MarginalCost': 300, 'Location': 1},
        {'Gen': 'Gen 1_3', 'Capacity': 300,  'MarginalCost': 800, 'Location': 1},
        {'Gen': 'Gen 2',   'Capacity': 1000, 'MarginalCost': 1000, 'Location': 2},
        {'Gen': 'Gen 3',   'Capacity': 1000, 'MarginalCost': 600, 'Location': 3}
    ]
    generators = [index+1 for index, gen in enumerate(gens)]
    location = {i+1: gen['Location'] for i, gen in enumerate(gens)}

    # Transmission line data
    linjer = [
        {'Line': 'L12', 'Capacity': 500, 'Substation': -20},
        {'Line': 'L13', 'Capacity': 500, 'Substation': -10},
        {'Line': 'L23', 'Capacity': 100, 'Substation': -30}
    ]

    # Load data (max demand and willingness-to-pay)
    last = [
        {'Load': 'Load1',   'Demand': 200, 'Location': 1},
        {'Load': 'Load2',   'Demand': 200, 'Location': 2},
        {'Load': 'Load3',   'Demand': 500, 'Location': 3}
    ]

    # Prepare dictionaries
    gen_cap = {i+1: gen['Capacity'] for i, gen in enumerate(gens)}  
    gen_cost = {i+1: gen['MarginalCost'] for i, gen in enumerate(gens)}
    demand = {i+1: ld['Demand'] for i, ld in enumerate(last)}
    linecap = {le['Line']: le['Capacity'] for le in linjer}
    substation = {le['Line']: le['Substation'] for le in linjer}

    DCOPF_model_multiple_gens(generators, gen_cap, gen_cost, demand, linecap, substation, location)

    # Create a model
def DCOPF_model_multiple_gens(generators, PGmax, C, D, linecap, substation, location_g):
    model = pyo.ConcreteModel()

    # Define sets
    model.n = pyo.Set(initialize=[1, 2, 3])                 # Noder
    model.l = pyo.Set(initialize=['L12', 'L13', 'L23'])     # Linjer
    model.d = pyo.Set(initialize=[1, 2, 3])                 # Laster
    model.g = pyo.Set(initialize=generators)                 # Generatorer

    model.PGmax = pyo.Param(model.g, initialize=PGmax)
    model.C = pyo.Param(model.g, initialize=C)
    model.D = pyo.Param(model.d, initialize=D)
    model.location_g = pyo.Param(model.g, initialize=location_g)


    model.linecap = pyo.Param(model.l, initialize=linecap)
    model.substation = pyo.Param(model.l, initialize=substation)
    # Defining Varaibles
    model.p_G = pyo.Var(model.g, within=pyo.NonNegativeReals)   # Generation per generator
    model.theta = pyo.Var(model.n)                              # Spenningsvinkel
    model.p_D = pyo.Var(model.d, within=pyo.NonNegativeReals)   # Demand per demand point
    model.flow = pyo.Var(model.l, within=pyo.Reals)             # Flow in lines


    # We maximize the social welfare using the following objective function
    def objective_rule(model):
        return  sum(model.C[g] * model.p_G[g] for g in model.g)
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Generator capacity constriant
    def generator_capacity_rule(model, g):
        return  model.p_G[g] <= model.PGmax[g]
    model.generator_capacity = pyo.Constraint(model.g, rule=generator_capacity_rule)

    def power_balance(model, n):
        flows = {
            1: model.flow['L12'] + model.flow['L13'],
            2: -model.flow['L12'] + model.flow['L23'],
            3: -model.flow['L13'] - model.flow['L23']
        }
        return sum(model.p_G[g] for g in model.g if model.location_g[g] == n) - model.D[n] == flows[n]
    model.balance = pyo.Constraint(model.n, rule=power_balance)


    # line capacity constraints
    def line_capacity_rule(model, l):
        return model.flow[l] <= model.linecap[l]
    model.line_capacity = pyo.Constraint(model.l, rule=line_capacity_rule)

    def line_capacity_lower_rule(model, l):
        return -model.linecap[l] <= model.flow[l]
    model.line_capacity_lower = pyo.Constraint(model.l, rule=line_capacity_lower_rule)

    def flow_rule(model, l):
        if l == 'L12':
            return model.flow[l] == model.substation[l] * (model.theta[1] - model.theta[2])
        elif l == 'L23':
            return model.flow[l] == model.substation[l] * (model.theta[2] - model.theta[3])
        elif l == 'L13':
            return model.flow[l] == model.substation[l] * (model.theta[1] - model.theta[3])
    model.flow_rule = pyo.Constraint(model.l, rule=flow_rule)

    # Reference angle constraint
    def ref_angle(model):
        return model.theta[1] == 0
    model.ref_angle = pyo.Constraint(rule=ref_angle)

    solver = SolverFactory("gurobi")    # Ensure Gurobi is installed and licenced
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT) # For dual variables
    results = solver.solve(model, tee=True)

    print(f"{'='*10} Optimal Solution {'='*10}")
    print("Social Welfare (SW):", model.objective())
    for g in model.g:
        print(f"Generation by {g}: {model.p_G[g].value} MW")

    # Market clearing price from the power balance constraint
    for n in model.n:
        dual_value = model.dual.get(model.balance[n])
        if dual_value is not None:
            print(f"Electricity Price at node {n}: {dual_value}")
        else:
            print(f"Node {n}: No dual value found.")
    
    print("\nGenerator cost")
    # prin generator cost
    for g in model.g:
        print(f"Generator {g} cost: {model.C[g]}")

    # # Extract and display dual values for generator capacity constraints 
    # print("Dual values for generator capacity constraints:")
    # for g in model.g:
    #     dual_value = model.dual.get(model.generator_capacity[g])
    #     if dual_value is not None:
    #         print(f"Generator {g}: {dual_value}")
    #     else:
    #         print(f"Generator {g}: No dual value found.")


    # for l in model.l:
    #     dual_value = model.dual.get(model.flow_rule[l])
    #     if dual_value is not None:
    #         print(f"Line {l}: {dual_value}")
    #     else:
    #         print(f"Line {l}: No dual value found.")

    #flyt i linjene
    print("\nFlow in lines (MW):")
    for l in model.l:
        print(f"{l}: {pyo.value(model.flow[l]):.2f}")
    # Print voltage angles
    print("\nVoltage angles (radians):")
    for n in model.n:
        print(f"Node {n}: {pyo.value(model.theta[n]):.2f}")

    #Extract and display dual values for demand capacity constraints
    


main()

