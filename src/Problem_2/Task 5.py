import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt


def main():
    # ----------------------------- data ----------------------------- #
    # Generator‑data
    gens = [
        {"Gen": "Gen 1_1", "Capacity": 300,  "MarginalCost": 200,   "CO2": 1500,  "Node": 1},
        {"Gen": "Gen 1_2", "Capacity": 400,  "MarginalCost": 300,   "CO2": 700,   "Node": 1},
        {"Gen": "Gen 1_3", "Capacity": 300,  "MarginalCost": 800,   "CO2": 100,   "Node": 1},
        {"Gen": "Gen 2",   "Capacity": 1000, "MarginalCost": 1000,  "CO2": 0,     "Node": 2},
        {"Gen": "Gen 3",   "Capacity": 1000, "MarginalCost": 600,   "CO2": 1000,  "Node": 3},
    ]   

    generators = [i+1 for i, gen in enumerate(gens)]
    location_g = {i+1: gen['Node'] for i, gen in enumerate(gens)}
    emissions = {i+1: gen['CO2'] for i, gen in enumerate(gens)}
    

    # Load-data (max demand and willingness-to-pay)
    load_values = [
        {"Load": "Load1",   "Demand": 200, "WTP": None, "Node": 1},  # uelastisk
        {"Load": "Load2_1", "Demand": 200, "WTP": 1300, "Node": 2},
        {"Load": "Load2_2", "Demand": 250, "WTP":  800, "Node": 2},
        {"Load": "Load2_3", "Demand": 250, "WTP":  500, "Node": 2},
        {"Load": "Load3",   "Demand": 500, "WTP": None, "Node": 3},  # uelastisk
    ]

    loads = [i+1 for i, load in enumerate(load_values)]
    location_d = {i+1: load['Node'] for i, load in enumerate(load_values)}
    utl = {i+1: load['WTP'] if load['WTP'] is not None else 0 for i, load in enumerate(load_values)}
     # Transmission line data
    linjer = [
        {'Line': 'L12', 'Capacity': 500, 'Substation': -20},
        {'Line': 'L13', 'Capacity': 500, 'Substation': -10},
        {'Line': 'L23', 'Capacity': 100, 'Substation': -30}
    ]

    # Prepare dictionaries
    gen_cap = {i+1: gen['Capacity'] for i, gen in enumerate(gens)}  
    gen_cost = {i+1: gen['MarginalCost'] for i, gen in enumerate(gens)}
    demand = {i+1: ld['Demand'] for i, ld in enumerate(load_values)}
    linecap = {le['Line']: le['Capacity'] for le in linjer}
    substation = {le['Line']: le['Substation'] for le in linjer}

    DCOPF_model_multiple_gens_and_loads(generators,loads,
                                        gen_cap,
                                        gen_cost,
                                        demand,
                                        utl,
                                        linecap,
                                        substation,
                                        location_g,
                                        location_d,
                                        emissions)

    # Create a model
def DCOPF_model_multiple_gens_and_loads(generators,loads, PGmax, C, D, U, linecap, substation, location_g, location_d, emissions):
    model = pyo.ConcreteModel()

    # Define sets
    model.n = pyo.Set(initialize=[1, 2, 3])                 # Noder
    model.l = pyo.Set(initialize=['L12', 'L13', 'L23'])     # Linjer
    model.d = pyo.Set(initialize=loads)                     # Laster
    model.g = pyo.Set(initialize=generators)                # Generatorer

    model.PGmax = pyo.Param(model.g, initialize=PGmax)
    model.C = pyo.Param(model.g, initialize=C)
    model.D = pyo.Param(model.d, initialize=D)
    model.U = pyo.Param(model.d, initialize=U)
    model.location_g = pyo.Param(model.g, initialize=location_g)
    model.location_d = pyo.Param(model.d, initialize=location_d)
    model.emissions = pyo.Param(model.g, initialize=emissions)

    model.linecap = pyo.Param(model.l, initialize=linecap)
    model.substation = pyo.Param(model.l, initialize=substation)
    # Defining Varaibles
    model.p_G = pyo.Var(model.g, within=pyo.NonNegativeReals)   # Generation per generator
    model.theta = pyo.Var(model.n)                              # Spenningsvinkel
    model.p_D = pyo.Var(model.d, within=pyo.NonNegativeReals)   # Demand per demand point
    model.flow = pyo.Var(model.l, within=pyo.Reals)             # Flow in lines


    # We maximize the social welfare using the following objective function
    #def objective_rule(model):
    #    return  sum(model.C[g] * model.p_G[g] for g in model.g)
    #model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    ## Task 2-4c
    def objective_rule(model):
        utility = sum(model.U[d] * model.p_D[d] for d in model.d)
        cost_prod = - sum(model.C[g] * model.p_G[g] for g in model.g)
        return   utility + cost_prod
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Generator capacity constriant
    def generator_capacity_rule(model, g):
        return  model.p_G[g] <= model.PGmax[g]
    model.generator_capacity = pyo.Constraint(model.g, rule=generator_capacity_rule)

    def emissions_rule(model, g):
    #    return model.p_G[g] >= 0.2 * sum(model.p_D[d] for d in model.d) if (model.emissions[g] == 0) else pyo.Constraint.Skip  #Kommenter ut denne til senere
        return model.p_G[g] * model.emissions[g] >= 1000000000
    model.emissions_rule = pyo.Constraint(model.g, rule=emissions_rule)

    # Demand constraints
    def demand_rule(model, d):
        return model.p_D[d] <= model.D[d]
    model.demand = pyo.Constraint(model.d, rule=demand_rule)

    def power_balance(model, n):
        flows = {
            1: model.flow['L12'] + model.flow['L13'],
            2: -model.flow['L12'] + model.flow['L23'],
            3: -model.flow['L13'] - model.flow['L23']
        }
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

    #plotting
    plt.bar([str(g) for g in model.g], [model.p_G[g].value for g in model.g], color='red')
    plt.xlabel('Generators')
    plt.ylabel('Generation (MW)')
    plt.title('Generation by each generator (Task 2-4)')
    plt.show()

    #Extract and display dual values for demand capacity constraints
    
main()