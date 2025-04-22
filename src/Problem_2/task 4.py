import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

def main():
    # Informasjon
    
    # Generator data
    gens = [
        {'Gen': 'Gen 1_1', 'Capacity': 300, 'MarginalCost': 200, 'Location': 1, 'Slack': True},
        {'Gen': 'Gen 1_2', 'Capacity': 400, 'MarginalCost': 300, 'Location': 1, 'Slack': True},
        {'Gen': 'Gen 1_3', 'Capacity': 300, 'MarginalCost': 800, 'Location': 1, 'Slack': True},
        {'Gen': 'Gen 2',   'Capacity': 1000, 'MarginalCost': 1000, 'Location': 2, 'Slack': False},
        {'Gen': 'Gen 3',   'Capacity': 1000, 'MarginalCost': 600, 'Location': 3, 'Slack': False}
    ]

    # Bruk en liste med generator-dictionaryer

    
    # Transmission line data
    line12 = {'Line': 'L12', 'Capacity': 500, 'Substation': -20}
    line13 = {'Line': 'L13', 'Capacity': 500, 'Substation': -10}
    line23 = {'Line': 'L23', 'Capacity': 100, 'Substation': -30}

    linjer = [line12, line13, line23]

    # Load data
    load1 = {'Load': 'Load1', 'Demand': 200, 'Location': 1}
    load2 = {'Load': 'Load2', 'Demand': 200, 'Location': 2}
    load3 = {'Load': 'Load3', 'Demand': 500, 'Location': 3}

    last = [load1, load2, load3]

    # Lag lister for generatorer, linjer og last

    gen_capacity = {g['Gen']: g['Capacity'] for g in gens}
    gen_cost = {g['Gen']: g['MarginalCost'] for g in gens}
    gen_node = {g['Gen']: g['Location'] for g in gens}


    demand = {ld['Location']: ld['Demand'] for ld in last}
    linecap = {le['Line']: le['Capacity'] for le in linjer}
    substation = {le['Line']: le['Substation'] for le in linjer}
    

    # Create a model
    model = pyo.ConcreteModel()
    # Define sets
    model.N = pyo.Set(initialize=[1, 2, 3])                 # Noder
    model.L = pyo.Set(initialize=['L12', 'L13', 'L23'])     # Linjer
    model.G = pyo.Set(initialize=list(gen_capacity.keys())) 

    # Variabler
    model.theta = pyo.Var(model.N)                          # Spenningsvinkel
    model.gen = pyo.Var(model.G, bounds=(0, None))          # Produksjon
    model.flow = pyo.Var(model.L)                           # Flyten i linjene

    # Parameter
    model.GenCapacity = pyo.Param(model.G, initialize=gen_capacity)
    model.GenCost = pyo.Param(model.G, initialize=gen_cost)
    model.GenNode = pyo.Param(model.G, initialize=gen_node)
    model.Demand = pyo.Param(model.N, initialize=demand)
    model.LineCap = pyo.Param(model.L, initialize=linecap)
    model.B = pyo.Param(model.L, initialize=substation)

    # Objective function - minimere produksjonskostnadene
    def obj_rule(model):
        return sum(model.gen[g] * model.GenCost[g] for g in model.G)
    model.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Begrensinger
    def power_balance_rule(model, n):
        prod = sum(model.gen[g] for g in model.G if model.GenNode[g] == n)

        flows = {
            1: model.flow['L12'] + model.flow['L13'],
            2: -model.flow['L12'] + model.flow['L23'],
            3: -model.flow['L13'] - model.flow['L23']
        }
        return prod - model.Demand[n] == flows[n]
    model.power_balance_const = pyo.Constraint(model.N, rule=power_balance_rule)

    def gen_capacity_rule(model, g):
        return model.gen[g] <= model.GenCapacity[g]
    model.gen_capacity_const = pyo.Constraint(model.G, rule=gen_capacity_rule)

    def line_flow_rule(model, l):
        if l == 'L12':
            return model.flow[l] == model.B[l] * (model.theta[1] - model.theta[2])
        elif l == 'L13':
            return model.flow[l] == model.B[l] * (model.theta[1] - model.theta[3])
        else:  # L23
            return model.flow[l] == model.B[l] * (model.theta[2] - model.theta[3])
    model.line_flow_const = pyo.Constraint(model.L, rule=line_flow_rule)

    def line_capacity_upper(model, l):
        return model.flow[l] <= model.LineCap[l]
    model.line_capacity_upper = pyo.Constraint(model.L, rule=line_capacity_upper)

    def line_capacity_lower(model, l):
        return -model.LineCap[l] <= model.flow[l]
    model.line_capacity_lower = pyo.Constraint(model.L, rule=line_capacity_lower)

    def ref_angle(model):
        return model.theta[1] == 0
    model.ref_angle_const = pyo.Constraint(rule=ref_angle)

    # Solve the model
    opt = SolverFactory('gurobi')
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(model, tee=True)
    
    # Skriv ut resultater
    print("\n=== DCOPF Resultater: Multiple Generators ===")
    print("\nProduksjon per generator (MW):")
    production = {}
    for g in model.G:
        production[g] = pyo.value(model.gen[g])
        print(f"{g} (Node {model.GenNode[g]}): {production[g]:.2f} MW")
    
    print("\nNode spenningsvinkler (radians):")
    for n in model.N:
        print(f"Node {n}: {pyo.value(model.theta[n]):.2f}")
    
    print("\nLinjeflyt (MW):")
    for l in model.L:
        print(f"Line {l}: {pyo.value(model.flow[l]):.2f}")
    
    print("\nNodal Prices (NOK/MWh):")
    for n in model.N:
        dual_vale = model.dual[model.power_balance_const[n]]
        print(f"Node {n}: {dual_vale:.2f} NOK/MWh")
    
    print(f"\nTotal System Cost: {pyo.value(model.OBJ):.2f} NOK")

    gen_names = list(production.keys())
    prod_values = [production[g] for g in gen_names]
    
    plt.figure(figsize=(8, 4))
    plt.bar(gen_names, prod_values)
    plt.xlabel("Generator")
    plt.ylabel("Produksjon (MW)")
    plt.title("Produksjon per generator")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()