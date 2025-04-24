import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

def main():
    # Generator data
    gens = [
        {'Gen': 'Gen 1_1', 'Capacity': 300, 'MarginalCost': 200, 'Location': 1},
        {'Gen': 'Gen 1_2', 'Capacity': 400, 'MarginalCost': 300, 'Location': 1},
        {'Gen': 'Gen 1_3', 'Capacity': 300, 'MarginalCost': 800, 'Location': 1},
        {'Gen': 'Gen 2',   'Capacity': 1000, 'MarginalCost': 1000, 'Location': 2},
        {'Gen': 'Gen 3',   'Capacity': 1000, 'MarginalCost': 600, 'Location': 3}
    ]

    # Transmission line data
    linjer = [
        {'Line': 'L12', 'Capacity': 500, 'Substation': -20},
        {'Line': 'L13', 'Capacity': 500, 'Substation': -10},
        {'Line': 'L23', 'Capacity': 100, 'Substation': -30}
    ]

    # Load data (max demand and willingness-to-pay)
    last = [
        {'Load': 'Load1',   'Demand': 200, 'MarginalWTP': None, 'Location': 1},
        {'Load': 'Load2_1', 'Demand': 200, 'MarginalWTP': 1300, 'Location': 2},
        {'Load': 'Load2_2', 'Demand': 250, 'MarginalWTP': 800,  'Location': 2},
        {'Load': 'Load2_3', 'Demand': 250, 'MarginalWTP': 500,  'Location': 2},
        {'Load': 'Load3',   'Demand': 500, 'MarginalWTP': None, 'Location': 3}
    ]

    # Prepare dictionaries
    gen_capacity = {g['Gen']: g['Capacity'] for g in gens}
    gen_cost     = {g['Gen']: g['MarginalCost'] for g in gens}
    gen_node     = {g['Gen']: g['Location']    for g in gens}

    load_ids     = [ld['Load'] for ld in last]
    load_max     = {ld['Load']: ld['Demand']               for ld in last}
    load_wtp     = {ld['Load']: ld['MarginalWTP'] or 0      for ld in last}
    load_node    = {ld['Load']: ld['Location']             for ld in last}

    linecap      = {l['Line']: l['Capacity']   for l in linjer}
    substation   = {l['Line']: l['Substation'] for l in linjer}

    # Build model
    model = pyo.ConcreteModel()
    model.N = pyo.Set(initialize=[1,2,3])
    model.L = pyo.Set(initialize=list(linecap.keys()))
    model.G = pyo.Set(initialize=list(gen_capacity.keys()))
    model.D = pyo.Set(initialize=load_ids)

    # Parameters
    model.GenCapacity = pyo.Param(model.G, initialize=gen_capacity)
    model.GenCost     = pyo.Param(model.G, initialize=gen_cost)
    model.GenNode     = pyo.Param(model.G, initialize=gen_node)
    model.LoadMax     = pyo.Param(model.D, initialize=load_max)
    model.WTP         = pyo.Param(model.D, initialize=load_wtp)
    model.LoadNode    = pyo.Param(model.D, initialize=load_node)
    model.LineCap     = pyo.Param(model.L, initialize=linecap)
    model.B           = pyo.Param(model.L, initialize=substation)

    # Variables
    model.gen   = pyo.Var(model.G, bounds=(0,None))
    model.load_var  = pyo.Var(model.D, bounds=lambda m, d: (0, m.LoadMax[d]))
    model.flow  = pyo.Var(model.L)
    model.theta = pyo.Var(model.N)

    # Objective (maximize welfare)
    def obj_rule(m):
        return sum(m.load_var[d] * m.WTP[d] for d in m.D) - sum(m.gen[g] * m.GenCost[g] for g in m.G)
    model.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    # Constraints
    def power_balance(m, n):
        prod = sum(m.gen[g] for g in m.G if m.GenNode[g] == n)
        cons = sum(m.load_var[d] for d in m.D if m.LoadNode[d] == n)
        flows = {
            1: m.flow['L12'] + m.flow['L13'],
            2: -m.flow['L12'] + m.flow['L23'],
            3: -m.flow['L13'] - m.flow['L23']
        }
        return prod - cons == flows[n]
    model.power_balance = pyo.Constraint(model.N, rule=power_balance)

    model.gen_cap = pyo.Constraint(model.G, rule=lambda m,g: m.gen[g] <= m.GenCapacity[g])
    model.line_flow = pyo.Constraint(model.L, rule=lambda m,l: (
        m.flow[l] == m.B[l] * (m.theta[1] - m.theta[2]) if l=='L12' else
        m.flow[l] == m.B[l] * (m.theta[1] - m.theta[3]) if l=='L13' else
        m.flow[l] == m.B[l] * (m.theta[2] - m.theta[3])
    ))
    model.line_cap_up   = pyo.Constraint(model.L, rule=lambda m,l: m.flow[l] <= m.LineCap[l])
    model.line_cap_down = pyo.Constraint(model.L, rule=lambda m,l: -m.LineCap[l] <= m.flow[l])
    model.ref_angle     = pyo.Constraint(rule=lambda m: m.theta[1]==0)

    # Solve with Gurobi persistent interface and import duals
    opt = SolverFactory('gurobi', solver_io='python')
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(model, tee=True)
    opt.close_global()

    # Output
    print("\n=== Resultater ===")
    for n in model.N:
        print(f"Node {n}, pris: {model.dual[model.power_balance[n]]:.2f}")
    for d in model.D:
        print(f"{d} forbruk: {pyo.value(model.load_var[d]):.2f}")

    # Print voltage angles
    print("\nSpenningsvinkler (rad):")
    for n in model.N:
        print(f"theta[{n}]: {pyo.value(model.theta[n]):.4f}")

    # Print line flows
    print("\nLinjeflyt (MW):")
    for l in model.L:
        print(f"{l}: {pyo.value(model.flow[l]):.2f}")

    # Total welfare (objective value)
    total_welfare = pyo.value(model.OBJ)
    print(f"\nTotal velferd (nytte - kostnad): {total_welfare:.2f} NOK")

    # Consumer and producer surplus
    consumer_surplus = 0
    producer_surplus = 0

    print("\nKonsumentoverskudd per last (NOK):")
    for d in model.D:
        node = model.LoadNode[d]
        price = model.dual[model.power_balance[node]]
        x = pyo.value(model.load_var[d])
        wtp = model.WTP[d]
        cs = x * (wtp - price)
        consumer_surplus += cs
        print(f"{d}: {cs:.2f}")

    print("\nProdusentoverskudd per generator (NOK):")
    for g in model.G:
        node = model.GenNode[g]
        price = model.dual[model.power_balance[node]]
        q = pyo.value(model.gen[g])
        cost = model.GenCost[g]
        ps = q * (price - cost)
        producer_surplus += ps
        print(f"{g}: {ps:.2f}")

    print(f"\nTotal konsumentoverskudd: {consumer_surplus:.2f} NOK")
    print(f"Total produsentoverskudd: {producer_surplus:.2f} NOK")

    # Congestion rents per line
    print("\nKapasitetsleie per linje (NOK):")
    total_congestion = 0
    line_nodes = {'L12': (1,2), 'L13': (1,3), 'L23': (2,3)}
    for l in model.L:
        i, j = line_nodes[l]
        pi_i = model.dual[model.power_balance[i]]
        pi_j = model.dual[model.power_balance[j]]
        f = pyo.value(model.flow[l])
        rent = (pi_i - pi_j) * f
        total_congestion += rent
        print(f"{l}: {rent:.2f}")
    print(f"\nTotal kapasitetsleie: {total_congestion:.2f} NOK")
    print("\nBegrensninger i linjene:")
    for l in model.L:
        flow = abs(pyo.value(model.flow[l]))
        if abs(flow - model.LineCap[l]) < 1e-4:
            print(f"Line {l} is congested at {flow:.2f} MW")

    # Plot production
    prod = [pyo.value(model.gen[g]) for g in model.G]
    plt.bar(list(model.G), prod)
    plt.xlabel("Generator")
    plt.ylabel("MW")
    plt.title("Produksjon per generator")
    plt.show()

if __name__ == "__main__":
    main()
