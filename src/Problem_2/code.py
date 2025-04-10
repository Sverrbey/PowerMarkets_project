import pyomo.environ as pyo
from pyomo.opt import SolverFactory

def main():
    # Informasjon
    
    # Generator data
    gen1 = {'Generator': 'G1', 'Capacity': 1000, 'Marginal cost': 300, 'Location': 'Node 1', 'Slack bus': True}
    gen2 = {'Generator': 'G2', 'Capacity': 1000, 'Marginal cost': 1000, 'Location': 'Node 2', 'Slack bus': False}
    gen3 = {'Generator': 'G3', 'Capacity': 1000, 'Marginal cost': 600, 'Location': 'Node 3', 'Slack bus': False}   # Endre til 1000 på oppgave c
 

    # Bruk en liste med generator-dictionaryer
    generators = [gen1, gen2, gen3]
    
    # Transmission line data
    line12 = {'Line': 'L12', 'Capacity': 500, 'Substation': -20}
    line13 = {'Line': 'L13', 'Capacity': 500, 'Substation': -10}
    line23 = {'Line': 'L23', 'Capacity': 100, 'Substation': -30}

    linjer = [line12, line13, line23]

    # Load data
    load1 = {'Load': 'Load1', 'Demand': 200, 'Location': 'Node 1'}
    load2 = {'Load': 'Load2', 'Demand': 200, 'Location': 'Node 2'}
    load3 = {'Load': 'Load3', 'Demand': 500, 'Location': 'Node 3'}

    last = [load1, load2, load3]

    # Lag lister for generatorer, linjer og last

    gen_cap = {i+1: gen['Capacity'] for i, gen in enumerate(generators)}
    gen_cost = {i+1: gen['Marginal cost'] for i, gen in enumerate(generators)}
    demand = {i+1: ld['Demand'] for i, ld in enumerate(last)}
    linecap = {le['Line']: le['Capacity'] for le in linjer}

    

    # Create a model
    model = pyo.ConcreteModel()
    # Define sets
    model.N = pyo.Set(initialize=[1, 2, 3])                 # Noder
    model.L = pyo.Set(initialize=['L12', 'L13', 'L23'])     # Linjer

    # Variabler
    model.theta = pyo.Var(model.N)                          # Spenningsvinkel
    model.gen = pyo.Var(model.N, bounds=(0, None))          # Produksjon
    model.flow = pyo.Var(model.L)                           # Flyten i linjene

    # Parameter
    model.Demand = pyo.Param(model.N, initialize=demand)
    model.GenCap = pyo.Param(model.N, initialize=gen_cap)
    model.GenCost = pyo.Param(model.N, initialize=gen_cost)
    model.LineCap = pyo.Param(model.L, initialize=linecap)
    model.B = pyo.Param(model.L, initialize={'L12': -20, 'L13': -10, 'L23': -30})

    # Objective function - minimere produksjonskostnadene
    def obj_rule(model):
        return sum(model.gen[n] * model.GenCost[n] for n in model.N)
    model.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Begrensinger
    def power_balance(model, n):
        flows = {
            1: model.flow['L12'] + model.flow['L13'],
            2: -model.flow['L12'] + model.flow['L23'],
            3: -model.flow['L13'] - model.flow['L23']
        }
        return model.gen[n] - model.Demand[n] == flows[n]
    model.power_balance_const = pyo.Constraint(model.N, rule=power_balance)

    def gen_capacity(model, n):
        return model.gen[n] <= model.GenCap[n]
    model.gen_capacity_const = pyo.Constraint(model.N, rule=gen_capacity)

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
    print("\n=== DCOPF Resultater ===")
    
    print("\nProduksjon (MW):")
    for n in model.N:
        print(f"Node {n}: {pyo.value(model.gen[n]):.2f}")
    
    print("\nNode vinkel (radians):")
    for n in model.N:
        print(f"Node {n}: {pyo.value(model.theta[n]):.2f}")
    
    print("\nFlyt i linjene (MW):")
    for l in model.L:
        print(f"Line {l}: {pyo.value(model.flow[l]):.2f}")
    
    print("\nNodal Prices (NOK/MWh):")
    for n in model.N:
        # Her henter vi dualverdier fra kraftbalansebetingelsen
        print(f"Node {n}: {model.dual[model.power_balance_const[n]]:.2f}")
    
    print("\nBegrensninger i linjene:")
    for l in model.L:
        flow = abs(pyo.value(model.flow[l]))
        if abs(flow - model.LineCap[l]) < 1e-4:
            print(f"Line {l} is congested at {flow:.2f} MW")
            
    print(f"\nTotal System Cost: {pyo.value(model.OBJ):.2f} NOK")

if __name__ == "__main__":
    main()