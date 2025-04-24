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
    load1 = {'Load': 'Load1', 'Demand': 200, 'Location': 1}
    load2 = {'Load': 'Load2', 'Demand': 200, 'Location': 2}
    load3 = {'Load': 'Load3', 'Demand': 500, 'Location': 3}

    last = [load1, load2, load3]

    gen_cap = {i+1: gen['Capacity'] for i, gen in enumerate(generators)}
    gen_cost = {i+1: gen['Marginal cost'] for i, gen in enumerate(generators)}
    demand = {i+1: ld['Demand'] for i, ld in enumerate(last)}
    linecap = {le['Line']: le['Capacity'] for le in linjer}
    substation = {le['Line']: le['Substation'] for le in linjer}

    DCOPF_model(gen_cap, gen_cost, demand, linecap, substation)

    # Create a model
def DCOPF_model(PGmax, C, D, linecap, substation):
    model = pyo.ConcreteModel()

    # Define sets
    model.n = pyo.Set(initialize=[1, 2, 3])                 # Noder
    model.l = pyo.Set(initialize=['L12', 'L13', 'L23'])     # Linjer
    model.d = pyo.Set(initialize=[1, 2, 3])                 # Laster
    model.g = pyo.Set(initialize=[1, 2, 3])                 # Generatorer

    model.PGmax = pyo.Param(model.g, initialize=PGmax)
    model.C = pyo.Param(model.g, initialize=C)
    model.D = pyo.Param(model.d, initialize=D)
    
    model.linecap = pyo.Param(model.l, initialize=linecap)
    model.substation = pyo.Param(model.l, initialize=substation)
    # Defining Varaibles
    model.p_G = pyo.Var(model.g, within=pyo.NonNegativeReals)   # Generation per generator
    model.theta = pyo.Var(model.n)                            # Spenningsvinkel
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

    def power_balance(model, g):
        flows = {
            1: model.flow['L12'] + model.flow['L13'],
            2: -model.flow['L12'] + model.flow['L23'],
            3: -model.flow['L13'] - model.flow['L23']
        }

        return model.p_G[g] - model.D[g] == flows[g]
    model.balance = pyo.Constraint(model.g, rule=power_balance)

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
    for g in model.g:
        dual_value = model.dual.get(model.balance[g])
        if dual_value is not None:
            print(f"Electricity Price at node {g}: {dual_value}")
        else:
            print(f"Node {g}: No dual value found.")

    # Extract and display dual values for generator capacity constraints 
    print("Dual values for generator capacity constraints:")
    for g in model.g:
        dual_value = model.dual.get(model.generator_capacity[g])
        if dual_value is not None:
            print(f"Generator {g}: {dual_value}")
        else:
            print(f"Generator {g}: No dual value found.")

    for l in model.l:
        dual_value = model.dual.get(model.flow_rule[l])
        if dual_value is not None:
            print(f"Line {l}: {dual_value}")
        else:
            print(f"Line {l}: No dual value found.")

    #Extract and display dual values for demand capacity constraints
    


main()

