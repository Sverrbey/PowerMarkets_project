import pyomo.environ as pyo
from pyomo.opt import SolverFactory

def DCOPF_model(N, L, D, G, PGmax, C, demands, linecap, suseptance, S_base):
    model = pyo.ConcreteModel()

    # Define sets
    model.n = pyo.Set(initialize=N)   # Noder
    model.l = pyo.Set(initialize=L)   # Linjer
    model.d = pyo.Set(initialize=D)   # Laster
    model.g = pyo.Set(initialize=G)   # Generatorer

    model.PGmax     = pyo.Param(model.g, initialize=PGmax)
    model.C         = pyo.Param(model.g, initialize=C)
    model.Demands   = pyo.Param(model.d, initialize=demands)
    
    model.linecap       = pyo.Param(model.l, initialize=linecap, within=pyo.NonNegativeReals)
    model.suseptance    = pyo.Param(model.l, initialize=suseptance)

    # Defining Varaibles
    model.p_G   = pyo.Var(model.g, within=pyo.NonNegativeReals)     # Generation per generator
    model.delta = pyo.Var(model.n)                                  # Spenningsvinkel
    model.p_D   = pyo.Var(model.d, within=pyo.NonNegativeReals)     # Demand per demand point
    model.flow  = pyo.Var(model.l, within=pyo.Reals)                # Flow in lines


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
            1: model.flow[1] + model.flow[2],
            2: -model.flow[1] + model.flow[3],
            3: -model.flow[2] - model.flow[3]
        }
        return model.p_G[g] - model.Demands[g] == flows[g]
    model.balance = pyo.Constraint(model.g, rule=power_balance)

    # Line capacity constraints
    def line_capacity_rule(model, l):
        return (-model.linecap[l], model.flow[l], model.linecap[l])
    model.line_capacity = pyo.Constraint(model.l, rule=line_capacity_rule)

    def flow_rule(model, l):
        if l == 1:
            return model.flow[l] == model.suseptance[l] * (model.delta[1] - model.delta[2])
        elif l == 3:
            return model.flow[l] == model.suseptance[l] * (model.delta[2] - model.delta[3])
        elif l == 2:
            return model.flow[l] == model.suseptance[l] * (model.delta[1] - model.delta[3])
    model.flow_rule = pyo.Constraint(model.l, rule=flow_rule)

    # Reference angle constraint
    def ref_angle(model):
        return model.delta[1] == 0
    model.ref_angle = pyo.Constraint(rule=ref_angle)

    solver = SolverFactory("gurobi")    # Ensure Gurobi is installed and licenced
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT) # For dual variables
    results = solver.solve(model, tee=True)

    print(f"{'='*10} Optimal Solution {'='*10}")
    print(f"Total Cost: {model.objective():.2f} NOK")
    
    # Extract and display the optimal generation for each generator
    print("\nOptimal generation [MW]:")
    for g in model.g:
        print(f"P_{g}^(gen): {S_base * model.p_G[g].value:.2f}")
    
    # Extract and display voltage angles for each node
    print("\nVoltage angles [rad]:") 
    for n in model.n:
        print(f"Delta {n}: {model.delta[n].value:.4f}")

    # Nodal price calculation
    print("\nNodal prices [NOK/MWh]:")
    for g in model.g:
        dual_value = model.dual.get(model.balance[g])
        if dual_value is not None:
            print(f"Electricity Price at node {g}: {dual_value/S_base:.2f}")
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




