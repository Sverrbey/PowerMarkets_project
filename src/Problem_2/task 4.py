"""
task 4.py  – TET4185 Problem 2‑4
--------------------------------
Full Pyomo‑modell for 3‑nodes DC‑OPF med
• flere generatorer i node 1
• flere (delvis fleksible) laster i node 2
• valgfritt kostnads‑ eller velferdsmål

Bruk:
    python task\ 4.py                 # velferdsmaksimering
    python task\ 4.py --cost_min      # kun kostnadsminimering (del 2‑4 a)
"""
import argparse
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

# ----------------------------- data ----------------------------- #
# Generator‑data
GENS = [
    {"Gen": "Gen 1_1", "Capacity": 300,  "MarginalCost": 200,  "Node": 1},
    {"Gen": "Gen 1_2", "Capacity": 400,  "MarginalCost": 300,  "Node": 1},
    {"Gen": "Gen 1_3", "Capacity": 300,  "MarginalCost": 800,  "Node": 1},
    {"Gen": "Gen 2",   "Capacity": 1000, "MarginalCost": 1000, "Node": 2},
    {"Gen": "Gen 3",   "Capacity": 1000, "MarginalCost": 600,  "Node": 3},
]

# Last‑data
LOADS = [
    {"Load": "Load1",   "Demand": 200, "WTP": None, "Node": 1},  # uelastisk
    {"Load": "Load2_1", "Demand": 200, "WTP": 1300, "Node": 2},
    {"Load": "Load2_2", "Demand": 250, "WTP":  800, "Node": 2},
    {"Load": "Load2_3", "Demand": 250, "WTP":  500, "Node": 2},
    {"Load": "Load3",   "Demand": 500, "WTP": None, "Node": 3},  # uelastisk
]

# Linje‑data  (B-verdiene er gitt negative i Excel; vi bruker absoluttverdiene)
LINES = [
    {"Line": "L12", "Capacity": 500, "B": 20},   # |‑20| = 20
    {"Line": "L13", "Capacity": 500, "B": 10},
    {"Line": "L23", "Capacity": 100, "B": 30},
]
LINE_ENDS = {"L12": (1, 2), "L13": (1, 3), "L23": (2, 3)}

# ----------------------------- modell --------------------------- #
def build_model(use_welfare=True) -> pyo.ConcreteModel():
    m = pyo.ConcreteModel(name="3‑node DC‑OPF")

    # --- menger --- #
    m.N = pyo.Set(initialize=[1, 2, 3])                           # noder
    m.L = pyo.Set(initialize=[l["Line"] for l in LINES])           # linjer
    m.G = pyo.Set(initialize=[g["Gen"]  for g in GENS])            # generatorer
    m.D = pyo.Set(initialize=[d["Load"] for d in LOADS])           # laster

    # --- parametere --- #
    m.CapGen = pyo.Param(m.G, initialize={g["Gen"]: g["Capacity"]      for g in GENS})
    m.Cost   = pyo.Param(m.G, initialize={g["Gen"]: g["MarginalCost"]  for g in GENS})
    m.NodeG  = pyo.Param(m.G, initialize={g["Gen"]: g["Node"]          for g in GENS})

    m.MaxLoad = pyo.Param(m.D, initialize={d["Load"]: d["Demand"] for d in LOADS})
    m.WTP     = pyo.Param(m.D, initialize={d["Load"]: d["WTP"] or 0   for d in LOADS})
    m.NodeD   = pyo.Param(m.D, initialize={d["Load"]: d["Node"]        for d in LOADS})

    m.CapLine = pyo.Param(m.L, initialize={l["Line"]: l["Capacity"] for l in LINES})
    m.B       = pyo.Param(m.L, initialize={l["Line"]: l["B"]        for l in LINES})

    # --- variabler --- #
    m.Pg   = pyo.Var(m.G, bounds=lambda m, g: (0, m.CapGen[g]))
    m.Pd   = pyo.Var(m.D, bounds=lambda m, d: (0, m.MaxLoad[d]))
    m.F    = pyo.Var(m.L, bounds=lambda m, l: (-m.CapLine[l], m.CapLine[l]), initialize=0)
    m.theta = pyo.Var(m.N, initialize=0)  # spenningvinkler (rad)

    # --- infleksible laster --- #
    inflex = [d["Load"] for d in LOADS if d["WTP"] is None]
    def must_serve(m, d):
        if d in inflex:
            return m.Pd[d] == m.MaxLoad[d]
        return pyo.Constraint.Skip
    m.cover_inflex = pyo.Constraint(m.D, rule=must_serve)

    # --- kraftbalanse --- #
    def power_balance(m, n):
        prod = sum(m.Pg[g] for g in m.G if m.NodeG[g] == n)
        cons = sum(m.Pd[d] for d in m.D if m.NodeD[d] == n)
        flow_out = sum(m.F[l] for l in m.L if LINE_ENDS[l][0] == n)
        flow_in  = sum(m.F[l] for l in m.L if LINE_ENDS[l][1] == n)
        return prod - cons == flow_out - flow_in
    m.balance = pyo.Constraint(m.N, rule=power_balance)

    # --- linjeflyt‑likning --- #
    def line_flow(m, l):
        i, j = LINE_ENDS[l]
        return m.F[l] == m.B[l] * (m.theta[i] - m.theta[j])
    m.line_eq = pyo.Constraint(m.L, rule=line_flow)

    # --- referansevinkel --- #
    m.theta_ref = pyo.Constraint(expr=m.theta[1] == 0)

    # --- mål --- #
    m.use_welfare = use_welfare
    def obj_rule(m):
        if m.use_welfare:
            return sum(m.Pd[d] * m.WTP[d] for d in m.D) - sum(m.Pg[g] * m.Cost[g] for g in m.G)
        return sum(m.Pg[g] * m.Cost[g] for g in m.G)
    sense = pyo.maximize if use_welfare else pyo.minimize
    m.OBJ = pyo.Objective(rule=obj_rule, sense=sense)

    # --- suffix for dualer --- #
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m

# ---------------------- løsning og rapport ---------------------- #
def solve_and_report(model: pyo.ConcreteModel):
    solver = SolverFactory("gurobi")
    results = solver.solve(model, tee=False)
    model.solutions.load_from(results)

    print("\n=== NODALE PRISER (dual balance‑eq) ===")
    for n in model.N:
        print(f"Node {n}: {model.dual[model.balance[n]]:7.2f}  NOK/MWh")

    print("\n=== GENERASJON (MW) ===")
    for g in model.G:
        print(f"{g:8s}: {pyo.value(model.Pg[g]):7.2f}")

    print("\n=== LAST (MW) ===")
    for d in model.D:
        print(f"{d:8s}: {pyo.value(model.Pd[d]):7.2f}")

    print("\n=== LINJEFLYT (MW) ===")
    for l in model.L:
        print(f"{l}: {pyo.value(model.F[l]):7.2f}")

    obj_val = pyo.value(model.OBJ)
    label = "Total velferd" if model.use_welfare else "Total kostnad"
    print(f"\n{label}: {obj_val:,.2f}  NOK")

    # Surpluser, kapasitetsleie
    consumer_surplus = sum(pyo.value(model.Pd[d]) * (model.WTP[d] - model.dual[model.balance[model.NodeD[d]]])
                           for d in model.D if model.WTP[d] > 0)
    producer_surplus = sum(pyo.value(model.Pg[g]) * (model.dual[model.balance[model.NodeG[g]]] - model.Cost[g])
                           for g in model.G)
    print(f"\nKonsumentoverskudd: {consumer_surplus:,.2f}  NOK")
    print(f"Produsentoverskudd: {producer_surplus:,.2f}  NOK")

    print("\nKapasitetsleie per linje (NOK):")
    total_rent = 0
    for l in model.L:
        i, j = LINE_ENDS[l]
        rent = (model.dual[model.balance[i]] - model.dual[model.balance[j]]) * pyo.value(model.F[l])
        total_rent += rent
        print(f"{l}: {rent:,.2f}")
    print(f"Total kapasitetsleie: {total_rent:,.2f}  NOK")

    # Plot produksjon
    plt.bar([str(g) for g in model.G], [pyo.value(model.Pg[g]) for g in model.G])
    plt.ylabel("MW")
    plt.title("Produksjon per generator")
    plt.tight_layout()
    plt.show()

# ----------------------------- main ----------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost_min", action="store_true",
                        help="Løs med kostnadsminimering i stedet for velferd")
    args = parser.parse_args()

    mdl = build_model(use_welfare=not args.cost_min)
    solve_and_report(mdl)