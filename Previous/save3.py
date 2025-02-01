# -*- coding: utf-8 -*-

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import re
import math

def main():
    """
        Main function that set up, execute, and store results
    """
    excel_sheets = [
        "Problem 2.2 - Base case",
        "Problem 2.3 - Generators",
        "Problem 2.4 - Loads",
        "Problem 2.5 - Environmental"]
    data = extract_data("Problem 2 data.xlsx")
    data = create_matrices(data[excel_sheets[1]])
    OPF(data)

    return

def extract_data(file):
    data_tot = {}
    excel_sheets = [
        "Problem 2.2 - Base case",
        "Problem 2.3 - Generators",
        "Problem 2.4 - Loads",
        "Problem 2.5 - Environmental"]
    data_names = ["Generator data", "Load data", "Transmission line data"]
    data_loc = {
        "Problem 2.2 - Base case": {
            "Generator data": ["B:E"],
            "Load data": ["K:L"],
            "Transmission line data": ["P:R"]
        },
        "Problem 2.3 - Generators": {
            "Generator data": ["B:E"],
            "Load data": ["K:L"],
            "Transmission line data": ["P:R"]
        },
        "Problem 2.4 - Loads": {
            "Generator data": ["B:E"],
            "Load data": ["K:M"],
            "Transmission line data": ["P:R"]
        },
        "Problem 2.5 - Environmental": {
            "Generator data": ["B:F"],
            "Load data": ["L:N"],
            "Transmission line data": ["R:T"]
        }
    }

    for sheet in excel_sheets:
        data = {}
        for d in data_names:
            # Load Excel file into a pandas DataFrame
            df = pd.read_excel(file, sheet_name=sheet, skiprows=[0, 1], usecols=data_loc[sheet][d][0])
            num = len(df.loc[:])  # Find length of dataframe
            df = df.set_index(np.arange(1, num + 1))  # Use a range of length as index
            df = df.to_dict()
            df[d] = np.arange(1, num + 1)
            data[d] = df

        # Creating a 'to and from' matrix
        t, f = {}, {}
        pattern = r'\d+'
        for i, lineData in enumerate(data["Transmission line data"]['Line'].values()):
            if type(lineData) != float:
                # Find all matches in the string
                matches = re.findall(pattern, lineData)
                t[i + 1] = int(matches[1])
                f[i + 1] = int(matches[0])
        data['Transmission line data']['to'] = t
        data['Transmission line data']['from'] = f
        # Removing the rows with value = 'nan'
        data_loads = {}

        for i, loadData in enumerate(data["Load data"]['Demand [MW]'].values()):
            if not math.isnan(loadData):
                data_loads[i + 1] = loadData

        # Duplicating the loads to match the indices

        data_loads_gens = {}
        for i, genData in enumerate(data["Generator data"]['Location'].values()):
            if genData == 'Node 1' and i == 0:
                data_loads_gens[i+1] = data_loads[1]/3
            elif genData == 'Node 1' and i == 1:
                data_loads_gens[i+1] = data_loads[1]/3
            elif genData == 'Node 1' and i == 2:
                data_loads_gens[i+1] = data_loads[1]/3
            elif genData == 'Node 2':
                data_loads_gens[i+1] = data_loads[2]
            elif genData == 'Node 3':
                data_loads_gens[i+1] = data_loads[3]

        data["Load data"]['Demand [MW]'] = data_loads_gens

        # Removing the rows with value = 'nan'
        data_lines = {}
        for i, lineData in enumerate(data["Transmission line data"]['Capacity [MW].1'].values()):
            if not math.isnan(lineData):
                data_lines[i + 1] = lineData
        data["Transmission line data"]['Capacity [MW].1'] = data_lines
        data_tot[sheet] = data
    return data_tot

def create_matrices(data):
    translate = {
        'Node 1': 1,
        'Node 2': 2,
        'Node 3': 3
    }
    nodeNum = len(list(data["Generator data"]['Capacity [MW]']))
    lineNum = len(list(data["Transmission line data"]["Capacity [MW].1"].keys()))

    B_matrix = np.zeros((nodeNum, nodeNum))  # Create empty matrix
    for n in range(1, nodeNum+1):                   # For every starting node
        for o in range(1, nodeNum+1):               # For every ending node
            for l in range(1, lineNum+1):           # For every line

                a = data["Generator data"]['Location'][n]
                if translate[a] == data["Transmission line data"]["from"][l]:                  # If starting node corresponds to start in line l
                    b = data["Generator data"]['Location'][o]
                    if translate[b] == data["Transmission line data"]["to"][l]:                # If ending node corresponds to end in line l

                        # Admittance added in [n-1,o-1]
                        B_matrix[n-1][o-1] = B_matrix[n-1][o-1] - data["Transmission line data"]["Susceptance [p.u]"][l]
                        # Admittance added in [o-1,n-1]
                        B_matrix[o-1][n-1] = B_matrix[o-1][n-1] - data["Transmission line data"]["Susceptance [p.u]"][l]

                        # Admittance added in [n-1,n-1]
                        B_matrix[n-1][n-1] = B_matrix[n-1][n-1] + data["Transmission line data"]["Susceptance [p.u]"][l]
                        # Admittance added in [n-1,n-1]
                        B_matrix[o-1][o-1] = B_matrix[o-1][o-1] + data["Transmission line data"]["Susceptance [p.u]"][l]

    data["Y"] = B_matrix         # Store the matrix in the dictionary

    DC_matrix = np.zeros((lineNum, nodeNum))  # Dimension CablesxNodes [h,n]
    for h in range(1, lineNum + 1):  # For each cable
        f = data["Transmission line data"]["from"][h]  # Find the starting node of the cable
        t = data["Transmission line data"]["to"][h]  # Find the ending node of the cable
        for n in range(1, nodeNum + 1):
            a = data["Generator data"]['Location'][n]
            if translate[a] == f:
                DC_matrix[h - 1][n - 1] = 1  # Store the cable in the matrix for -1 position. Store value as 1 (meaning positive direction of flow)
            if translate[a] == t:
                DC_matrix[h - 1][n - 1] = -1  # Store the cable in the matrix for -1 position. Store value as -1 (meaning negative direction of flow)

    data["I"] = DC_matrix  # Store the matrix in the dictionary
    return data

def OPF(Data):
    """
       Set up the optimization model, run it and store the data in a .xlsx file
    """
    translate = {
        'Node 1': 1,
        'Node 2': 2,
        'Node 3': 3
    }

    model = pyo.ConcreteModel()  # Establish the optimization model, as a concrete model in this case

    nodeList = list(Data["Generator data"]['Capacity [MW]'])
    nodeList = np.array(nodeList)
    lineList = list(Data["Transmission line data"]["Capacity [MW].1"].keys())
    lineList = np.array(lineList)

    """ Sets """
    model.N = pyo.Set(ordered=True, initialize=nodeList)  # Set for generators
    model.L = pyo.Set(ordered=True, initialize=lineList)  # Set for lines

    """ Parameters """
    model.Pu_base = pyo.Param(initialize=1000)  # Parameter for per unit factor, initializing to 1
    model.P_cap = pyo.Param(model.N, initialize=Data["Generator data"]["Capacity [MW]"])
    model.Cost_gen = pyo.Param(model.N, initialize=Data["Generator data"]["Marginal cost NOK/MWh]"])
    model.Demand = pyo.Param(model.N, initialize=Data["Load data"]["Demand [MW]"])

    model.DC_cap = pyo.Param(model.L, initialize=Data["Transmission line data"]["Capacity [MW].1"])
    model.DC_from = pyo.Param(model.L, initialize=Data['Transmission line data']['from'])
    model.DC_to = pyo.Param(model.L, initialize= Data['Transmission line data']['to'])

    """ Variables """
    model.theta = pyo.Var(model.N)   # Variable for angle on bus for every node
    model.gen = pyo.Var(model.N)    # Variable for generated power on every node
    model.flow = pyo.Var(model.L)    # Variable for flow of power on every node

    """
        Objective function
        Minimize cost associated with production and shedding of generation
    """
    def ObjRule(model):  # Define objective function
        return sum(model.gen[n] * model.Cost_gen[n] for n in model.N)
    model.OBJ = pyo.Objective(rule=ObjRule, sense=pyo.minimize)  # Create objective function based on given function

    """ Constraints """
    def Min_gen(model, n):
        return model.gen[n] >= 0
    model.Min_gen_const = pyo.Constraint(model.N, rule=Min_gen)

    # Maximum generation
    # Every generating unit cannot provide more than maximum capacity

    def Max_gen(model, n):
        return model.gen[n] <= model.P_cap[n]
    model.Max_gen_const = pyo.Constraint(model.N, rule=Max_gen)

    # Maximum from-flow line
    # Sets the higher gap of line flow from unit n
    def From_flow(model, l):
        return model.flow[l] <= model.DC_cap[l]
    model.From_flow_L = pyo.Constraint(model.L, rule=From_flow)

    # Maximum to-flow line
    # Sets the higher gap of line flow to unit n (given as negative flow)

    def To_flow(model, l):
        return model.flow[l] >= -model.DC_cap[l]
    model.To_flow_L = pyo.Constraint(model.L, rule=To_flow)

    # Set the reference node to have a theta == 0
    def ref_node(model, n):
        reference_nodes = [1, 2, 3]  # Task 2.3
        if n in reference_nodes:
            return model.theta[n] == 0
        else:
            return pyo.Constraint.Skip
    model.ref_node_const = pyo.Constraint(model.N, rule=ref_node)

    # Loadbalance; that generation meets demand (and transfer from lines and cables)
    def LoadBal(model, n):
        a = translate[Data['Generator data']['Location'][n]]
        if a == 1:  # Load balance in node 1
            return model.gen[1] + model.gen[2] + model.gen[3] == model.Demand[1] + model.Demand[2] + model.Demand[3] +\
                sum(Data["Y"][0][o - 1] * model.theta[o] * model.Pu_base for o in model.N) +\
                sum(Data["Y"][1][o - 1] * model.theta[o] * model.Pu_base for o in model.N) +\
                sum(Data["Y"][2][o - 1] * model.theta[o] * model.Pu_base for o in model.N) +\
                sum(Data["I"][h - 1][0 - 1] * model.flow[h] for h in model.L) +\
                sum(Data["I"][h - 1][1 - 1] * model.flow[h] for h in model.L) +\
                sum(Data["I"][h - 1][2 - 1] * model.flow[h] for h in model.L)

        return model.gen[n] == model.Demand[n] + sum(Data["Y"][n - 1][o - 1] * model.theta[o] * model.Pu_base for o in model.N) + sum(Data["I"][h - 1][n - 1] * model.flow[h] for h in model.L)

    model.LoadBal_const = pyo.Constraint(model.N, rule=LoadBal)

    #  Flow balance; that flow in line is equal to change in phase angle multiplied with the admittance for the line
    def FlowBal(model, l):

        LHS = model.flow[l]/model.Pu_base
        RHS = 0
        f = model.DC_from[l]
        t = model.DC_to[l]
        if f == 1 and t == 2:  # Line 1
            a = 4
            RHS += -Data["Y"][0][a - 1] * (model.theta[1] - model.theta[a])
            RHS += -Data["Y"][1][a - 1] * (model.theta[2] - model.theta[a])
            RHS += -Data["Y"][2][a - 1] * (model.theta[3] - model.theta[a])
        elif f == 1 and t == 3:  # Line 2
            a = 5
            RHS += -Data["Y"][0][a - 1] * (model.theta[1] - model.theta[a])
            RHS += -Data["Y"][1][a - 1] * (model.theta[2] - model.theta[a])
            RHS += -Data["Y"][2][a - 1] * (model.theta[3] - model.theta[a])
        elif f == 2 and t == 3:  # Line 3
            RHS += -Data["Y"][4 - 1][5 - 1] * (model.theta[5] - model.theta[4])
        return LHS == RHS
    model.FlowBal_const = pyo.Constraint(model.L, rule=FlowBal)

    """
        Compute the optimization problem
    """

    # Set the solver for this
    opt = SolverFactory('gurobi')

    # Enable dual variable reading -> important for dual values of results
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Solve the problem
    results = opt.solve(model, load_solutions=True)

    # Write result on performance
    #results.write(num=1)
    model.dual.display()
    model.display()

    NodeData = {}
    DualNode = {}
    for node in model.N:
        DualNode[node] = round(model.dual[model.LoadBal_const[node]], 1)
    NodeData["Price"] = DualNode
    print(NodeData)
    return

main()