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
    data = extract_data("Problem 2 data edit.xlsx")
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
            "Transmission line data": ["O:R"]
        },
        "Problem 2.3 - Generators": {
            "Generator data": ["B:E"],
            "Load data": ["K:L"],
            "Transmission line data": ["O:R"]
        },
        "Problem 2.4 - Loads": {
            "Generator data": ["B:E"],
            "Load data": ["K:M"],
            "Transmission line data": ["O:R"]
        },
        "Problem 2.5 - Environmental": {
            "Generator data": ["B:F"],
            "Load data": ["L:N"],
            "Transmission line data": ["Q:T"]
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

        # Removing the rows with value = 'nan'
        data_loads = {}

        for key, val in data["Load data"]['Demand [MW]'].items():
            if not math.isnan(val):
                data_loads[key] = val
        data_loads_loc = {}
        for key, val in data["Load data"]['Location.1'].items():
            if not math.isnan(val):
                data_loads_loc[key] = int(val)

        data["Load data"]['Demand [MW]'] = data_loads
        data["Load data"]['Location.1'] = data_loads_loc

        new_load = {}
        # Creating an array for multiple loads on a node
        for key, val in data["Load data"]['Location.1'].items():
            new_load[val] = []

        for key, val in data["Load data"]['Location.1'].items():
            new_load[val].append(data["Load data"]['Demand [MW]'][key])


        # Duplicating the loads to match the indices
        data_loads_gens = {}
        for key, val in data["Load data"]['Demand [MW]'].items():
            if key == 1:
                for i in range(1, 3+1):
                    data_loads_gens[i] = {1: new_load[1][0], 2: 0, 3: 0}
            elif key == 2 and len(new_load[2]) == 1:
                data_loads_gens[4] = {1: new_load[2][0], 2: 0, 3: 0}
            elif key == 2:
                data_loads_gens[4] = {1: new_load[2][0], 2: new_load[2][1], 3: new_load[2][2]}
            else:
                data_loads_gens[5] = {1: new_load[3][0], 2: 0, 3: 0}

        data["Load data"]['Demand [MW]'] = data_loads_gens
        # Removing the rows with value = 'nan'
        data_lines = {}
        for key, val in data["Transmission line data"]['Capacity [MW].1'].items():
            if not math.isnan(val):
                data_lines[key] = int(val)
        data["Transmission line data"]['Capacity [MW].1'] = data_lines
        data_lines = {}
        for key, val in data["Transmission line data"]['From'].items():
            if not math.isnan(val):
                data_lines[key] = int(val)
        data["Transmission line data"]['From'] = data_lines

        data_lines = {}
        for key, val in data["Transmission line data"]['To'].items():
            if not math.isnan(val):
                data_lines[key] = int(val)
        data["Transmission line data"]['To'] = data_lines

        data_tot[sheet] = data
    return data_tot

def create_matrices(data):


    nodeNum = len(list(data["Generator data"]['Capacity [MW]']))
    lineNum = len(list(data["Transmission line data"]["Capacity [MW].1"].keys()))

    B_matrix = np.zeros((nodeNum, nodeNum))  # Create empty matrix
    for n in range(1, nodeNum+1):                   # For every starting node
        for o in range(1, nodeNum+1):               # For every ending node
            for l in range(1, lineNum+1):           # For every line
                if n == data["Transmission line data"]["From"][l]:                  # If starting node corresponds to start in line l
                    if o == data["Transmission line data"]["To"][l]:                # If ending node corresponds to end in line l

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
        f = data["Transmission line data"]["From"][h]  # Find the starting node of the cable
        t = data["Transmission line data"]["To"][h]  # Find the ending node of the cable
        for n in range(1, nodeNum + 1):
            if n == f:
                DC_matrix[h - 1][n - 1] = 1  # Store the cable in the matrix for -1 position. Store value as 1 (meaning positive direction of flow)
            if n == t:
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
    loadList = np.array([1, 2, 3])

    """ Sets """
    model.N = pyo.Set(ordered=True, initialize=nodeList)  # Set for generators
    model.L = pyo.Set(ordered=True, initialize=lineList)  # Set for lines
    model.Loads = pyo.Set(ordered=True, initialize=loadList)  # Set for loads
    """ Parameters """
    model.Pu_base = pyo.Param(initialize=1000)  # Parameter for per unit factor, initializing to 1
    model.P_cap = pyo.Param(model.N, initialize=Data["Generator data"]["Capacity [MW]"])
    model.Cost_gen = pyo.Param(model.N, initialize=Data["Generator data"]["Marginal cost NOK/MWh]"])

    #model.Demand = pyo.Param(model.N, model.Loads, initialize=Data["Load data"]["Demand [MW]"])
    demand_data = {}
    for gen_index, load_data in Data["Load data"]["Demand [MW]"].items():
        if isinstance(load_data, dict):  # Check if load_data is a dictionary
            for load_index, demand_value in load_data.items():
                demand_data[(gen_index, load_index)] = demand_value
        else:  # If load_data is an integer
            demand_data[(gen_index, 1)] = load_data  # Assume it's a single load value
    model.Demand = pyo.Param(model.N, model.Loads, initialize=demand_data)

    model.DC_cap = pyo.Param(model.L, initialize=Data["Transmission line data"]["Capacity [MW].1"])
    model.DC_from = pyo.Param(model.L, initialize=Data['Transmission line data']['From'])
    model.DC_to = pyo.Param(model.L, initialize=Data['Transmission line data']['To'])

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
        # Task 2.2
        # return model.gen[n] == model.Demand[n] + sum(Data["Y"][n - 3][o - 1] * model.theta[o] * model.Pu_base for o in model.L)  # + sum(Data["I"][h-1][n-1]*model.flow[h] for h in model.L)

        # Task 2.3
        if Data['Generator data']['Location'][n] == 1:  # Load balance in node 1
            demand1_sum = sum(model.Demand[1, l] for l in model.Loads)
            demand2_sum = sum(model.Demand[2, l] for l in model.Loads)
            demand3_sum = sum(model.Demand[3, l] for l in model.Loads)
            return model.gen[1] + model.gen[2] + model.gen[3] == demand1_sum + demand2_sum + demand3_sum +\
                sum(Data["Y"][0][o - 1] * model.theta[o] * model.Pu_base for o in model.N) +\
                sum(Data["Y"][1][o - 1] * model.theta[o] * model.Pu_base for o in model.N) +\
                sum(Data["Y"][2][o - 1] * model.theta[o] * model.Pu_base for o in model.N) +\
                sum(Data["I"][h - 1][0] * model.flow[h] for h in model.L) +\
                sum(Data["I"][h - 1][1] * model.flow[h] for h in model.L) +\
                sum(Data["I"][h - 1][2] * model.flow[h] for h in model.L)
        if Data['Generator data']['Location'][n] == 2:
            demand_sum = sum(model.Demand[n, l] for l in model.Loads)
            return model.gen[4] == demand_sum + \
                sum(Data["Y"][3][o - 1] * model.theta[o] * model.Pu_base for o in model.N) + \
                sum(Data["I"][h - 1][3] * model.flow[h] for h in model.L)
        demand_sum = sum(model.Demand[n, l] for l in model.Loads)
        return model.gen[n] == demand_sum + sum(Data["Y"][n - 1][o - 1] * model.theta[o] * model.Pu_base for o in model.N) + sum(Data["I"][h - 1][n - 1] * model.flow[h] for h in model.L)
        #return model.gen[n] == model.Demand[n] + sum(Data["Y"][n - 1][o - 1] * model.theta[o] * model.Pu_base for o in model.N) #+ sum(Data["I"][h - 1][n - 1] * model.flow[h] for h in model.L)
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
        #nodeName = Data["Generator data"]["Location"][node]
        #DualNode[nodeName] = round(model.dual[model.LoadBal_const[node]], 1)
        DualNode[node] = round(model.dual[model.LoadBal_const[node]], 1)
    NodeData["Price"] = DualNode
    print(NodeData)
    return

main()