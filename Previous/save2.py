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
            df = pd.read_excel(file, sheet_name=sheet, skiprows=[0, 1])
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
            if genData == 'Node 1':
                data_loads_gens[i+1] = data_loads[1]
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
    print(data_tot)
    return data_tot

def create_matrices(data):
    """
    num_nodes = 3
    Ybus = np.zeros((num_nodes, num_nodes))  # Create empty matrix

    Y12 = data['Transmission line data']['Susceptance [p.u]'][1]
    Y13 = data['Transmission line data']['Susceptance [p.u]'][2]
    Y23 = data['Transmission line data']['Susceptance [p.u]'][3]

    # Calculating the diagonal elements in the Ybus
    Y11 = -Y12 - Y13
    Y22 = -Y12 - Y23
    Y33 = -Y13 - Y23

    # Adding the elements to the Y matrix
    Ybus[0][1] = Ybus[1][0] = Y12
    Ybus[0][2] = Ybus[2][0] = Y13
    Ybus[1][2] = Ybus[2][1] = Y23

    Ybus[0][0] = Y11
    Ybus[1][1] = Y22
    Ybus[2][2] = Y33

    data['Y'] = Ybus
    """
    B_matrix = np.zeros((3, 3))  # Create empty matrix
    for n in range(1, 3+1):                   # For every starting node
        for o in range(1, 3+1):               # For every ending node
            for l in range(1, 3+1):           # For every line
                if n == data["Transmission line data"]["from"][l]:                  # If starting node corresponds to start in line l
                    if o == data["Transmission line data"]["to"][l]:                # If ending node corresponds to end in line l

                        B_matrix[n-1][o-1] = B_matrix[n-1][o-1] - data["Transmission line data"]["Susceptance [p.u]"][l]  # Admittance added in [n-1,o-1]

                        B_matrix[o-1][n-1] = B_matrix[o-1][n-1] - data["Transmission line data"]["Susceptance [p.u]"][l]  # Admittance added in [o-1,n-1]

                        B_matrix[n-1][n-1] = B_matrix[n-1][n-1] + data["Transmission line data"]["Susceptance [p.u]"][l]  # Admittance added in [n-1,n-1]

                        B_matrix[o-1][o-1] = B_matrix[o-1][o-1] + data["Transmission line data"]["Susceptance [p.u]"][l]  # Admittance added in [n-1,n-1]

    data["Y"] = B_matrix         # Store the matrix in the dictionary

    # Start creating the I_matrix
    I_matrix = np.zeros((3, 3))  # Dimension CablesxNodes [h,n]

    I_matrix[0][1] = 1
    I_matrix[1][0] = -1

    I_matrix[0][2] = 1
    I_matrix[2][0] = -1

    I_matrix[1][2] = 1
    I_matrix[2][1] = -1

    data["I"] = I_matrix

    return data

def OPF(Data):
    """
       Set up the optimization model, run it and store the data in a .xlsx file
    """

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
    model.DC_to = pyo.Param(model.L, initialize=Data['Transmission line data']['to'])

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

    # Maximum from-flow cable
    # Sets the higher gap of cable flow from unit n

    def FlowBalDC_max(model, h):
        return model.flow[h] <= model.DC_cap[h]
    model.FlowBalDC_max_const = pyo.Constraint(model.L, rule=FlowBalDC_max)

    # Maximum to-flow cable
    # Sets the higher gap of cable flow to unit n (given as negative flow)

    def FlowBalDC_min(model, h):
        return model.flow[h] >= -model.DC_cap[h]
    model.FlowBalDC_min_const = pyo.Constraint(model.L, rule=FlowBalDC_min)

    # Set the reference node to have a theta == 0
    def ref_node(model):
        return model.theta[1] == 0
    model.ref_node_const = pyo.Constraint(rule=ref_node)

    # Loadbalance; that generation meets demand (and transfer from lines and cables)
    def LoadBal(model, n):
        # Task 2.2
        #return model.gen[n] == model.Demand[n] + sum(Data["Y"][n - 3][o - 1] * model.theta[o] * model.Pu_base for o in model.L)  # + sum(Data["I"][h-1][n-1]*model.flow[h] for h in model.L)

        # Task 2.3
        LH = 0
        for m in model.N:  # n = 4, m = 3
            z = m - 3
            y = n - 3
            itr = 0
            if Data["Generator data"]['Location'][n] == 'Node 1':
                if Data["Generator data"]['Location'][m] == 'Node 1':
                    itr += Data["Y"][0][0] * model.theta[m] * model.Pu_base
                else:
                    itr += Data["Y"][0][z] * model.theta[m] * model.Pu_base
            else:
                if Data["Generator data"]['Location'][m] == 'Node 1':
                    itr += Data["Y"][y][0] * model.theta[m] * model.Pu_base
                else:
                    itr += Data["Y"][y][z] * model.theta[m] * model.Pu_base
            LH += itr
        return model.gen[n] == model.Demand[n] + LH
    model.LoadBal_const = pyo.Constraint(model.N, rule=LoadBal)

    #  Flow balance; that flow in line is equal to change in phase angle multiplied with the admittance for the line
    def FlowBal(model, l):
        return model.flow[l]/model.Pu_base == (model.theta[model.DC_from[l]] - model.theta[model.DC_to[l]]) * -Data["Y"][model.DC_from[l]-1][model.DC_to[l]-1]
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

    return

main()