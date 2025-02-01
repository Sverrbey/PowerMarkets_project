# -*- coding: utf-8 -*-

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import re
import math

# Function to extract integer from a string
def extract_integer(x):
    if isinstance(x, str):
        match = re.search(r'\d+', x)
        if match:
            return int(match.group())
        elif isinstance(x, float) and not np.isnan(x):  # Check if x is a non-NaN float
            return round(x)
    return None

def dataframe_to_dict(df, index_col):
    result_dict = {}
    for index, row in df.iterrows():
        # Check if the index value is not NaN and is a valid integer
        if pd.notnull(row[index_col]) and row[index_col] == int(row[index_col]):
            idx = int(row[index_col])  # Ensure the index value is an integer
            if idx not in result_dict:
                result_dict[idx] = []
            # Append values to the list corresponding to the index
            result_dict[idx].append(row.drop(index_col).to_dict())
    return result_dict

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
            "Generator data": ["B:E", 3, 2],
            "Load data": ["K:L", 3, 1],
            "Transmission line data": ["P:R", 3]
        },
        "Problem 2.3 - Generators": {
            "Generator data": ["B:E", 5, 2],
            "Load data": ["K:L", 3, 1],
            "Transmission line data": ["P:R", 3]
        },
        "Problem 2.4 - Loads": {
            "Generator data": ["B:E", 5, 2],
            "Load data": ["K:M", 5, 2],
            "Transmission line data": ["P:R", 3]
        },
        "Problem 2.5 - Environmental": {
            "Generator data": ["B:F", 5, 2],
            "Load data": ["L:N", 5, 2],
            "Transmission line data": ["R:T", 3]
        }
    }
    pattern = r'\d+'
    for sheet in excel_sheets:
        data = {}
        for d in data_names:
            # Load Excel file into a pandas DataFrame
            df = pd.read_excel(file, sheet_name=sheet, skiprows=[0, 1], usecols=data_loc[sheet][d][0])
            num = len(df.loc[:])  # Find length of dataframe
            data_loc[sheet][d][1] = num  # Store length of dataframe in dictionary

            if d == "Transmission line data":
                df = df.set_index(np.arange(1, num + 1))  # Use a range of length as index
                df = df.to_dict()
            else:  # Indexing for generator data for arbitrary amount of generators
                col = data_loc[sheet][d][2]
                loc = df.columns[col]
                df['Index'] = df[loc].apply(extract_integer)
                df = dataframe_to_dict(df, 'Index')
            data[d] = df
        # Creating a 'to and from' matrix
        t, f = {}, {}
        for i, lineData in enumerate(data["Transmission line data"]['Line'].values()):
            if type(lineData) != float:
                # Find all matches in the string
                matches = re.findall(pattern, lineData)
                t[i + 1] = int(matches[1])
                f[i + 1] = int(matches[0])

        data_lines = {}
        for i, lineData in enumerate(data["Transmission line data"]['Capacity [MW].1'].values()):
            if not math.isnan(lineData):
                data_lines[i + 1] = lineData

        data["Transmission line data"]['Capacity [MW].1'] = data_lines
        data['Transmission line data']['to'] = t
        data['Transmission line data']['from'] = f

        data_load = {}
        for i, x in enumerate(data['Load data'].values()):
            data_load[i+1] = x[0]
        data['Load data'] = data_load
        data_tot[sheet] = data
        #print(data_tot[sheet]['Transmission line data']['Capacity [MW].1'])
    return data_tot

def create_matrices(data):
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

    # Start creating the I_matrix
    I_matrix = np.zeros((3, 3))  # Dimension CablesxNodes [h,n]
    I_matrix[0][1] = 1
    I_matrix[1][0] = -1
    I_matrix[0][2] = 1
    I_matrix[2][0] = -1
    I_matrix[1][2] = 1
    I_matrix[2][1] = -1

    data["I"] = I_matrix

    return(data)

def OPF(Data):
    """
       Set up the optimization model, run it and store the data in a .xlsx file
    """
    model = pyo.ConcreteModel()  # Establish the optimization model, as a concrete model in this case

    """ Sets """

    nodeList = list(Data["Generator data"].keys())
    lineList = list(Data["Transmission line data"]["Capacity [MW].1"].keys())

    model.N = pyo.Set(ordered=True, initialize=nodeList)  # Set for generators
    model.L = pyo.Set(ordered=True, initialize=lineList)  # Set for lines


    """ Parameters """
    # Arbitrary parameters
    model.P_cap = pyo.Param(model.N, within=pyo.NonNegativeReals, mutable=True)  # Parameter for max production for every node
    model.Cost_gen = pyo.Param(model.N, within=pyo.NonNegativeReals, mutable=True)  # Parameter for generation cost for every node
    model.Demand = pyo.Param(model.N, within=pyo.NonNegativeReals, mutable=True)  # Parameter for generation cost for every node
    for n in model.N:
        model.P_cap[n] = Data["Generator data"][n][0]["Capacity [MW]"]
        model.Cost_gen[n] = Data["Generator data"][n][0]["Marginal cost NOK/MWh]"]
        model.Demand[n] = Data["Load data"][n]["Demand [MW]"]  # Parameter for demand for every node

    model.DC_cap = pyo.Param(model.L, initialize=Data["Transmission line data"]["Capacity [MW].1"])  # Parameter for Cable capacity for every cable
    model.Pu_base = pyo.Param(initialize=1000)  # Parameter for per unit factor, initializing to 1
    model.DC_from = pyo.Param(model.L,initialize=Data['Transmission line data']['from'])  # Parameter for starting node for every line
    model.DC_to = pyo.Param(model.L, initialize=Data['Transmission line data']['to'])  # Parameter for ending node for every line

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
        return model.gen[n] == model.Demand[n] +\
            sum(Data["Y"][n-1][o-1]*model.theta[o]*model.Pu_base for o in model.N) #+ sum(Data["I"][h-1][n-1]*model.flow[h] for h in model.L)
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