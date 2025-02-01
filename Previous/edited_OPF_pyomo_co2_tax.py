# -*- coding: utf-8 -*-
"""
Optimal Power Flow with DC power flow
   =====================================
  (c) Gerard Doorman, December 2012
      Hossein farahmand, February 2016

Originally implemented for MOSEL XPRESS

Converted to Python/Pyomo for semester 2018/2019:

  (c) Kasper Emil Thorvaldsen, December 2018

"""

import numpy as np
import sys
# import time
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

import matplotlib.pyplot as plt

def Main():

    """
    Main function that set up, execute, and store results
    """

    Data = Read_Excel("Nordic_wet_task34.xlsx")    # Master dictionary, created from input data dictionary

    # Quick check to see that the reference node is valid:
    if any([Data["Reference node"] <= 0, Data["Reference node"] > Data["Nodes"]["NumNodes"]]):
        print("Invalid reference node, the number of nodes does not allow this node to be reference!")
        sys.exit()

    # Quick check to see that the method is a valid option:
    if all((Data["DCFlow"] != 1, Data["DCFlow"] != 0)):
        print("Invalid method chosen. Either choose 1 or 0")
        sys.exit()

    # End user-specified parameters

    # Create matrices for the lines and cables
    Data = Create_matrices(Data)

    """
    Everything is set up, now we run the model 
    """

    OPF_model(Data)     # Run the model with the set data

    return

# Prepares the datasheet for reading
def Read_Excel(name):

    """
    Reads input excel file and reads the data into dataframes.
    Separates between each sheet, and stores into one dictionary
    """

    data = {}                       # Dictionary storing the input data

    Excel_sheets = ["Node Parameters", "AC Branch Parameters", "DC Link Parameters"]    # Name of each sheet
    Data_names = {"Node Parameters": "Nodes", "AC Branch Parameters": "AC-lines", "DC Link Parameters": "DC-lines"}    # Names for data for each sheet
    Num_Names = {"Node Parameters": "NumNodes", "AC Branch Parameters": "NumAC", "DC Link Parameters": "NumDC"}        # Names for numbering
    List_Names = {"Node Parameters": "NodeList", "AC Branch Parameters": "ACList", "DC Link Parameters": "DCList"}     # Names for numbering

    for sheet in Excel_sheets:      # For each sheet
        df = pd.read_excel(name, sheet_name=sheet, skiprows=1)  # Read sheet, exclude title

        df = df.set_index(df.columns[0])                        # Set first column as index
        num = len(df.loc[:])                                    # Find length of dataframe
        df = df.to_dict()

        df[Num_Names[sheet]] = num                              # Store length of dataframe in dictionary
        df[List_Names[sheet]] = np.arange(1, num+1)

        data[Data_names[sheet]] = df                            # Store dataframe in dictionary

        # End for

    # Extract data from the declaration sheet

    df = pd.read_excel(name, sheet_name="Declarations", skiprows=1)  # Get from Declaration sheet
    df = df.set_index(df.columns[0])                                 # Make first column index
    df = df.to_dict()                                                # Convert to dictionary

    data["DCFlow"] = df["Value"][1]  # True: DC power flow, False: Transport network (ATC)
    data["Reference node"] = df["Value"][2]  # The reference node where Theta = 0
    data["pu-Base"] = df["Value"][3]  # The per unit base [kW]
    data["ShedCost"] = df["Value"][4]  # Cost of shedding load

    return data        # Return datasheet
# B-matrix, DC-matrix, X-matrix
def Create_matrices(Data):

    """
    Setting up matrices for the problems:
        - B-matrix  -> Admittance matrix for the lines. Used in DCOPF
        - DC-matrix -> Bus incidence matrix for the DC cables. Used in both DCOPF and ATC
        - X-matrix  -> Bus incidence matrix for AC cables. Used in ATC
    """

    # Start creating admittance matrix X. Adding in [n-1] is due to avoiding the 0-th index. This is only for DCOPF

    B_matrix = np.zeros((Data["Nodes"]["NumNodes"], Data["Nodes"]["NumNodes"]))  # Create empty matrix

    for n in range(1, Data["Nodes"]["NumNodes"]+1):                   # For every starting node
        for o in range(1, Data["Nodes"]["NumNodes"]+1):               # For every ending node
            for l in range(1, Data["AC-lines"]["NumAC"]+1):           # For every line
                if n == Data["AC-lines"]["From"][l]:                  # If starting node corresponds to start in line l
                    if o == Data["AC-lines"]["To"][l]:                # If ending node corresponds to end in line l

                        B_matrix[n-1][o-1] = B_matrix[n-1][o-1] - Data["AC-lines"]["Admittance"][l]  # Admittance added in [n-1,o-1]

                        B_matrix[o-1][n-1] = B_matrix[o-1][n-1] - Data["AC-lines"]["Admittance"][l]  # Admittance added in [o-1,n-1]

                        B_matrix[n-1][n-1] = B_matrix[n-1][n-1] + Data["AC-lines"]["Admittance"][l]  # Admittance added in [n-1,n-1]

                        B_matrix[o-1][o-1] = B_matrix[o-1][o-1] + Data["AC-lines"]["Admittance"][l]  # Admittance added in [n-1,n-1]

    Data["B-matrix"] = B_matrix         # Store the matrix in the dictionary

    # Start creating the DC-matrix
    DC_matrix = np.zeros((Data["DC-lines"]["NumDC"], Data["Nodes"]["NumNodes"]))    # Dimension CablesxNodes [h,n]

    for h in range(1, Data["DC-lines"]["NumDC"]+1):     # For each cable

        node_pos = Data["DC-lines"]["From"][h]          # Find the starting node of the cable
        DC_matrix[h-1][node_pos-1] = 1                  # Store the cable in the matrix for -1 position. Store value as 1 (meaning positive direction of flow)

        node_pos = Data["DC-lines"]["To"][h]            # Find the ending node of the cable
        DC_matrix[h-1][node_pos-1] = -1                 # Store the cable in the matrix for -1 position. Store value as -1 (meaning negative direction of flow)

    Data["DC-matrix"] = DC_matrix       # Store the matrix in the dictionary

    # Start creating the X-matrix

    X_matrix = np.zeros((Data["AC-lines"]["NumAC"],Data["Nodes"]["NumNodes"]))      # Dimension LinesxNodes [l,n]

    for l in range(1,Data["AC-lines"]["NumAC"]+1):      # For each line

        node_pos = Data["AC-lines"]["From"][l]          # Find the starting node
        X_matrix[l-1][node_pos-1] = 1                   # Store the line starting position as a value 1

        node_pos = Data["AC-lines"]["To"][l]            # Find the ending node
        X_matrix[l-1][node_pos-1] = -1                  # Store the line ending position as a value -1

    Data["X-matrix"] = X_matrix         # Store the matrix in the dictionary

    return(Data)

def OPF_model(Data):

    """
    Set up the optimization model, run it and store the data in a .xlsx file
    """

    model = pyo.ConcreteModel()  # Establish the optimization model, as a concrete model in this case

    """
    Sets
    """

    model.L = pyo.Set(ordered=True, initialize=Data["AC-lines"]["ACList"])  # Set for AC lines
    model.N = pyo.Set(ordered=True, initialize=Data["Nodes"]["NodeList"])   # Set for nodes
    model.H = pyo.Set(ordered=True, initialize=Data["DC-lines"]["DCList"])  # Set for DC lines

    """Parameters"""

    # Nodes
    model.Demand = pyo.Param(model.N, initialize=Data["Nodes"]["DEMAND"])  # Parameter for demand for every node
    model.P_min = pyo.Param(model.N, initialize=Data["Nodes"]["GENMIN"])  # Parameter for minimum production for every node
    model.P_max = pyo.Param(model.N, initialize=Data["Nodes"]["GENCAP"])  # Parameter for max production for every node
    model.Cost_gen = pyo.Param(model.N, initialize=Data["Nodes"]["GENCOST"]) # Parameter for generation cost for every node
    model.Cost_shed = pyo.Param(initialize=Data["ShedCost"])                  # Parameter for cost of shedding power
    model.Pu_base = pyo.Param(initialize=Data["pu-Base"])                   # Parameter for per unit factor

    # AC-lines
    model.P_AC_max = pyo.Param(model.L, initialize=Data["AC-lines"]["Cap From"])    # Parameter for max transfer from node, for every line
    model.P_AC_min = pyo.Param(model.L, initialize=Data["AC-lines"]["Cap To"])      # Parameter for max transfer to node, for every line
    model.AC_from = pyo.Param(model.L, initialize=Data["AC-lines"]["From"])         # Parameter for starting node for every line
    model.AC_to = pyo.Param(model.L, initialize=Data["AC-lines"]["To"])             # Parameter for ending node for every line

    # DC-lines
    model.DC_cap = pyo.Param(model.H, initialize=Data["DC-lines"]["Cap"])           # Parameter for Cable capacity for every cable

    """
    Variables
    """

    # Nodes
    model.theta = pyo.Var(model.N)                                       # Variable for angle on bus for every node
    model.gen = pyo.Var(model.N)                                         # Variable for generated power on every node
    model.shed = pyo.Var(model.N, within=pyo.NonNegativeReals)  # Variable for shed power on every node

    # AC-lines
    model.flow_AC = pyo.Var(model.L)                                      # Variable for power flow on every line

    # DC-lines
    model.flow_DC = pyo.Var(model.H)                                      # Variable for power flow on every cable

    """
    Objective function
    Minimize cost associated with production and shedding of generation
    """

    def ObjRule(model):  # Define objective function
        return sum(model.gen[n]*model.Cost_gen[n] for n in model.N) +\
                sum(model.shed[n]*model.Cost_shed for n in model.N)
    model.OBJ = pyo.Objective(rule=ObjRule, sense=pyo.minimize)   # Create objective function based on given function

    """
    Constraints
    """

    # Minimum generation
    # Every generating unit must provide at least the minimum capacity

    def Min_gen(model, n):
        return model.gen[n] >= model.P_min[n]
    model.Min_gen_const = pyo.Constraint(model.N, rule=Min_gen)

    # Maximum generation
    # Every generating unit cannot provide more than maximum capacity

    def Max_gen(model, n):
        return model.gen[n] <= model.P_max[n]
    model.Max_gen_const = pyo.Constraint(model.N, rule=Max_gen)

    # Maximum from-flow line
    # Sets the higher gap of line flow from unit n

    def From_flow(model, l):
        return model.flow_AC[l] <= model.P_AC_max[l]
    model.From_flow_L = pyo.Constraint(model.L, rule=From_flow)

    # Maximum to-flow line
    # Sets the higher gap of line flow to unit n (given as negative flow)

    def To_flow(model, l):
        return model.flow_AC[l] >= -model.P_AC_min[l]
    model.To_flow_L = pyo.Constraint(model.L, rule=To_flow)

    # Maximum from-flow cable
    # Sets the higher gap of cable flow from unit n

    def FlowBalDC_max(model, h):
        return model.flow_DC[h] <= model.DC_cap[h]
    model.FlowBalDC_max_const = pyo.Constraint(model.H, rule=FlowBalDC_max)

    # Maximum to-flow cable
    # Sets the higher gap of cable flow to unit n (given as negative flow)

    def FlowBalDC_min(model, h):
        return model.flow_DC[h] >= -model.DC_cap[h]
    model.FlowBalDC_min_const = pyo.Constraint(model.H, rule=FlowBalDC_min)

    # If we want to run the model using DC Optimal Power Flow
    if Data["DCFlow"]:

        # Set the reference node to have a theta == 0

        def ref_node(model):
            return model.theta[Data["Reference node"]] == 0
        model.ref_node_const = pyo.Constraint(rule=ref_node)

        # Loadbalance; that generation meets demand, shedding, and transfer from lines and cables

        def LoadBal(model, n):
            return model.gen[n] + model.shed[n] == model.Demand[n] +\
                sum(Data["B-matrix"][n-1][o-1]*model.theta[o]*model.Pu_base for o in model.N) +\
                sum(Data["DC-matrix"][h-1][n-1]*model.flow_DC[h] for h in model.H)
        model.LoadBal_const = pyo.Constraint(model.N, rule=LoadBal)

        # Flow balance; that flow in line is equal to change in phase angle multiplied with the admittance for the line

        def FlowBal(model, l):
            return model.flow_AC[l]/model.Pu_base == ((model.theta[model.AC_from[l]]- model.theta[model.AC_to[l]])*-Data["B-matrix"][model.AC_from[l]-1][model.AC_to[l]-1])
        model.FlowBal_const = pyo.Constraint(model.L, rule=FlowBal)

    else:           # If we are to run this using ATC-rules

        # Loadbalance; that generation meets demand, shedding, and transfer from lines and cables
        def LoadBal(model, n):
            return model.gen[n] + model.shed[n] == model.Demand[n] +\
                   sum(Data["X-matrix"][l-1][n-1]*model.flow_AC[l] for l in model.L) +\
                   sum(Data["DC-matrix"][h-1][n-1]*model.flow_DC[h] for h in model.H)
        model.LoadBal_const = pyo.Constraint(model.N, rule=LoadBal)

    """
    Compute the optimization problem
    """

    # Set the solver for this
    opt = SolverFactory('gurobi')
    # opt = SolverFactory('gurobi', solver_io="python")

    # Enable dual variable reading -> important for dual values of results
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Solve the problem
    results = opt.solve(model, load_solutions=True)

    # Write result on performance
    #results.write(num=1)

    #model.display()

    # Run function that store results
    Store_model_data(model, Data)

    return

def Store_model_data(model, Data):

    """
    Stores the results from the optimization model run into an excel file
    """

    # Create empty dictionaries that will be filled
    NodeData = {}
    ACData = {}
    DCData = {}
    MiscData = {}

    # Node data

    # Write dictionaries for each node related value
    Theta = {}
    Gen = {}
    Shed = {}
    Demand = {}
    CostGen = {}
    CostShed = {}
    DualNode = {}

    # For every node, store the data in the respective dictionary
    for node in model.N:

        #If we have DC OPF, we want to store Theta, if not then we skip it
        if Data["DCFlow"]:
            Theta[node] = round(model.theta[node].value, 4)

        DualNode[node] = round(model.dual[model.LoadBal_const[node]], 1)
        Gen[node] = round(model.gen[node].value, 4)
        Shed[node] = round(model.shed[node].value, 4)
        Demand[node] = round(model.Demand[node], 4)
        CostGen[node] = round(model.gen[node].value*model.Cost_gen[node], 4)
        CostShed[node] = round(model.shed[node].value*model.Cost_shed, 4)

    # Store Node Data
    NodeData["Theta [rad]"] = Theta
    NodeData["Gen"] = Gen
    NodeData["Shed"] = Shed
    NodeData["Demand"] = Demand
    NodeData["MargCost"] = Data["Nodes"]["GENCOST"]
    NodeData["CostGen"] = CostGen
    NodeData["CostShed"] = CostShed
    NodeData["Node Name"] = Data["Nodes"]["NNAMES"]
    NodeData["Price"] = DualNode

    # AC-line data
    ACFlow = {}
    DualFrom = {}
    DualTo = {}

    # For every line, store the result
    for line in model.L:
        ACFlow[line] = round(model.flow_AC[line].value, 4)

        # Only if DCOPF is true
        if Data["DCFlow"]:
            DualFrom[line] = round(model.dual[model.FlowBal_const[line]]/Data["pu-Base"], 1)

        # If not, then we store the dual values for both the max and minimum flow constraints
        else:
            DualFrom[line] = round(model.dual[model.From_flow_L[line]], 1)
            DualTo[line] = round(model.dual[model.To_flow_L[line]], 1)

    # Extract data from input that can be shown with the results
    ACData["AC Flow"] = ACFlow
    ACData["Max power from"] = Data["AC-lines"]["Cap From"]
    ACData["Max power to"] = Data["AC-lines"]["Cap To"]
    ACData["From Node"] = Data["AC-lines"]["From"]
    ACData["To Node"] = Data["AC-lines"]["To"]

    # This one is only necessary to include if we have DC OPF
    if Data["DCFlow"]:
        ACData["Admittance"] = Data["AC-lines"]["Admittance"]
        ACData["Dual Value"] = DualFrom

    else:
        ACData["Dual Value from"] = DualFrom
        ACData["Dual Value to"] = DualTo

    # DC-line data
    DCFlow = {}

    # For every cable, store the result
    for cable in model.H:
        DCFlow[cable] = round(model.flow_DC[cable].value, 4)

    DCData["DC Flow"] = DCFlow
    DCData["Capacity"] = Data["DC-lines"]["Cap"]
    DCData["From Node"] = Data["DC-lines"]["From"]
    DCData["To Node"] = Data["DC-lines"]["To"]


    #Misc
    Objective = round(model.OBJ(), 4)
    DCOPF = Data["DCFlow"]

    MiscData["Objective"] = {1: Objective}
    MiscData["DCOPF"] = {2: DCOPF}


    #Convert the dictionaries to objects for Pandas
    NodeData = pd.DataFrame(data=NodeData)
    ACData = pd.DataFrame(data=ACData)
    DCData = pd.DataFrame(data=DCData)
    MiscData = pd.DataFrame(data=MiscData)

    # Decide what the name of the output file should be
    if Data["DCFlow"]:
        output_file = "DCOPF_results.xlsx"
    else:
        output_file = "ATC_results.xlsx"

    # Store each result in an excel file, given a separate sheet
    with pd.ExcelWriter(output_file) as writer:
        NodeData.to_excel(writer, sheet_name="Node")
        ACData.to_excel(writer, sheet_name="AC")
        DCData.to_excel(writer, sheet_name="DC")
        MiscData.to_excel(writer, sheet_name="Misc")

    print("\n\n")
    print("The results are now stored in the excel file: " + output_file)
    print("This program will now end")

    return()

def plotting(name):
    """
        Reads input from excel file and plots the data using matplotlib.
        Separates between each sheet, and plots them as separate plots
        """

    data = {}  # Dictionary storing the input data

    excel_sheets = ["Node", "AC", "DC", "Misc"]  # Name of each sheet
    data_names = {"Node": "Nodes", "AC": "AC-lines", "DC": "DC-lines", "Misc": "misc"}  # Names for data for each sheet
    num_names = {"Node": "NumNodes", "AC": "NumAC", "DC": "NumDC", "Misc": "NumMisc"}  # Names for numbering
    list_names = {"Node": "NodeList", "AC": "ACList", "DC": "DCList", "Misc": "MiscList"}  # Names for numbering

    # We read each of the sheets and store data in our variable 'data'
    for sheet in excel_sheets:  # For each sheet
        df = pd.read_excel(name, sheet_name=sheet)  # Read sheet
        df = df.set_index(df.columns[0])  # Set first column as index
        num = len(df.loc[:])  # Find length of dataframe
        df = df.to_dict()
        df[num_names[sheet]] = num  # Store length of dataframe in dictionary
        df[list_names[sheet]] = np.arange(0, num)
        data[data_names[sheet]] = df  # Store dataframe in dictionary

    print(data['Nodes'].keys())
    for key in data['Nodes'].keys():
        print(data['Nodes'][key])

    return

#plotting('DCOPF_results.xlsx')

Main()
