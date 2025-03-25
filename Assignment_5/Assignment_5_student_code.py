# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 08:41:35 2022

@author: lukew
"""
from statsmodels.duration.survfunc import survdiff

"""
Created on Wed Jan 30 14:35:13 2019

@author: Kasper E. Thorvaldsen
January 2019

This script reads in data from an excel file about area pricing,
and uses this data to solve an optimization problem
And at the end stores the results in a new excel file

The objective of this code is to show how one can set up a whole problem
from start to finish, using the package Pandas to deal with input data management

This script is divided into three sections:
    - Read the input file (excel file)
    - Use the input file to setup an optimization problem which is solved
    - Store results and then store the results in a new excel file


Below is the mathematical representation of the area pricing problem

Mathematical formulation:

    max z = sum(cons1[n]*MC_cons1[n] for n in Demand Area 1) +
            sum(cons2[n]*MC_cons2[n] for n in Demand Area 2) -
            sum(prod1[n]*MC_prod1[n] for n in Supply Area 1) -
            sum(prod2[n]*MC_prod2[n] for n in Supply Area 2)

    subject to:

        Capacity constraints
            cons1[n] <= Cap_C1[n] for all n in Demand Area 1
            cons2[n] <= Cap_C2[n] for all n in Demand Area 2
            prod1[n] <= Cap_P1[n] for all n in Supply Area 1
            prod2[n] <= Cap_P2[n] for all n in Supply Area 2

        Power Equality

            sum(cons1[n] for n in Demand Area 1) == sum(prod1[n] for n in Supply Area 1) - transfer
            sum(cons2[n] for n in Demand Area 1) == sum(prod"[n] for n in Supply Area 1) + transfer

        Transfer capacity
        -Cap21 <= transfer <= Cap12
"""

import numpy as np
import sys
import time
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def Read_Excel(name):
    """
    Reads input excel file and reads the data into dataframes.
    Separates between each sheet, and stores into one dictionary
    """

    data = {}  # Dictionary storing the input data

    # Dictionaries used to store the data we extract
    # Note that we have 3 types of data we want to get
    Supply = {}
    Demand = {}
    Transfer = {}

    # Specific data here about amount of areas
    # This could instead of being written here have been written in an excel sheet
    #Areas = [1, 2]
    Areas = [1,2,3]

    # For each supplier in the areas
    for sup in Areas:
        # Find the relevant sheet name, and extract the data into the variable df
        # skiprows = 1 is used to ignore the header in the excel sheet
        # we use pandas function to read this, the data will now be stored as a DataFrame
        df = pd.read_excel(name, sheet_name=("Supply_" + str(sup)), skiprows=1, nrows=11,
                           usecols=['Company', 'MC', 'Capacity'])  # , engine='openpyxl')

        # We set the indexing of this dataframe to be based on the 0th column
        # This gives us the key values 1, 2, 3 for what number the producer is
        df = df.set_index(df.columns[0])

        # We store the length of the this dictionary, to find the
        # Number of suppliers present
        num = len(df.loc[:])

        # Now, we convert this variable from a Dataframe
        # To a dictionary
        df = df.to_dict()

        # We store the dictionary in the main dictionary
        Supply[sup] = df

    # We also store the number of suppliers present as a list
    # Note that this method requires that we have equivalent
    # Number of suppliers in both areas
    # semi-ok coding..
    Supply["ListSuppliers"] = np.arange(1, num + 1)

    # We do the same for the demand side
    for dem in Areas:
        # Store the specified demand sheet as a DataFrame
        df = pd.read_excel(name, sheet_name=("Demand_" + str(dem)), skiprows=1, nrows=10,
                           usecols=['Company', 'Utility', 'Capacity'])  # ,engine='openpyxl')

        # Set the indexing to be the 1st column
        df = df.set_index(df.columns[0])

        # Store length of demand
        num = len(df.loc[:])

        # Convert to dictionary
        df = df.to_dict()

        # Store dictionary
        Demand[dem] = df

    # Store number of demand as a list
    Demand["ListDemands"] = np.arange(1, num + 1)

    # Start with transfer data
    # Extract transfer data
    df = pd.read_excel(name, sheet_name="Connection", skiprows=1)  # , engine='openpyxl')

    # change indexing
    df = df.set_index(df.columns[0])

    # Convert to dictionary
    df = df.to_dict()

    # Store dictionary
    Transfer = df

    # Store all dictionaries in a common main dictionary
    data["Supply"] = Supply
    data["Demand"] = Demand
    data["Transfer"] = Transfer
    data["Areas"] = Areas

    return (data)  # Return datasheet



def Opt_model_Area(Data):
    """
    This model creates and solves an optimization problem consisting of area pricing
    And Power exchange between two different areas
    Each area has a number of supplier and demand units that wants to trade power between each other
    Based on their marginal costs

    Also, there is a connection line between the areas so that power can transfer between them
    This line has power constraint in both directions that can differ in quantity


    """

    model = pyo.ConcreteModel()

    """
    Sets:
        List of suppliers
        List of demands
    Note that the list is used in both areas
    So the number of suppliers in area 1 and 2 must be the same

    One could in principle be able to include a list of areas
    However, the main problem is the way the input data is being extracted

    Example: If areas were included, the dictionary for marginal cost for consumers
    must have a 2-dimensional setup, where the row and columns would need to specify areas and consumers

    value_example: MC[A1,1] = 25
    However, in the case our input data is being set up, the marginal cost is 1-dimensional:
        MC_A1[1] = 25
    And to be able to set the optimization model with an area included, the whole input data file
    Must have a new layout. Example, it would probably be better to have an excel-sheet of 
    MC_consumers instead of all data for area1_demand in each sheet. The sheets should specify type of data, not data for an area

    This shows how input data setup is necessary to be done on an accurate layout to provide optimal
    setup.


    """

    # Notice how the sets are not specified on area. This could have been the case,
    # but then you would need to create additional sets for each area individually
    # So this is a simplified method to solve this
    # Set for suppliers. Consist of list of suppliers
    model.S = pyo.Set(initialize=Data["Supply"]["ListSuppliers"])

    # Set for consumers. Consist of list of demand units
    model.D = pyo.Set(initialize=Data["Demand"]["ListDemands"])

    """
    Parameters
    """

    # Marginal cost for suppliers and demands
    # Suppliers. Extract data from the correct area, within the correct dictionary path
    model.Sup1_MC = pyo.Param(model.S, initialize=Data["Supply"][1]["MC"])
    model.Sup2_MC = pyo.Param(model.S, initialize=Data["Supply"][2]["MC"])
    model.Sup3_MC = pyo.Param(model.S, initialize=Data["Supply"][3]["MC"])


    # Demand units MC. Again, be precise about placement of data
    model.Dem1_MC = pyo.Param(model.D, initialize=Data["Demand"][1]["Utility"])
    model.Dem2_MC = pyo.Param(model.D, initialize=Data["Demand"][2]["Utility"])
    model.Dem3_MC = pyo.Param(model.D, initialize=Data["Demand"][3]["Utility"])


    # Capacity for suppliers and demands
    # Supplier capacity
    model.Sup1_Cap = pyo.Param(model.S, initialize=Data["Supply"][1]["Capacity"])
    model.Sup2_Cap = pyo.Param(model.S, initialize=Data["Supply"][2]["Capacity"])
    model.Sup3_Cap = pyo.Param(model.S, initialize=Data["Supply"][3]["Capacity"])

    # Demand capacity
    model.Dem1_Cap = pyo.Param(model.D, initialize=Data["Demand"][1]["Capacity"])
    model.Dem2_Cap = pyo.Param(model.D, initialize=Data["Demand"][2]["Capacity"])
    model.Dem3_Cap = pyo.Param(model.D, initialize=Data["Demand"][3]["Capacity"])
    # We add parameters for the tranmission capacity between areas.
    # Note that both parameters have positive values, we add direction in the code
    # Parameter for direction 1 -> 2
    model.Trans12 = pyo.Param(initialize=Data["Transfer"]["Cap i-j"][1])

    # Parameter for direction 2 -> 1
    model.Trans21 = pyo.Param(initialize=Data["Transfer"]["Cap j-i"][1])

    # Parameter for direction 1 -> 3
    model.Trans13 = pyo.Param(initialize=Data["Transfer"]["Cap i-j"][2])

    # Parameter for direction 3 -> 1
    model.Trans31 = pyo.Param(initialize=Data["Transfer"]["Cap j-i"][2])

    # Parameter for direction 2 -> 3
    model.Trans23 = pyo.Param(initialize=Data["Transfer"]["Cap i-j"][3])

    # Parameter for direction 3 -> 2
    model.Trans32 = pyo.Param(initialize=Data["Transfer"]["Cap j-i"][3])



    """
    Variables
    """

    # Variables for demand, based on number of units
    model.demandA1 = pyo.Var(model.D, within=pyo.NonNegativeReals)
    model.demandA2 = pyo.Var(model.D, within=pyo.NonNegativeReals)
    model.demandA3 = pyo.Var(model.D, within=pyo.NonNegativeReals)


    # Varialbe for supply, based on number of suppliers
    model.supplyA1 = pyo.Var(model.S, within=pyo.NonNegativeReals)
    model.supplyA2 = pyo.Var(model.S, within=pyo.NonNegativeReals)
    model.supplyA3 = pyo.Var(model.S, within=pyo.NonNegativeReals)

    # We include the variable for transmission. We don't need to specify lower/upper bounds yet
    # This will be done as constraints
    #model.transfer = pyo.Var()

    #"""
    model.transfer_12_var = pyo.Var()
    model.transfer_13_var = pyo.Var()
    model.transfer_23_var = pyo.Var()
    
    #"""


    """
    Objective function
    """

    # The objective function is to maximize total surplus (consumer + producer surplus) for both areas
    # Since we have not a set for areas, we must sum up for each area
    # This surplus is found by calculating the area under the demand curve, and subtracting this by the
    # Cost from the producers. This is equivalent to the consumer + producer surplus

    def Objective(model):
        return (sum(model.demandA1[n] * model.Dem1_MC[n] + model.demandA2[n] * model.Dem2_MC[n] + model.demandA3[n] * model.Dem3_MC[n] for n in model.D) - sum(
            model.supplyA1[n] * model.Sup1_MC[n] + model.supplyA2[n] * model.Sup2_MC[n] + model.supplyA3[n] * model.Sup3_MC[n] for n in model.S))

    model.OBJ = pyo.Objective(rule=Objective, sense=pyo.maximize)

    """
    Constraints
    """

    # Power equality area 1.
    # Note the minus sign on the model.transfer variable
    # By using this negative sign, we then say that transmission from area 1->2 is positive direction (same for 1->3)
    # Power equality is in this case sum of consumption is equal to sum of production - transmission amount
    def PowerEq1(model):
        #return (sum(model.demandA1[n] for n in model.D) == sum(model.supplyA1[n] for n in model.S) - model.transfer)
        return (sum(model.demandA1[n] for n in model.D) == sum(model.supplyA1[n] for n in model.S) - model.transfer_12_var - model.transfer_13_var)

    model.PE1_const = pyo.Constraint(rule=PowerEq1)


    # Power equality area 2 in connection with area 1 and area 3
    # Power equality is sum of consumption Equal to sum of production + transmission amount
    def PowerEq2(model):
        return (sum(model.demandA2[n] for n in model.D) == sum(model.supplyA2[n] for n in model.S) + model.transfer_12_var - model.transfer_23_var)

    model.PE2_const = pyo.Constraint(rule=PowerEq2)

    # Power equality area 3 in connection with area 1 and area 2
    # Power equality is sum of consumption Equal to sum of production + transmission amount
    def PowerEq3(model):
        return (sum(model.demandA3[n] for n in model.D) == sum(model.supplyA3[n] for n in model.S) + model.transfer_13_var + model.transfer_23_var)

    model.PE3_const = pyo.Constraint(rule=PowerEq3)


    # Capacity constraint
    # Suppliers in area 1 cannot produce more than their capacity
    def SupplyMax1(model, n):
        return (model.supplyA1[n] <= model.Sup1_Cap[n])

    model.SM1 = pyo.Constraint(model.S, rule=SupplyMax1)

    # Capacity constraint
    # Suppliers in area 2 cannot produce more than their capacity
    def SupplyMax2(model, n):
        return (model.supplyA2[n] <= model.Sup2_Cap[n])


    # Capacity constraint
    # Suppliers in area 3 cannot produce more than their capacity

    model.SM2 = pyo.Constraint(model.S, rule=SupplyMax2)

    def SupplyMax3(model, n):
        return (model.supplyA3[n] <= model.Sup3_Cap[n])

    model.SM3 = pyo.Constraint(model.S, rule=SupplyMax3)

    # Capacity constraint
    # Demand units in area 1 cannot consume more than their capacity
    def DemandMax1(model, n):
        return (model.demandA1[n] <= model.Dem1_Cap[n])

    model.DM1 = pyo.Constraint(model.D, rule=DemandMax1)

    # Capacity constraint
    # Demand units in area 2 cannot consume more than their capacity
    def DemandMax2(model, n):
        return (model.demandA2[n] <= model.Dem2_Cap[n])

    model.DM2 = pyo.Constraint(model.D, rule=DemandMax2)

    # Capacity constraint
    # Demand units in area 3 cannot consume more than their capacity
    def DemandMax3(model, n):
        return (model.demandA3[n] <= model.Dem3_Cap[n])

    model.DM3 = pyo.Constraint(model.D, rule=DemandMax3)

    # Transfer constraints

    # We set the upper boundary to be the maximum transmission capacity from area 1 to area 2
    # By doing this, we again specify that area 1 -> 2 is the positive direction
    def T12Cap(model):
        return (model.transfer_12_var <= model.Trans12)

    model.T12 = pyo.Constraint(rule=T12Cap)

    # Likewise, we specify the lower boundary of the transmission capacity
    # We say that transmission capacity cannot be lower than the negative capacity from area 2 -> 1
    # By having a negative transmission value, power flows from area 2 to 1
    def T21Cap(model):
        return (model.transfer_12_var >= -model.Trans21)

    model.T21 = pyo.Constraint(rule=T21Cap)

    # We again specify that area 1 -> 3 is the positive direction
    def T13Cap(model):
        return (model.transfer_13_var <= model.Trans13)

    model.T13 = pyo.Constraint(rule=T13Cap)

    # power flows from area 3 to 1
    def T31Cap(model):
        return (model.transfer_13_var >= -model.Trans31)

    model.T31 = pyo.Constraint(rule=T31Cap)

    # We again specify that area 2 -> 3 is the positive direction
    def T23Cap(model):
        return (model.transfer_23_var <= model.Trans23)

    model.T23 = pyo.Constraint(rule=T23Cap)

    # power flows from area 3 to 2
    def T32Cap(model):
        return (model.transfer_23_var >= -model.Trans32)

    model.T32 = pyo.Constraint(rule=T32Cap)



    # Solver to use
    opt = SolverFactory("gurobi")

    # Allow finding dual data
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Solve the problem
    results = opt.solve(model, load_solutions=True)

    # Display results
    model.display()
    model.dual.display()

    # We use the function StoreResults to store data in an excel file
    StoreResults(Data,model)
    PrintResults(Data, model)

    return ()


def StoreResults(Data,model):


    #This function takes in a data dictionary, and an optimization model that has been solved
    #This function will then store relevant information to the dictionary, and store all data in an Excel-file

    #Note that this function is very raw data management, and could be improved heavily. The author encourages
    #feedback on this for better solutions for this!



    #We create dictionaries that will store data from each area
    #The data here is results from the suppliers, including amount supplied and the price for production
    #Note that the price will be the same for all units in each area.
    SA1_supply  = {}
    SA1_price   = {}
    SA2_supply  = {}
    SA2_price   = {}
    SA3_supply  = {}
    SA3_price   = {}


    #For each supplier
    for sup in model.S:

        SA1_supply[sup]  = model.supplyA1[sup].value                        #Store supply amount for area 1 supplier 'sup'
        SA1_price[sup]   = abs(round(model.dual[model.PE1_const],3))        #Store market price for area 1 supplier 'sup
        SA2_supply[sup]  = model.supplyA2[sup].value                        #Store supply amount for area 2 supplier 'sup'
        SA2_price[sup]   = abs(round(model.dual[model.PE2_const],3))        #Store market price for area 2 supplier 'sup
        SA3_supply[sup]  = model.supplyA3[sup].value                        #Store supply amount for area 3 supplier 'sup'
        SA3_price[sup]   = abs(round(model.dual[model.PE3_const],3))        #Store market price for area 3 supplier 'sup


    #We do the same we did for the suppliers to the demand units
    #Create dictionaries that store information regarding the demand units
    DA1_demand  = {}
    DA1_price   = {}
    DA2_demand  = {}
    DA2_price   = {}
    DA3_demand  = {}
    DA3_price   = {}

    #For each demand unit
    for dem in model.D:

        DA1_demand[dem]  = model.demandA1[dem].value                        #Store demand amount for area 1 demand unit 'dem'
        DA1_price[dem]   = abs(round(model.dual[model.PE1_const],3))        #Store market price for area 1 demand unit 'dem'
        DA2_demand[dem]  = model.demandA2[dem].value                        #Store demand amount for area 2 demand unit 'dem'
        DA2_price[dem]   = abs(round(model.dual[model.PE2_const],3))        #Store market price for area 2 demand unit 'dem'
        DA3_demand[dem]  = model.demandA3[dem].value                        #Store demand amount for area 3 demand unit 'dem'
        DA3_price[dem]   = abs(round(model.dual[model.PE3_const],3))        #Store market price for area 3 demand unit 'dem'


    #Then, we store all of these results in the Data dictionary
    #We make sure to store the results in the corresponding area

    #Store production amount for area 1 and 2 and 3
    Data["Supply"][1]["supplied"] = SA1_supply
    Data["Supply"][2]["supplied"] = SA2_supply
    Data["Supply"][3]["supplied"] = SA3_supply

    #Store market price for suppliers in area 1 and 2 and 3
    Data["Supply"][1]["price"] = SA1_price
    Data["Supply"][2]["price"] = SA2_price
    Data["Supply"][3]["price"] = SA3_price


    #Store consumption amount for area 1 and 2 and 3
    Data["Demand"][1]["Bought"] = DA1_demand
    Data["Demand"][2]["Bought"] = DA2_demand
    Data["Demand"][3]["Bought"] = DA3_demand


    #Store market price for demand units in area 1 and 2 and 3
    Data["Demand"][1]["price"] = DA1_price
    Data["Demand"][2]["price"] = DA2_price
    Data["Demand"][3]["price"] = DA3_price


    #Then, we store transmission data in the library
    #We want information regarding amount, and also dual values on the constraints
    #The dual values will help us see the benefit for more transmission capacity
    #This dual value will also tell us the difference in price between the areas
    #If there is a price difference, there is a benefit for the transmission operators to gain
    #For instance, if the price in area 1 is 250 NOK/MWh, and area 2 has 400 NOK/MWh,
    #Then the transmission operator will basically earn 150 NOK/MWh as benefit for transferring this energy
    #Buying for 250 and selling for 400
    #Therefore, we also include the benefit the operator has for sending power

    Data["Transfer"]["Amount i_j"] = {1:model.transfer_12_var.value, 2:model.transfer_13_var.value, 3:model.transfer_23_var.value}                                  #Store power sent between areas
    Data["Transfer"]["T_i_j dual"] = {1:abs(round(model.dual[model.T12],3)), 2:abs(round(model.dual[model.T13],3)), 3:abs(round(model.dual[model.T23],3))}          #Store dual benefit transmission i -> j
    Data["Transfer"]["T_j_i dual"] = {1:abs(round(model.dual[model.T21],3)), 2:abs(round(model.dual[model.T31],3)), 3:abs(round(model.dual[model.T32],3))}          #Store dual benefit transmission j -> i
    Data["Transfer"]["Benefit T_i_j"] = {1:model.transfer_12_var.value*abs(round(model.dual[model.T12],3)), 2:model.transfer_13_var.value*abs(round(model.dual[model.T13],3)), 3:model.transfer_23_var.value*abs(round(model.dual[model.T23],3))}      #Store total benefit for transmission i -> j
    Data["Transfer"]["Benefit T_j_i"] = {1:model.transfer_12_var.value*abs(round(model.dual[model.T21],3)), 2:model.transfer_13_var.value*abs(round(model.dual[model.T31],3)), 3:model.transfer_23_var.value*abs(round(model.dual[model.T32],3))}      #Store total benefit for transmission j -> i

    #Convert to pandas
    #We store each of the dictionary sections as a pandas Dataframe, so we can
    #Store the data in an excel-file

    A1Sup = pd.DataFrame(data=Data["Supply"][1])        #Store supplier data for area 1
    A2Sup = pd.DataFrame(data=Data["Supply"][2])        #Store supplier data for area 2
    A3Sup = pd.DataFrame(data=Data["Supply"][3])        #Store supplier data for area 3
    A1Dem = pd.DataFrame(data=Data["Demand"][1])        #Store demand data for area 1
    A2Dem = pd.DataFrame(data=Data["Demand"][2])        #Store demand data for area 2
    A3Dem = pd.DataFrame(data=Data["Demand"][3])        #Store demand data for area 3
    Trans = pd.DataFrame(data=Data["Transfer"])         #Store transmission data between area 1 and 2


    #Then, we start storing this data into the output excel file
    #Note that the excel file can exist before doing this procedure, if it doesn't it will be created


    with pd.ExcelWriter("output_three_Areas_0.xlsx") as writer:              #Set the location for excel file as the variable name 'writer'

        A1Sup.to_excel(writer, sheet_name= "Area 1 supplier")       #Store supplier data for area 1 in the excel sheet
        A2Sup.to_excel(writer, sheet_name= "Area 2 supplier")       #Store supplier data for area 2 in the excel sheet
        A3Sup.to_excel(writer, sheet_name= "Area 3 supplier")       #Store supplier data for area 3 in the excel sheet
        A1Dem.to_excel(writer, sheet_name= "Area 1 demand")         #Store demand data for area 1 in the excel sheet
        A2Dem.to_excel(writer, sheet_name= "Area 2 demand")         #Store demand data for area 2 in the excel sheet
        A3Dem.to_excel(writer, sheet_name= "Area 3 demand")         #Store demand data for area 3 in the excel sheet
        Trans.to_excel(writer, sheet_name= "Transfer data")         #Store transmission data in the excel sheet

    #Return as we are done with this
    return()
    #"""


def PrintResults(Data, m):
    print(f'RESULTS:')
    # Quantity is sum of all producers producing:
    q_traded = sum(m.supplyA1.extract_values().values()) + sum(m.supplyA2.extract_values().values()) + sum(
        m.supplyA3.extract_values().values())
    print(f'Quantity traded: {round(q_traded)} MWh')
    price_A1 = abs(m.dual[m.PE1_const])
    print(f'Market price area 1: {price_A1} NOK/MWh')
    price_A2 = abs(m.dual[m.PE2_const])
    print(f'Market price area 2: {price_A2} NOK/MWh')
    price_A3 = abs(m.dual[m.PE3_const])
    print(f'Market price area 3: {price_A3} NOK/MWh')

    print(f'Quantity transferred: {round(m.transfer_12_var.value)} MWh')
    print(f'Quantity transferred: {round(m.transfer_13_var.value)} MWh')
    print(f'Quantity transferred: {round(m.transfer_23_var.value)} MWh')
	
	# TODO insert your code here




data = Read_Excel("three_Area_Input.xlsx")
Opt_model_Area(data)
#print(data)