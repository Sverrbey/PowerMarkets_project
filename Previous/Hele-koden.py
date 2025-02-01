# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:32:32 2024

@author: aasordal
"""

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np

# Load the Excel file
FILE = "Problem 2 data.xlsx"

# Importing the sheet name from excel file
sheet = 'Problem 2.2 - Base case'

x = 0  # Assigning a number to each sheet, in order to distinguish the different cases
if sheet == 'Problem 2.2 - Base case':
    x = 1
elif sheet == 'Problem 2.3 - Generators':
    x = 2
elif sheet == 'Problem 2.5 - Environmental':
    x = 3

def extract_data(sheet):
    # Load Excel file into a pandas DataFrame
    df = pd.read_excel(FILE, sheet, header=None, index_col=None)

    # Select the range of cells containing the generator data
    # Then making a list for all the relevant information from the excel file
    generator_data = None
    if x == 1:
        generator_data = df.iloc[3:6, 0:5]
    elif x == 2:
        generator_data = df.iloc[3:8, 0:5]
    elif x == 3:
        generator_data = df.iloc[3:8, 0:6]

    Generator = generator_data.loc[:, 0].values.tolist()
    Capacity_G = generator_data.loc[:, 1].values.tolist()
    Marginal_cost = generator_data.loc[:, 2].values.tolist()
    Location_G = generator_data.loc[:, 3].values.tolist()
    
    Slack_bus = None
    Emissions = None
    if x == 1 or x == 2:
        Slack_bus = generator_data.loc[:, 4].values.tolist()
    elif x == 3:   
        Emissions = generator_data.loc[:, 4].values.tolist()
        Slack_bus = generator_data.loc[:, 5].values.tolist()

    # Select the range of cells that contain the load data
    load_data = None
    Location_Lo = None
    Marginal_cost_Dem = None
    if x == 1 or x == 2:
        load_data = df.iloc[3:6, 9: 12]
        Location_Lo = load_data.iloc[:, 2].values.tolist()
    elif x == 3:
        load_data = df.iloc[3:8, 10: 14]
        Marginal_cost_Dem = load_data.iloc[:, 2].values.tolist()
        Location_Lo = load_data.iloc[:, 3].values.tolist()

    Load_unit = load_data.iloc[:, 0].values.tolist()
    Demand = load_data.iloc[:, 1].values.tolist()

    # Select the range of cells that contain the transmission line data
    Trasnsmission_line_data = None
    if x == 1 or x == 2:
        Trasnsmission_line_data = df.iloc[3:6, 15:18]
    elif x == 3:
        Trasnsmission_line_data = df.iloc[3:6, 17:20]

    Line =  Trasnsmission_line_data.iloc[:, 0].values.tolist()
    Capacity_Li = Trasnsmission_line_data.iloc[:, 1].values.tolist()
    Admittance = Trasnsmission_line_data.iloc[:, 2].values.tolist()
    

    # making the Y matrix        

    n = 3 #number of nodes
    
    Ybus = np.zeros((n, n))
    
    #Finding the off diagonal elements in the Ybus
    
    Y12 = Admittance[0]
    Y13 = Admittance[1]
    Y23 = Admittance[2]
    
    #Calculating the diagonal elements in the Ybus
    
    Y11 = -Y12 -Y13
    Y22 = -Y12 -Y23
    Y33 = -Y13 -Y23
    
    # Adding the elements to the Y matrix
    
    Ybus[0][1] = Ybus[1][0] = Y12
    Ybus[0][2] = Ybus[2][0] = Y13
    Ybus[1][2] = Ybus[2][1] = Y23
    
    Ybus[0][0] = Y11
    Ybus[1][1] = Y22
    Ybus[2][2] = Y33
    
    # Removing the row and the column of the slack bus
    
    Y_del_col = np.delete(Ybus,0, 1) # removing the column that corresponds to the slack bus
    Y = np.delete(Y_del_col,0, 0) # removing the row that corresponds to the slack bus
    
    Y = Y*-1
    
    # Finding the F matrix
    
    Fbus = np.zeros((n, n))
    
    Fbus[0][0] = Admittance [0]
    Fbus[1][0] = Admittance [1]
    Fbus[2][1] = Admittance [2]
    
    Fbus[0][1] = -Admittance [0]
    Fbus[1][2] = -Admittance [1]
    Fbus[2][2] = -Admittance [2]
    
    # Removing the column of the slack bus
    
    F = np.delete(Fbus,0, 1) 
    
    # returning the relevant info
    
    if x == 1 or x == 2:
        return Generator, Capacity_G, Marginal_cost, Location_G, Slack_bus, Load_unit, Demand, Location_Lo, Line, Capacity_Li, Admittance, F, Y
    elif x == 3:
        return Generator, Capacity_G, Marginal_cost, Location_G, Slack_bus, Load_unit, Demand, Location_Lo, Line, Capacity_Li, Admittance, F, Y, Emissions, Marginal_cost_Dem

def Power_market():
    
    data = extract_data(sheet) # using the defined function do extract the relevant data
    
    dict_1 = { # making a dict_1ionary to make it easier to obtain the rigth data
    'Generator': 0,'Capacity_G': 1,'Marginal_cost': 2,
    'Location_G': 3,'Slack_bus': 4,'Load_unit': 5,
    'Demand': 6,'Location_Lo': 7,'Line': 8,
    'Capacity_Li': 9,'Admittance': 10,'F': 11,'Y': 12,
    'Emissions': 13 ,'Marginal_cost_Dem' : 14
    }

    
    model = pyo.ConcreteModel() # creating the model
    
    # defining the variables
    
    model.P1 = pyo.Var(within = pyo.NonNegativeReals)
    model.P2 = pyo.Var(within = pyo.NonNegativeReals)
    model.P3 = pyo.Var(within = pyo.NonNegativeReals)
    
    # Defining new variables for each of the generators at node 1
    model.P1_1 = None
    model.P1_2 = None
    model.P1_3 = None

    # Defining the different demands at node 2 as variables and one for the total
    model.D2_1 = None
    model.D2_2 = None
    model.D2_3 = None
    model.D2 = None
    
    # defining new variables for each of the generators at node 1
    
    if x == 2 or x == 3:
        model.P1_1 =pyo.Var(within = pyo.NonNegativeReals)
        model.P1_2 =pyo.Var(within = pyo.NonNegativeReals)
        model.P1_3 =pyo.Var(within = pyo.NonNegativeReals)
    
    # defining the differnt demands at node 2 as variables and one for the total 
    
    if x == 3:
    
        model.D2_1 = pyo.Var(within = pyo.NonNegativeReals)
        model.D2_2 = pyo.Var(within = pyo.NonNegativeReals)
        model.D2_3 = pyo.Var(within = pyo.NonNegativeReals)
        
        model.D2 = pyo.Var(within = pyo.NonNegativeReals)
        
    # defining the angles for node 2 and 3
    
    #delta1 = 0
    model.delta2 = pyo.Var(within=pyo.Reals)
    model.delta3 = pyo.Var(within=pyo.Reals)


    def objective(model): 
        
        # Defing objective to minimize production cost for base case and the extended case with multiple generators
        if x == 1:
            return model.P1 * data[dict_1['Marginal_cost']][0] + model.P2 * data[dict_1['Marginal_cost']][1] + model.P3 * data[dict_1['Marginal_cost']][2] 
        elif x == 2:
            return model.P1_1 * data[dict_1['Marginal_cost']][0] + model.P1_2 * data[dict_1['Marginal_cost']][1] + model.P1_3 * data[dict_1['Marginal_cost']][2] + model.P2 * data[dict_1['Marginal_cost']][3] + model.P3 *data[dict_1['Marginal_cost']][4]
        elif x == 3:
        # Defining objective to maximize social welfare for the extended case with environmental constraints
            revenue = model.D2_1 * data[dict_1['Marginal_cost_Dem']][1] + model.D2_2 *data[dict_1['Marginal_cost_Dem']][2] + model.D2_3 * data[dict_1['Marginal_cost_Dem']][3]
        
            cost = model.P1_1 * data[dict_1['Marginal_cost']][0] + model.P1_2 * data[dict_1['Marginal_cost']][1] + model.P1_3 * data[dict_1['Marginal_cost']][2] + model.P2*data[dict_1['Marginal_cost']][3] + model.P3*data[dict_1['Marginal_cost']][4]

            return (revenue-cost)
    
    if x == 1 or x == 2: # minimize for the first two cases
        model.OBJ = pyo.Objective(rule = objective, sense = pyo.minimize)
    elif x == 3: # maximize for the third extended case
        model.OBJ = pyo.Objective(rule = objective, sense = pyo.maximize)
    
    
    if x == 3:
        
        # Environmental constraints, comment out the CES and uncomment Cap and trade to run the model with cap and trade 
        # CES  
        
        def P_Emi_cap(model):
            return (model.P2 == 0.2*(model.P1 + model.P2 + model.P3))
        model.P_Emi_const = pyo.Constraint(rule = P_Emi_cap)
        
        #Cap and trade
        
        # def P_Emi_cap(model):
        #     return (model.P1_1 * data[dict_1['Emissions']][0] + model.P1_2 * data[dict_1['Emissions']][1] + model.P1_3 * data[dict_1['Emissions']][2] + model.P2 * data[dict_1['Emissions']][3] + model.P3 * data[dict_1['Emissions']][4] <= 950000)
        # model.P_Emi_const = pyo.Constraint(rule = P_Emi_cap)
    
    
    # Capacity limits for the power generated 
        
    if x == 1:
        # P1 < P1_max 
        def P1_cap(model): 
            return (1 * model.P1 <= data[dict_1['Capacity_G']][0]    )
        model.P1_const = pyo.Constraint(rule = P1_cap)
        
        # P2 < P2_max

        def P2_cap(model):
            return (1 * model.P2 <= data[dict_1['Capacity_G']][1]    )
        model.P2_const = pyo.Constraint(rule = P2_cap)

        # P3 < P3_max

        def P3_cap(model):
            return (1 * model.P3 <= data[dict_1['Capacity_G']][2]    )
        model.P3_const = pyo.Constraint(rule = P3_cap)
    
    elif x == 2 or x == 3:
        
        # Making sure that the total production at node 1 (P1), equals the sum of the generators at node 1
        
        def P1_cap(model):
            return (model.P1 == model.P1_1 + model.P1_2 + model.P1_3)
        model.P1_const = pyo.Constraint(rule = P1_cap)

        # P1_1 < P1_1_max  
        def P1_1_cap(model): 
            return (1 * model.P1_1 <= data[dict_1['Capacity_G']][0]    )
        model.P1_1_const = pyo.Constraint(rule = P1_1_cap)
        
        # P1_2 < P1_2_max
    
        def P1_2_cap(model): 
            return (1 * model.P1_2 <= data[dict_1['Capacity_G']][1]    )
        model.P1_2_const = pyo.Constraint(rule = P1_2_cap)
        
        # P1_3 < P1_3_max
    
        def P1_3_cap(model): 
            return (1 * model.P1_3 <= data[dict_1['Capacity_G']][2]    )
        model.P1_3_const = pyo.Constraint(rule = P1_3_cap)

        # P2 < P2_max
    
        def P2_cap(model):
            return (1 * model.P2 <= data[dict_1['Capacity_G']][3]    )
        model.P2_const = pyo.Constraint(rule = P2_cap)
    
        # P3 < P3_max
    
        def P3_cap(model):
            return (1 * model.P3 <= data[dict_1['Capacity_G']][4]    )
        model.P3_const = pyo.Constraint(rule = P3_cap)

    # Making sure the production at the slack bus covers the rest of the demand
    
    if x == 1 or x == 2:
        def Pslack_cap(model):
            return (model.P1 == data[dict_1['Demand']][0] + (data[dict_1['Demand']][1]- model.P2) + (data[dict_1['Demand']][2] - model.P3) )
        model.Pslack_const = pyo.Constraint(rule = Pslack_cap)
    
    elif x == 3:
    
    # Making sure the production at the slack bus covers the rest of the demand with D2 as a variable     
        def Pslack_cap(model):
            return (model.P1 == data[dict_1['Demand']][0] + (model.D2 - model.P2) + (data[dict_1['Demand']][4] - model.P3) )
        model.Pslack_const = pyo.Constraint(rule = Pslack_cap)
        
        # Capacity limits for the demand variables  
        
        def D2_1_cap(model):
            return (model.D2_1 <= data[dict_1['Demand']][1])
        model.D2_1_const = pyo.Constraint(rule = D2_1_cap)
        
        def D2_2_cap(model):
            return (model.D2_2 <= data[dict_1['Demand']][2])
        model.D2_2_const = pyo.Constraint(rule = D2_2_cap)
        
        def D2_3_cap(model):
            return (model.D2_3 <= data[dict_1['Demand']][3])
        model.D2_3_const = pyo.Constraint(rule = D2_3_cap)
        
        # Making sure the total demand D2 equals the sum of the demands covered at node 2
        
        def D2_eq_cap(model):
            return (model.D2 == model.D2_1 + model.D2_2 + model.D2_3)
        model.D2_eq_const = pyo.Constraint(rule = D2_eq_cap)
    
    
        # Demand cover for node 2
            
        def D2_cap(model):
            return (model.P2/100 - model.D2 /100== data[dict_1['Y']][0][0]*model.delta2 + data[dict_1['Y']][0][1]*model.delta3 )
        model.D2_const = pyo.Constraint(rule = D2_cap)
        
        # Demand cover for node 3
        
        def D3_cap(model):
            return (model.P3/100 - data[dict_1['Demand']][4]/100 == data[dict_1['Y']][1][0]*model.delta2 + data[dict_1['Y']][1][1]*model.delta3 )
        model.D3_const = pyo.Constraint(rule = D3_cap)

    if x == 1 or x == 2:
        
        # Demand cover for node 2
        
        def D2_cap(model):
            return (model.P2/100 - data[dict_1['Demand']][1] /100== data[dict_1['Y']][0][0]*model.delta2 + data[dict_1['Y']][0][1]*model.delta3 )
        model.D2_const = pyo.Constraint(rule = D2_cap)

         # Demand cover for node 3
     
        def D3_cap(model):
            return (model.P3/100 - data[dict_1['Demand']][2]/100 == data[dict_1['Y']][1][0]*model.delta2 + data[dict_1['Y']][1][1]*model.delta3 )
        model.D3_const = pyo.Constraint(rule = D3_cap)

     


    # Keeping the transmission within its capacity limits
    
    # Transmission line 1-2
    
    def Tran12min_cap(model):
        return(-data[dict_1['Capacity_Li']][0]/100 <= data[dict_1['F']][0][0]* model.delta2)
    model.Tranmin12_const = pyo.Constraint(rule = Tran12min_cap)
    
    def Tran12max_cap(model):
        return( data[dict_1['F']][0][0]* model.delta2  <= data[dict_1['Capacity_Li']][0]/100)
    model.Tran12max_const = pyo.Constraint(rule = Tran12max_cap)
   
    # Transmission line 1-3
     
    def Tran13min_cap(model):
        return(-data[dict_1['Capacity_Li']][1]/100 <=  data[dict_1['F']][1][1]* model.delta3)
    model.Tran13min_const = pyo.Constraint(rule = Tran13min_cap)
    
    def Tran13max_cap(model):
        return( data[dict_1['F']][1][1]* model.delta3 <= data[dict_1['Capacity_Li']][1]/100)
    model.Tran13max_const = pyo.Constraint(rule = Tran13max_cap)

     
    # Transmission line 2-3
    
    def Tran23min_cap(model):
        return(-data[dict_1['Capacity_Li']][2]/100 <= data[dict_1['F']][2][0]* model.delta2 + data[dict_1['F']][2][1]* model.delta3)
    model.Tran23min_const = pyo.Constraint(rule = Tran23min_cap)

    def Tran23max_cap(model):
        return(data[dict_1['F']][2][0]* model.delta2 + data[dict_1['F']][2][1]* model.delta3 <= data[dict_1['Capacity_Li']][2]/100)
    model.Tran23max_const = pyo.Constraint(rule = Tran23max_cap)


    opt = SolverFactory('gurobi', solver_io = "python")


    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
     
    results = opt.solve(model, load_solutions = True)
     
     # Display
    model.display()
    
    model.dual.display()
    #model.pprint()
     
Power_market()   


