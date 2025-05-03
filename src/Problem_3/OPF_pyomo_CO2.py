# -*- coding: utf-8 -*-
"""

Optimal Power Flow with DC power flow
   =====================================

  (c) Gerard Doorman, December 2012
      Hossein farahmand, February 2016

Originally implemented for MOSEL XPRESS

Converted to Python/Pyomo for semester 2018/2019:

  (c) Kasper Emil Thorvaldsen, December 2018

  
Utilized to solve the course project for TET4185 Power Markets
    (c) Bastian Øie, May 2025
        Sverre Beyer, May 2025
        Aurora Vinslid, May 2025
"""

import numpy as np
import sys
import time
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

def OPF_model_CO2(Data):
    
    """
    Set up the optimization model, run it and store the data in a .xlsx file
    """
    
    
    model = pyo.ConcreteModel() #Establish the optimization model, as a concrete model in this case


    """
    Sets
    """    
    model.L = pyo.Set(ordered = True, initialize = Data["AC-lines"]["ACList"])  #Set for AC lines
    
    model.N = pyo.Set(ordered = True, initialize = Data["Nodes"]["NodeList"])   #Set for nodes
    
    model.H = pyo.Set(ordered = True, initialize = Data["DC-lines"]["DCList"])  #Set for DC lines
    
    """Parameters"""
    
    #Nodes
    
    model.Demand    = pyo.Param(model.N, initialize = Data["Nodes"]["DEMAND"])  #Parameter for demand for every node
    
    model.P_min     = pyo.Param(model.N, initialize = Data["Nodes"]["GENMIN"])  #Parameter for minimum production for every node
    
    model.P_max     = pyo.Param(model.N, initialize = Data["Nodes"]["GENCAP"])  #Parameter for max production for every node
    
    model.Cost_gen  = pyo.Param(model.N, initialize = Data["Nodes"]["GENCOST"]) #Parameter for generation cost for every node
    
    model.Cost_shed = pyo.Param(initialize = Data["ShedCost"])                  #Parameter for cost of shedding power
    
    model.Pu_base   = pyo.Param(initialize = Data["pu-Base"])                   #Parameter for per unit factor

    model.EmissionValue = pyo.Param(model.N, initialize = Data["Nodes"]["EmissionValue"])         #Parameter for emission value for every node  
    
    model.Cost_emission = pyo.Param(initialize = Data["Emissions_cost"])                      #Parameter for cost of emissions
    
    #AC-lines
   
    model.P_AC_max  = pyo.Param(model.L, initialize = Data["AC-lines"]["Cap From"])     #Parameter for max transfer from node, for every line
    
    model.P_AC_min  = pyo.Param(model.L, initialize = Data["AC-lines"]["Cap To"])       #Parameter for max transfer to node, for every line
    
    model.AC_from   = pyo.Param(model.L, initialize = Data["AC-lines"]["From"])         #Parameter for starting node for every line
    
    model.AC_to     = pyo.Param(model.L, initialize = Data["AC-lines"]["To"])           #Parameter for ending node for every line
    
    #DC-lines
    
    model.DC_cap    = pyo.Param(model.H, initialize = Data["DC-lines"]["Cap"])          #Parameter for Cable capacity for every cable
    
    
    
    """
    Variables
    """
    
    #Nodes
    model.theta     = pyo.Var(model.N)                                      #Variable for angle on bus for every node
    
    model.gen       = pyo.Var(model.N)                                      #Variable for generated power on every node
    
    model.shed      = pyo.Var(model.N, within = pyo.NonNegativeReals)       #Variable for shed power on every node
    
    model.e         = pyo.Var(model.N, within=pyo.NonNegativeReals)         #Variable for emission on every node
    #AC-lines
    
    model.flow_AC   = pyo.Var(model.L)                                      #Variable for power flow on every line
    
    #DC-lines
    
    model.flow_DC   = pyo.Var(model.H)                                      #Variable for power flow on every cable
    
    
    """
    Objective function
    Minimize cost associated with production and shedding of generation
    """
    
    def ObjRule(model): #Define objective function
        return ( sum(model.gen[n]*model.Cost_gen[n] for n in model.N) + \
                sum(model.shed[n]*model.Cost_shed for n in model.N)  + \
                sum(model.gen[n]*model.Cost_emission*model.EmissionValue[n] for n in model.N) )
    model.OBJ       = pyo.Objective(rule = ObjRule, sense = pyo.minimize)   #Create objective function based on given function
    
    
    """
    Constraints
    """
    
    #Emission Constraints
    def produced_emission(model,n):
        return model.e[n] == model.gen[n]*model.EmissionValue[n] 
    model.produced_emission = pyo.Constraint(model.N, rule = produced_emission)

    #Minimum generation
    #Every generating unit must provide at least the minimum capacity
    
    def Min_gen(model,n):
        return(model.gen[n] >= model.P_min[n])
    model.Min_gen_const = pyo.Constraint(model.N, rule = Min_gen)
    
    #Maximum generation
    #Every generating unit cannot provide more than maximum capacity

    def Max_gen(model,n):
        return(model.gen[n] <= model.P_max[n])
    model.Max_gen_const = pyo.Constraint(model.N, rule = Max_gen)
    
    #Maximum from-flow line
    #Sets the higher gap of line flow from unit n
    
    def From_flow(model,l):
        return(model.flow_AC[l] <= model.P_AC_max[l])
    model.From_flow_L = pyo.Constraint(model.L, rule = From_flow)
    
    #Maximum to-flow line
    #Sets the higher gap of line flow to unit n (given as negative flow)
    
    def To_flow(model,l):
        return(model.flow_AC[l] >= -model.P_AC_min[l])
    model.To_flow_L = pyo.Constraint(model.L, rule = To_flow)
    
    #Maximum from-flow cable
    #Sets the higher gap of cable flow from unit n
    
    def FlowBalDC_max(model,h):
        return(model.flow_DC[h] <= model.DC_cap[h])
    model.FlowBalDC_max_const = pyo.Constraint(model.H, rule = FlowBalDC_max)
    
    #Maximum to-flow cable
    #Sets the higher gap of cable flow to unit n (given as negative flow)
    
    def FlowBalDC_min(model,h):
        return(model.flow_DC[h] >= -model.DC_cap[h])
    model.FlowBalDC_min_const = pyo.Constraint(model.H, rule = FlowBalDC_min)
    
    
    #If we want to run the model using DC Optimal Power Flow
    if Data["DCFlow"] == True:
        
        #Set the reference node to have a theta == 0
        
        def ref_node(model):
            return(model.theta[Data["Reference node"]] == 0)
        model.ref_node_const = pyo.Constraint(rule = ref_node)
        
        
        #Loadbalance; that generation meets demand, shedding, and transfer from lines and cables
        
        def LoadBal(model,n):
            return(model.gen[n] + model.shed[n] == model.Demand[n] +\
            sum(Data["B-matrix"][n-1][o-1]*model.theta[o]*model.Pu_base for o in model.N) + \
            sum(Data["DC-matrix"][h-1][n-1]*model.flow_DC[h] for h in model.H))
        model.LoadBal_const = pyo.Constraint(model.N, rule = LoadBal)
        
        #Flow balance; that flow in line is equal to change in phase angle multiplied with the admittance for the line
        
        def FlowBal(model,l):
            return(model.flow_AC[l]/model.Pu_base == ((model.theta[model.AC_from[l]]- model.theta[model.AC_to[l]])*-Data["B-matrix"][model.AC_from[l]-1][model.AC_to[l]-1]))
        model.FlowBal_const = pyo.Constraint(model.L, rule = FlowBal)
        
        
        
        
        
        
    else:           #If we are to run this using ATC-rules
        
        #Loadbalance; that generation meets demand, shedding, and transfer from lines and cables
        
        def LoadBal(model,n):
            return( model.gen[n] + model.shed[n] == model.Demand[n] +\
                   sum(Data["X-matrix"][l-1][n-1]*model.flow_AC[l] for l in model.L) + \
                   sum(Data["DC-matrix"][h-1][n-1]*model.flow_DC[h] for h in model.H)
                   )
        model.LoadBal_const = pyo.Constraint(model.N, rule = LoadBal)
        
        
        
    """
    Compute the optimization problem
    """
        
    #Set the solver for this
    opt         = SolverFactory("gurobi")
    
    
    #Enable dual variable reading -> important for dual values of results
    model.dual      = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    
    
    #Solve the problem
    results     = opt.solve(model, load_solutions = True)
    
    #Write result on performance
    results.write(num=1)

    #Run function that store results
    Store_model_data(model,Data)
    
    return()

def Store_model_data(model,Data):
    
    """
    Stores the results from the optimization model run into an excel file
    """
    
    
    #Create empty dictionaries that will be filled
    NodeData    = {}
    ACData      = {}
    DCData      = {}
    MiscData    = {}
    

    
    #Node data
    
    #Write dictionaries for each node related value
    Theta       = {}
    Gen         = {}
    Shed        = {}
    Demand      = {}
    CostGen     = {}
    CostShed    = {}
    DualNode    = {}    
    
    #Emission
    Emission    = {}


    #For every node, store the data in the respective dictionary
    for node in model.N:
        
        #If we have DC OPF, we want to store Theta, if not then we skip it
        if Data["DCFlow"] == True:
            Theta[node]         = round(model.theta[node].value,4)

        DualNode[node]      = round(model.dual[model.LoadBal_const[node]],1) 
        Gen[node]           = round(model.gen[node].value,4)
        Shed[node]          = round(model.shed[node].value,4)
        Demand[node]        = round(model.Demand[node],4)
        CostGen[node]       = round(model.gen[node].value*model.Cost_gen[node],4)
        CostShed[node]      = round(model.shed[node].value*model.Cost_shed,4)
                
        #Emissions
        Emission[node] = round(model.e[node].value,4)

    
    
    #Store Node Data
    NodeData["Theta [rad]"]       = Theta
    NodeData["Gen"]         = Gen
    NodeData["Shed"]        = Shed
    NodeData["Demand"]      = Demand
    NodeData["MargCost"]    = Data["Nodes"]["GENCOST"]
    NodeData["CostGen"]     = CostGen
    NodeData["CostShed"]    = CostShed
    NodeData["Emission"]    = Emission
    NodeData["Node Name"]   = Data["Nodes"]["NNAMES"]
    NodeData["Price"]       = DualNode
    
    #AC-line data
    ACFlow      = {}
    DualFrom    = {}
    DualTo      = {}
    
    #For every line, store the result
    for line in model.L:
        ACFlow[line]        = round(model.flow_AC[line].value,4)
        
        #Only if DCOPF is true
        if Data["DCFlow"] == True:
            DualFrom[line]      = round(model.dual[model.FlowBal_const[line]]/Data["pu-Base"],1)
            
        #If not, then we store the dual values for both the max and minimum flow constraints
        else:
            DualFrom[line]      = round(model.dual[model.From_flow_L[line]],1)
            DualTo[line]        = round(model.dual[model.To_flow_L[line]],1)
    
    #Extract data from input that can be shown with the results
    ACData["AC Flow"]           = ACFlow
    ACData["Max power from"]    = Data["AC-lines"]["Cap From"]
    ACData["Max power to"]      = Data["AC-lines"]["Cap To"]
    ACData["From Node"]         = Data["AC-lines"]["From"]
    ACData["To Node"]           = Data["AC-lines"]["To"]
    
    
    #This one is only necessary to include if we have DC OPF
    if Data["DCFlow"] == True:
        ACData["Admittance"]        = Data["AC-lines"]["Admittance"]
        ACData["Dual Value"]        = DualFrom
    
    else:
        ACData["Dual Value from"]       = DualFrom
        ACData["Dual Value to"]         = DualTo

        
    
    #DC-line data
    DCFlow      = {}
    
    #For every cable, store the result
    for cable in model.H:
        DCFlow[cable]       = round(model.flow_DC[cable].value,4)
    
    DCData["DC Flow"]           = DCFlow
    DCData["Capacity"]          = Data["DC-lines"]["Cap"]
    DCData["From Node"]         = Data["DC-lines"]["From"]
    DCData["To Node"]           = Data["DC-lines"]["To"]
    
    
    #Misc
    Objective   = round(model.OBJ(),4)
    DCOPF       = Data["DCFlow"]
    
    MiscData["Objective"]   = {1:Objective}
    MiscData["DCOPF"]       = {2:DCOPF}  
    
    
    #Convert the dictionaries to objects for Pandas
    NodeData    = pd.DataFrame(data=NodeData)
    ACData      = pd.DataFrame(data=ACData)
    DCData      = pd.DataFrame(data=DCData)
    MiscData    = pd.DataFrame(data=MiscData) 
    
    folder_file_path = "src/Problem_3/"   #Path to the folder where the file should be stored

    #Decide what the name of the output file should be
    if Data["DCFlow"] == True:
        output_file = folder_file_path+"DCOPF_results(CO2).xlsx"
    else:
        output_file = folder_file_path+"ATC_results(CO2).xlsx"
    
    #Store each result in an excel file, given a separate sheet
    with pd.ExcelWriter(output_file) as writer:
        NodeData.to_excel(writer, sheet_name= "Node")
        ACData.to_excel(writer, sheet_name= "AC")
        DCData.to_excel(writer, sheet_name= "DC")
        MiscData.to_excel(writer, sheet_name= "Misc")
        
    print("\n\n")
    print("The results are now stored in the excel file: " + output_file)
    print("This program will now end")

    return()

