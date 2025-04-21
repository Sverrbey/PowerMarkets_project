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

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def main():
    """ Methodology to run the project problems
    """
    
    plt.figure(figsize=(10, 7))
    xs = np.linspace(-1500, 1500, 100)

    #### DCOPF

    # -----------------------------
    # Step 1: Define the system
    # -----------------------------

    # PTDF matrix: 2 CNEs, 3 bidding zones (A, B, C)
    # Assume C is slack (PTDFs = 0)
    PTDF = np.array([
        [0.67, 0.33, 0.0],  # CNE 1: Flow = 0.67*NP_A + 0.33*NP_B
        [0.33, 0.67, 0.0],  # CNE 2: Flow = 0.33*NP_A + 0.67*NP_B
    ])

    # Remaining Available Margins (RAM) for each CNE
    RAM = np.array([1000, 1000])

    # -----------------------------
    # Step 2: Define NP grid
    # -----------------------------

    # We eliminate NP_C by enforcing balance: NP_C = -NP_A - NP_B
    x = np.linspace(-2000, 2000, 400)  # NP_A range
    y = np.linspace(-2000, 2000, 400)  # NP_B range
    X, Y = np.meshgrid(x, y)

    # Balance constraint: NP_C = -NP_A - NP_B
    Z = -X - Y

    # -----------------------------
    # Step 3: Calculate flows on each CNE
    # -----------------------------

    # Flow = PTDF * NP vector (per CNE)
    flow_1 = PTDF[0, 0] * X + PTDF[0, 1] * Y + PTDF[0, 2] * Z
    flow_2 = PTDF[1, 0] * X + PTDF[1, 1] * Y + PTDF[1, 2] * Z

   

   
    # -----------------------------
    # Step 4: Plot
    # -----------------------------
   
    # Feasible region mask: flows must be within +/- RAM
    mask = (np.abs(flow_1) <= RAM[0]) & (np.abs(flow_2) <= RAM[1])

    plt.contourf(X, Y, mask, levels=[0, 0.5, 1], colors=['white','blue'],alpha=0.7)

    feasible_region_patch = mpatches.Patch(color='blue', label='DCOPF')
    

    # Constraint lines: solve for NP_B = (RAM - PTDF_A * NP_A) / PTDF_B
    # CNE 1 boundaries
    cne1_upper = (RAM[0] - PTDF[0][0] * x) / PTDF[0][1]
    cne1_lower = (-RAM[0] - PTDF[0][0] * x) / PTDF[0][1]

    # CNE 2 boundaries
    cne2_upper = (RAM[1] - PTDF[1][0] * x) / PTDF[1][1]
    cne2_lower = (-RAM[1] - PTDF[1][0] * x) / PTDF[1][1]

    # Plot the constraint lines
    line1, = plt.plot(x, cne1_upper, '--r', label='CNE 1 Upper')
    line2, = plt.plot(x, cne1_lower, '--r', label='CNE 1 Lower')
    line3, = plt.plot(x, cne2_upper, '--b', label='CNE 2 Upper')
    line4, = plt.plot(x, cne2_lower, '--b', label='CNE 2 Lower')

    

    #### 

    #### ATC 
    # Create a figure and axis
    y_max = np.full_like(xs, 1500)
    y_min = np.full_like(xs, -1500)
    upper = np.vectorize(upper_c)
    lower = np.vectorize(lower_c) 
    
    # If both node A and B can both generate and consume power, the ATC is given by:
    y1 = np.where(xs < 0, lower(xs), y_min)  # Use `lower(xs)` when xs < 0, otherwise use `y_min`
    y2 = np.where(xs > 0, upper(xs), y_max)  # Use `upper(xs)` when xs > 0, otherwise use `y_max`
    plt.plot(xs, y1, c='orange')  # Upper limit
    plt.plot(xs, y2, c='orange')  # Lower limit
    ATC_area = plt.fill_between(xs, y1, y2, color='orange',label='ATC', alpha=1)

    
    plt.plot(xs, y_max, '--', c='orange')  # Max power line
    plt.plot(y_max, xs, '--', c='orange')  # Max power line
    plt.plot(xs, y_min, '--', c='orange')  # Min power line
    plt.plot(y_min, xs, '--', c='orange')  # Min power line

    

      

    # Add text annotations for max and min lines
    plt.text(0, 1500, 'Max export A', color='brown', fontsize=10, ha='left', va='bottom')  # Text for max line
    plt.text(0, -1500, 'Max import A', color='brown', fontsize=10, ha='right', va='top')   # Text for min line
    # Add text annotations for max and min lines
    plt.text(1500, 0, 'Max export B', color='brown', fontsize=10, ha='left', va='bottom')  # Text for max line
    plt.text(-1500, 0, 'Max import B', color='brown', fontsize=10, ha='right', va='top')   # Text for min line

    # Explaining text
    plt.text(650, 250, 'C is consuming', color='brown',fontsize=10, ha='center')
    plt.text(650, -250, 'A and C are consuming', color='brown', fontsize=10, ha='center')
    plt.text(-650, 250,'B and C are consuming', color='brown', fontsize=10, ha='center')
    plt.text(-650, -250,'A and B are consuming', color='brown', fontsize=10, ha='center')

    ####

    # Set x and y ticks
    plt.ylim(-2500,2500)
    plt.xlim(-2500,2500)

    # Combine all handles for the legend
    handles = [feasible_region_patch, ATC_area, line1, line2, line3, line4]
    plt.legend(handles=handles, loc='upper right')

    plt.xlabel("Net Position A (MW)")
    plt.ylabel("Net Position B (MW)")
    plt.title("Feasible Solution Domain of DCOPF and ATC ")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    #plt.grid()
    plt.show()

def lower_c(xs):
    if xs < 0:
        return (-xs) - 1500
    else:
        return 0

def upper_c(xs):
    if xs > 0:
        return (-xs) + 1500
    else:
        return 0

def ATC_bounds(xs):
    if xs < 0:
        return (xs) + 1500
    else:
        return (-xs) + 1500

if __name__ == "__main__":
    main()