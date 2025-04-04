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

def main():
    """ Methodology to run the project problems
    """
    xs = np.linspace(-1500, 1500, 100)

    y_max = np.full_like(xs, 1500)
    y_min = np.full_like(xs, -1500)

    plt.plot(xs, y_max, '-', c='orange')  # Max power line
    plt.plot(y_max, xs, '-', c='orange')  # Max power line
    plt.plot(xs, y_min, '-', c='orange')  # Min power line
    plt.plot(y_min, xs, '-', c='orange')  # Min power line
    upper = np.vectorize(upper_c)
    lower = np.vectorize(lower_c)   

    # Assuming `lower` and `y_min` are functions or arrays of the same shape as `xs`
    y1 = np.where(xs < 0, lower(xs), y_min)  # Use `lower(xs)` when xs < 0, otherwise use `y_min`
    y2 = np.where(xs > 0, upper(xs), y_max)  # Use `upper(xs)` when xs > 0, otherwise use `y_max`
    plt.plot(xs, y1, c='orange')  # Upper limit
    plt.plot(xs, y2, c='orange')  # Lower limit
    plt.fill_between(xs, y1, y2, color='orange',label='ATC', alpha=0.2)

    # Add text annotations for max and min lines
    plt.text(0, 1500, 'Max export A', color='orange', fontsize=10, ha='center', va='bottom')  # Text for max line
    plt.text(0, -1500, 'Max import A', color='orange', fontsize=10, ha='center', va='top')   # Text for min line
    # Add text annotations for max and min lines
    plt.text(1500, 1500, 'Max export B', color='orange', fontsize=10, ha='center', va='bottom')  # Text for max line
    plt.text(-1500, -1500, 'Max import B', color='orange', fontsize=10, ha='center', va='top')   # Text for min line


    # Add x and y axes
    plt.axvline(x=0, color='gray', linestyle='-', linewidth=1)  # Vertical axis
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=1)  # Horizontal axis

    # Set x and y ticks
    plt.ylim(-2000,2000)
    plt.xlim(-2000,2000)

    # Add grid for better visualization
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add labels and title
    plt.xlabel("Net balance B")
    plt.ylabel("Net balance A")
    plt.title("Power Flow Visualization")

    plt.legend()
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
if __name__ == "__main__":
    main()