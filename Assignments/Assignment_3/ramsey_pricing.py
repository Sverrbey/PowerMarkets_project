import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

#Note: In both the industry and household segments the demand is both multiplied and divided by 1e+3

def industry(d):
    return (180 - 0.5*d) # [NOK/GWh]

def household(d):
    return (550 - 3*d) # [NOK/GWh]

def combined(d):
    return (720 - 3.5*d) # [NOK/GWh]

avg_price = 60 

ds = np.linspace(0, 250, 250) # Demand [GW]

fig, axs = plt.subplots(1,3, figsize=(10,5), sharey=True)

# Plot the industry demand curve
axs[0].plot(ds, industry(ds), label='Industry')
axs[0].axhline(y=avg_price, color='r', linestyle='--', label='Average price')
axs[0].axvline(x=0, color='gray', linestyle='-')  # Add vertical line
axs[0].axhline(y=0, color='gray', linestyle='-')  # Add horizontal line
axs[0].set_title('Industry')
axs[0].set_xlabel('Demand [GWh]')
axs[0].set_ylabel('Price [NOK/GWh]')
axs[0].legend()

# Plot the household demand curve
axs[1].plot(ds, household(ds), label='Household')
axs[1].axhline(y=avg_price, color='r', linestyle='--', label='Average price')
axs[1].axvline(x=0, color='gray', linestyle='-')  # Add vertical line
axs[1].axhline(y=0, color='gray', linestyle='-')  # Add horizontal line
axs[1].set_title('Household')
axs[1].set_xlabel('Demand [GWh]')
axs[1].legend()

# Plot the combined demand curve
axs[2].plot(ds, combined(ds), label='Combined')
axs[2].axhline(y=avg_price, color='r', linestyle='--', label='Average price')
axs[2].axvline(x=0, color='gray', linestyle='-')  # Add vertical line
axs[2].axhline(y=0, color='gray', linestyle='-')  # Add horizontal line
axs[2].set_title('Combined')
axs[2].set_xlabel('Demand [GWh]')
axs[2].legend()


"""a. What is the demand quantity for both demand curves with this given average power price """

# In order to solve this we can utilize the function fsolve from the scipy.optimize library
demand_industry     = fsolve(lambda x: industry(x) - avg_price, 0)
demand_household    = fsolve(lambda x: household(x) - avg_price, 0)
demand_combined     = fsolve(lambda x: combined(x) - avg_price, 0)

print(f'The demand quantity for the industry is {demand_industry[0]:.2f} GWh')
print(f'The demand quantity for the household is {demand_household[0]:.2f} GWh')
print(f'The demand quantity for the combined is {demand_combined[0]:.2f} GWh')

axs[0].axvline(x=demand_industry[0] , color='b', linestyle='-', alpha=0.4, label='Industry')
axs[1].axvline(x=demand_household[0], color='b', linestyle='-', alpha=0.4, label='Household')
axs[2].axvline(x=demand_combined[0] , color='b', linestyle='-', alpha=0.4, label='Combined')


""" b. The grid company needs an annual revenue of 3.5e+6 NOK/year to cover their expenses and other cost related parts.
    They want to cover this by introducing a grid tariff that is included in the average power price. If we use Ramsay pricing to find
    the grid tariff, what are the prices and demands in the area for each demand curve?
    
    A tariff has a set of elemnts: 

    * The fixed element is a neutral element which is independent of the customer's consumption and it has
    no influence on the consumption pattern.

    * The energy charge is the consumption dependent element tied to the energy consumption. It can be constant or variable over time

    * The demand charge

    """




def ramsey_price(d, demand, avg_price):
    return (avg_price - demand(d)) / (1 - 1 / d)

# Target revenue
annual_revenue = 3.5e+6 # [NOK/year]

hours_year = 24*365 # hours in a year

revenue = annual_revenue/hours_year # [NOK/year]

print(f'{revenue:e}')
# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()