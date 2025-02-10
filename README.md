## PowerMarkets_project
Group project

- This could be the remember type file. For example the overview of the optimization problem or a step-by-step guide.


** Note: The following is taken from a general example, it might not apply directly to our case
# Objective Function
    The objective function is to `maximize` social welfare which is defined as the difference between the total utility obtained from electricity consumption and the total generation cost.

# Constraints
    1. Power Balance: The total demand must equal the total generation
    2. Generation and Consumption Limits: Each generator and consumer is subject to their respective capacity limits



## Implementation
This section provides a detailed explanation of the Python code used to implement the market-clearing model with Pyomo.

# Step 1: Reading Data from Excel
This block reads the Excel file and stores the data in two Pandas DataFrames. The data will be later converted into Python lists and dictionaries for use in Pyomo.

# Step 2: Preparing Data for Pyomo
The generators and demands are converted into lists while their respecctive parameters (maximum generation, cost, limits, and utility) are stored in dictionaries. This structure simplifies the subsequent model definition in Pyomo.

# Step 3: Creating the Pyomo Model
When we define the Pyomo model by setting up the sets, parameters, and decision variables.

A concrete Pyomo model is created and the sets (g for generators and d for demands) are initialized. The parameters are then delcared to link each set with its corresponding numerical data.
    3.1 Defining Variables 
    The variables represent the power output of the generators and the power consumption of demand points. Both are restricted to non-negative real numbers. 

# Step 4: Objective Function and Costraints

    4.1 Objective Function
    This function calculates the difference between the total utility (from consumption) and the total cost (from generation), thereby representting the social welfare

    4.2 Constrains
    We add constrains to the model to enforce generator capacity, demand limits, and power balance. Each constraint is defined by a Python function:

    * The generator capacity constraint ensures that no generator produces more than its maximum capacity
    * The demand capacity constraint ensures that consumption does not exceed the offered limit
    * The power balance constraint guarantees that total generation equals total consumption

# Step 5: Solving the Model
We now solve the model using the Gurobi solver. Additionally, we delacre a suffix to capture dual values (shadow prices).
The solver is invoked to solve the optimization problem. The Suffix object is used to import dual information from the solver, which will be used later to interpret the shadow prices associated with the constraints.

# Step 6: Displaying Results and Extracting Dual Values
Finally, we print the optimal solution, including generation, demand, and market-clearing price. We also extract and display the dual values for the generator capacity and demand capacity constraints.

This block prints the optimal social welfare, the generation and demand levels, and extracts the dual values (shadow prices) for each constraint

*The dual of the power balance constraint represents the market-clearing price.
*The dual values for the generator and demand capacity constraint provides marginal values, indication how the objectie would chjange if the respective limits were relaxed.

