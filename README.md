## TET4185 - Power Markets project
This repository contains the implementation of various optimization models for analyzing power markets. The project includes tasks such as creating optimization models, extending them with multiple generators and loads, and incorporating environmental constraints.

## Prerequisites

Before running the code, ensure you have the following installed:
- Python 3.8 or later
- Required Python packages (install using `pip install -r requirements.txt`)
- Gurobi solver (ensure it is installed and licensed)

## Project Structure

- **`main.py`**: The main entry point for running the different tasks in the project.
- **`src/Problem_2/`**: Contains the implementation of optimization models for Tasks 2-2, 2-3, 2-4, and 2-5.
- **`src/Problem_3/`**: Contains the implementation of optimization models for Task 3.
- **`data/`**: Contains input data files for the models.

## How to Run the Code

The `main.py` file contains function calls for different tasks. To run a specific task, **comment out or uncomment** the corresponding function call in the `main()` function.

### Steps to Run:

1. Open the `main.py` file.
2. Locate the `main()` function.
3. Comment out or uncomment the relevant function call for the task you want to run.
4. Run the script using the following command:
   ```bash
   python main.py


## Guide to Running Each Task

## Task 2-2: Creating the Optimization Model

To run Task 2-2a, uncomment the line 46 in main.py:

`DCOPF_model(N, L, D, G, PGmax, C, demands, linecap, susceptance, S_base)`

To run Task 2-2c, uncomment line 51 in main.py:

`DCOPF_model(N, L, D, G, PGmax, C, demands, linecap, susceptance, S_base)`

## Task 2-3: Extending the model: Multiple generators

To run Task 2-3, uncomment line 62 in main.py:

`DCOPF_model_multiple_generators(N, L, D, G, PGmax, C, demands, linecap, susceptance, location_g, S_base)`

## Task 2-4: Extending the model: Multiple loads

To run Task 2-4a, uncomment the line 73 in main.py:

`DCOPF_model_multiple_generators_and_loads(N, L, D, G, PGmax, C, demands, U, linecap, susceptance, location_g, location_d, S_base)`

To run Task 2-4c, uncomment line 76 in main.py:

`DCOPF_model_multiple_generators_and_loads_SW(N, L, D, G, PGmax, C, demands, U, linecap, susceptance, location_g, location_d, S_base)`

## Task 2-5: Extending the model: Environmental constraints

To run Task 2-5b, uncomment the line 89 in main.py:

`DCOPF_model_multiple_gens_and_loads_emissions_CES(N, L, D, G, PGmax, C, demands, U, linecap, susceptance, location_g, location_d, emissions, S_base)`

To run Task 2-5b, uncomment line 92 in main.py:

`DCOPF_model_multiple_gens_and_loads_emissions_cap_and_trade(N, L, D, G, PGmax, C, demands, U, linecap, susceptance, location_g, location_d, emissions, S_base)`

## Task 3-2: Analyzing a wet-year scenario

To run Task 3-2 with DCFlow, uncomment line 102 in main.py:

`OPF_model(Data)`
    
To run Task 3-2 with ATC, uncomment line 104 in main.py:

`OPF_model(Data)`

## Task 3-3: Analyzing a dry-year scenario

To run Task 3-3 with DCFlow, uncomment line 111 in main.py:

`OPF_model(Data)`
    
To run Task 3-3 with ATC, uncomment line 113 in main.py:

`OPF_model(Data)`

## Task 3-4: Phasing out baseload produciton

To run Task 3-4 with DCFlow, uncomment line 122 in main.py:

`OPF_model(Data)`
    
To run Task 3-4 with ATC, uncomment line 124 in main.py:

`OPF_model(Data)`


## Task 3-5: Emission trading system

To run Task 3-5 with DCFlow, uncomment line 148 in main.py:

`OPF_model(Data)`
    

