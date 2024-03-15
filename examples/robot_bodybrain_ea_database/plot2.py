import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np

# Load the data
with open("Generational Info", 'r') as file:
    # Load JSON data from the file
    data = json.load(file)

# Convert the data to a pandas dataframe
columns = list(data[0].keys())
df = pd.DataFrame([], columns = columns)
for sample in data:
    df = pd.concat([df, pd.DataFrame(sample, index = [0], columns = columns)], 
                   ignore_index = True, axis = 0)
    
def add2plot(variable, std = False):
    """Goal:
        Add a variable to the plot.
    -------------------------------------------------------------------------------------------
    Input:
        variable: The variable to add to the plot.
        std: Whether to add the standard deviation to the plot.
    -------------------------------------------------------------------------------------------
    Output:
        None"""
    # Plot Mean
    p = plt.plot(df['Generation'], df[f"{variable}_mean"] / max(df[f"{variable}_max"]), label = f"Mean {variable}")

    # Add std
    if std:
        lwb = ((df[f"{variable}_mean"] - df[f"{variable}_std"]) / max(df[f"{variable}_max"])).astype(float)
        ub = ((df[f"{variable}_mean"] + df[f"{variable}_std"]) / max(df[f"{variable}_max"])).astype(float)
        plt.fill_between(df['Generation'].astype(int), lwb, ub, color = p[0].get_color(), alpha = 0.2,)
    
    #plt .plot(df['Generation'], df[f"{variable}_min"] / max(df[f"{variable}_max"]), "--", label = f"Min {variable}")
    #plt.plot(df['Generation'], df[f"{variable}_max"] / max(df[f"{variable}_max"]), "--", label = f"Max {variable}")
# add2plot("Efficiency")
# add2plot("Energy")
# add2plot("X-distance")
#add2plot("Size")
#add2plot("Proportion")
    
measures = ["Size", "Proportion", "Limbs", "Maxrel", "Meanrel", 
                    "Stdrel", "Joints", "Joint-Brick Ratio", "Symmetry", 
                    "Coverage", "Branching", "Surface", "Modules", 
                    "Active Hinges", "Bricks", "Energy", "Efficiency", 
                    "X-distance", "Y-distance"]
for measure in measures:
    add2plot(measure, std = True)
    plt.xlabel("Generation", size = 16, fontweight = "bold")
    plt.ylabel(measure.title(), size = 16, fontweight = "bold")
    plt.title(f"{measure.title()} over Generations", size = 16, fontweight = "bold")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()