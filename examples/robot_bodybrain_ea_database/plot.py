"""Plot fitness over generations for all experiments, averaged."""

# Set algorithm, mode and file name from command line arguments.
import os
import sys

algo = sys.argv[1]
mode = sys.argv[2]
file_name = sys.argv[3]
assert algo in ["GRN", "CPPN"], "ALGORITHM must be either GRN or CPPN"
assert mode in ["random search", "evolution"], "MODE must be either random search or evolution"
assert type(file_name) == str, "FILE_NAME must be a string"
assert file_name.endswith(".sqlite"), "FILE_NAME must end with sqlite"
os.environ["ALGORITHM"] = algo
os.environ["MODE"] = mode
os.environ["DATABASE_FILE"] = file_name

# Import parameters
import config

# Import other modules
import matplotlib.pyplot as plt
import pandas
from experiment import Experiment
from generation import Generation
from individual import Individual
from population import Population
from sqlalchemy import select
from sqlalchemy.orm import Session

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging


def main(column, path) -> None:
    """Run the program."""
    setup_logging()

    # Open database
    dbengine = open_database_sqlite(config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS)

    # Get pandas data
    df = select_data(dbengine, column)

    # Select even generations
    first_generation = df.loc[df["generation_index"] == 0, :]
    df = df.loc[(df.loc[:, "generation_index"] % 2) == 0, :]
    df = pandas.concat([first_generation, df])
    df["generation_index"] = (df.loc[:, "generation_index"] / 2).astype(int).values    

    # Get max and mean fitness per experiment per generation
    agg_per_experiment_per_generation = (
        df.groupby(["experiment_id", "generation_index"])
        .agg({column: ["max", "mean"]})
        .reset_index()
    )

    # Aggregate over experiments
    agg_per_experiment_per_generation.columns = [
        "experiment_id",
        "generation_index",
        f"max_{column}",
        f"mean_{column}",
    ]

    agg_per_generation = (
        agg_per_experiment_per_generation.groupby("generation_index")
        .agg({f"max_{column}": ["mean", "std"], f"mean_{column}": ["mean", "std"]})
        .reset_index()
    )

    # Set columns
    agg_per_generation.columns = [
        "generation_index",
        f"max_{column}_mean",
        f"max_{column}_std",
        f"mean_{column}_mean",
        f"mean_{column}_std", ]

    plt.figure()

    # Plot max
    plt.plot(
        agg_per_generation["generation_index"],
        agg_per_generation[f"max_{column}_mean"],
        label=f"Max {column}",
        color="b",
    )
    plt.fill_between(
        agg_per_generation["generation_index"],
        agg_per_generation[f"max_{column}_mean"] - agg_per_generation[f"max_{column}_std"],
        agg_per_generation[f"max_{column}_mean"] + agg_per_generation[f"max_{column}_std"],
        color="b",
        alpha=0.2,
    )

    # Plot mean
    plt.plot(
        agg_per_generation["generation_index"],
        agg_per_generation[f"mean_{column}_mean"],
        label=f"Mean {column}",
        color="r",
    )
    plt.fill_between(
        agg_per_generation["generation_index"],
        agg_per_generation[f"mean_{column}_mean"]
        - agg_per_generation[f"mean_{column}_std"],
        agg_per_generation[f"mean_{column}_mean"]
        + agg_per_generation[f"mean_{column}_std"],
        color="r",
        alpha=0.2,
    )

    plt.xlabel("Generation Index", fontweight = "bold", size = 16)
    plt.ylabel(column.title(), fontweight = "bold", size = 16)
    plt.title(f"{column.title()}", fontweight = "bold", size = 16)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(path + f"//{column}.png")
    #plt.show()


def select_data(dbengine, column: str) -> pandas.DataFrame:
    """Goal:
        Select the data of the column
    -------------------------------------------------------------
    Input:
        dbengine: ?
        column: The column that needs to be selected
    --------------------------------------------------------------
    Output:
        df: pd.Dataframe"""
    df = pandas.read_sql(
        select(Experiment.id.label("experiment_id"), Generation.generation_index, getattr(Individual, column),)
        .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
        .join_from(Generation, Population, Generation.population_id == Population.id)
        .join_from(Population, Individual, Population.id == Individual.population_id),
        dbengine,
    )

    return df

if __name__ == "__main__":
    # Columns
    columns = ["fitness", "energy_used", 
               
               "x_distance", "tot_xdistance", "xmax", "y_distance", "tot_ydistance",
               
               "min_dx", "dx25", "mean_dx", "median_dx", "dx75", "max_dx", "std_dx",
               
               "min_dy", "dy25", "mean_dy", "median_dy", "dy75", "max_dy", "std_dy",
                
                "energy_used_min", "energy_used_25", "energy_used_mean", "energy_used_median",
                "energy_used_75", "energy_used_max", "energy_used_std",

                "force_std_motor_min", "force_std_motor_25", "force_std_motor_mean", "force_std_motor_median",
                "force_std_motor_75", "force_std_motor_max", "force_std_motor_std",

                "force_std_all_min", "force_std_all_25", "force_std_all_mean", "force_std_all_median",
                "force_std_all_75", "force_std_all_max", "force_std_all_std",

                "efficiency", "efficiency_min", "efficiency_25", "efficiency_mean", "efficiency_median",
                "efficiency_75", "efficiency_max", "efficiency_std",

                "balance"]

    # If folder does not exist, create it
    path = f"C:\\Users\\niels\\OneDrive\\Documenten\\GitHub\\revolve2\\Test\\{os.environ['ALGORITHM']}\\BehavioralMeasures\\plots"
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot
    for column in columns:
        main(column = column, path = path)


# Planning
# Alle variabelen even doorlopen met plots en kijken of het is zoals gewenst
# 2D check opnieuw maken
# Paar plaatjes controleren
# Probleem met balance
# Crossover voor brein indien GRN?
# Moet ik de offspring ook niet opslaan???