"""Plot fitness over generations for all experiments, averaged."""

import config
import os
os.environ['ALGORITHM'] = config.ALGORITHM

import matplotlib.pyplot as plt
import pandas
from experiment import Experiment
from generation import Generation
from individual import Individual
from population import Population
from sqlalchemy import select

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging


def main(column) -> None:
    """Run the program."""
    setup_logging()

    # Open database
    dbengine = open_database_sqlite(config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS)

    # Get pandas data
    df = select_data(dbengine, column)

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
    plt.title(f"Mean and Max '{column.title()}' across Repetitions with Std as Shade", fontweight = "bold", size = 16)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


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
    main(column = "fitness")


# Planning
# Alle variabelen even doorlopen met plots en kijken of het is zoals gewenst
# 2D check opnieuw maken
# Paar plaatjes controleren
# Probleem met balance
# Crossover voor brein indien GRN?