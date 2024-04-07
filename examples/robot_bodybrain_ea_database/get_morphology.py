import sys
import os
# Set algorithm, mode and file name from command line arguments.
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
# Set parameters
import config
os.environ['MAXPARTS'] = str(config.MAX_PARTS)

if os.environ["ALGORITHM"] == "CPPN":
    from genotype import Genotype
elif os.environ["ALGORITHM"] == "GRN":
    from genotype_grn import Genotype
else:
    raise ValueError("ALGORITHM must be either GRN or CPPN")


import concurrent.futures
from get_morphology_function import get_morphologies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment import Experiment
from generation import Generation
from individual import Individual
from population import Population
from sqlalchemy.orm import Session
from sqlalchemy import select

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging

def select_data(dbengine) -> pd.DataFrame:
    """Goal:
        Select the data of the column
    -------------------------------------------------------------
    Input:
        dbengine: ?
    --------------------------------------------------------------
    Output:
        df: pd.Dataframe"""
    
    with Session(dbengine) as ses:
        rows = ses.execute(
            select(Genotype, Experiment.id.label("experiment_id"), Generation.generation_index,
                   Individual.id.label("individual_index"))
            .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id)
            .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id),
        ).fetchall()

    return rows

def main():
    # Setup logging
    setup_logging()

    # Initialize dataframe
    df = []

    # Open database
    dbengine = open_database_sqlite(config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS)

    # Get pandas data
    rows = select_data(dbengine)
    nrows = len(rows)
    print(f"Number of rows: {nrows}")

    # Get morphologies

    with concurrent.futures.ProcessPoolExecutor(max_workers = config.NUM_SIMULATORS
                ) as executor:
                    futures = [
                        executor.submit(get_morphologies, row, config.ZDIRECTION,
                                        config.CPPNBIAS, config.CPPNCHAINLENGTH, config.CPPNEMPTY,
                                        config.MAX_PARTS, config.MODE_COLLISION, config.MODE_CORE_MULT,
                                        config.MODE_SLOTS4FACE, config.MODE_SLOTS4FACE_ALL,
                                        config.MODE_NOT_VERTICAL) for row in rows]
                     
    dicts = [future.result() for future in futures]
    
    # Convert to dataframe
    df = pd.DataFrame(dicts)


    # Create directory
    #path = f"C:\\Users\\niels\\OneDrive\\Documenten\\GitHub\\revolve2\\Test\\{os.environ['ALGORITHM']}\\Morphologies"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    import uuid
    uuid = uuid.uuid4()
    df.to_csv(f"morphological_measures_experiment_{uuid}.csv", index = False)
# # # Get max and mean fitness per experiment per generation
# # agg_per_experiment_per_generation = (
# #     df.groupby(["experiment_id", "generation_index"])
# #     .agg({column: ["max", "mean"]})
# #     .reset_index()
# # )

# # # Aggregate over experiments
# # agg_per_experiment_per_generation.columns = [
# #     "experiment_id",
# #     "generation_index",
# #     f"max_{column}",
# #     f"mean_{column}",
# # ]

if __name__ == "__main__":
    main()