import logging
import sys
import os
# Set algorithm, mode and file name from command line arguments.
algo = sys.argv[1]
mode = sys.argv[2]
file_name = sys.argv[3]
experiment_id_start = sys.argv[4]
pop2start = sys.argv[5]
os.environ["NEXP"] = sys.argv[6]
os.environ["NPOP"] = sys.argv[7]
assert algo in ["GRN", "GRN_system", "GRN_system_adv", "CPPN"], "ALGORITHM must be either GRN, GRN_system, GRN_system_adv, or CPPN"
assert mode in ["random search", "evolution"], "MODE must be either random search or evolution"
assert type(file_name) == str, "FILE_NAME must be a string"
assert file_name.endswith(".sqlite"), "FILE_NAME must end with sqlite"
os.environ["ALGORITHM"] = algo
os.environ["MODE"] = mode
os.environ["DATABASE_FILE"] = file_name
os.environ["POP2START"] = pop2start
os.environ["EXPERIMENT_ID_START"] = experiment_id_start
# Set parameters
import config
os.environ['MAXPARTS'] = str(config.MAX_PARTS)

if os.environ["ALGORITHM"] == "CPPN":
    from genotype import Genotype
elif os.environ["ALGORITHM"] in ["GRN", "GRN_system", "GRN_system_adv"]:
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
from sqlalchemy import select, and_

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging

def select_data(dbengine, experiment_id, min_population_id, max_population_id) -> pd.DataFrame:
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
                   Individual.id.label("individual_index"), Individual.body_id)
            .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id)
            .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id).where(
            and_(Experiment.id == experiment_id,
                 Generation.generation_index.between(min_population_id, max_population_id))
            ),
        ).fetchall()

    return rows

def main():
    # Setup logging
    setup_logging()

    # Initialize dataframe
    df = []

    # Open database
    dbengine = open_database_sqlite(config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS)

    # Get morphologies
    for exp in range(int(os.environ["EXPERIMENT_ID_START"]), int(os.environ["NEXP"]) + 1):
        if exp == int(os.environ["EXPERIMENT_ID_START"]):
            popid2start = int(os.environ["POP2START"])
        else:
            popid2start = 0
        
        for pop in range(popid2start * 2, (int(os.environ["NPOP"]) * 2) + 1, 10):
            data = select_data(dbengine, exp, pop, int(pop + 9))
            
            with concurrent.futures.ProcessPoolExecutor(max_workers = config.NUM_SIMULATORS
                        ) as executor:
                            futures = [
                                executor.submit(get_morphologies, row, config.ZDIRECTION,
                                                config.CPPNBIAS, config.CPPNCHAINLENGTH, config.CPPNEMPTY,
                                                config.MAX_PARTS, config.MODE_COLLISION, config.MODE_CORE_MULT,
                                                config.MODE_SLOTS4FACE, config.MODE_SLOTS4FACE_ALL,
                                                config.MODE_NOT_VERTICAL) for row in data]
                            
            dicts = [future.result() for future in futures]
            
            # Convert to dataframe
            df = pd.DataFrame(dicts)
            logging.info(f"Experiment {exp}: population {pop} done")
            df.to_csv(f"morphological_measures_experiment_{file_name.split('.')[0]}_{exp}_{pop}.csv", index = False)

if __name__ == "__main__":
    main()