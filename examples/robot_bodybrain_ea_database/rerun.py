"""Rerun the best robot between all experiments."""

import logging
import os
import sys

# Set algorithm, mode and file name from command line arguments.
algo = sys.argv[1]
mode = sys.argv[2]
file_name = sys.argv[3]
headless = sys.argv[4]
assert headless in ["True", "False"], "HEADLESS must be either True or False"
assert algo in ["GRN", "CPPN"], "ALGORITHM must be either GRN or CPPN"
assert mode in ["random search", "evolution"], "MODE must be either random search or evolution"
assert type(file_name) == str, "FILE_NAME must be a string"
assert file_name.endswith(".sqlite"), "FILE_NAME must end with sqlite"
os.environ["ALGORITHM"] = algo
os.environ["MODE"] = mode
os.environ["DATABASE_FILE"] = file_name
os.environ["HEADLESS"] = headless

# Import parameters
import config
os.environ['MAXPARTS'] = str(config.MAX_PARTS)

# Import the genotype
if os.environ["ALGORITHM"] == "CPPN":
    from genotype import Genotype
elif os.environ["ALGORITHM"] == "GRN":
    from genotype_grn import Genotype
else:
    raise ValueError("ALGORITHM must be either GRN or CPPN")

# Import other modules
from evaluator import Evaluator
from individual import Individual
from sqlalchemy import select
from sqlalchemy.orm import Session

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging


def main() -> None:
    """Perform the rerun."""
    setup_logging()

    # Load the best individual from the database.
    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS
    )

    with Session(dbengine) as ses:
        rows = ses.execute(
            select(Genotype, Individual.fitness, Individual.energy_used, Individual.efficiency,
                   Individual.x_distance, Individual.y_distance)
            .join_from(Genotype, Individual, Genotype.id == Individual.genotype_id)
            .filter(Genotype.id == 1)
            # .order_by(Individual.fitness.desc()).limit(1000) #Individual.population_id.desc()

        ).first()
        # Get individual with id 1
        # rows = ses.execute(
        #     select(Individual.id, Genotype, Individual.fitness)
        #     .join_from(Genotype, Individual, Genotype.id == Individual.genotype_id)
        #     .filter(Individual.id == 1)
        # ).first()


    
    print(rows[0])
    print(rows[1])
    exit(0)
    for irow, row in enumerate(rows[0:301]):
        genotype = row[0]
        print(genotype)
        fitness = row[1]
        energy_used = row[2]
        efficiency = row[3]
        x_distance = row[4]
        y_distance = row[5]

        if os.environ["ALGORITHM"] == "CPPN":
            modular_robot = genotype.develop(zdirection = config.ZDIRECTION, include_bias = config.CPPNBIAS,
                include_chain_length = config.CPPNCHAINLENGTH, include_empty = config.CPPNEMPTY,
                max_parts = config.MAX_PARTS, mode_collision = config.MODE_COLLISION,
                mode_core_mult = config.MODE_CORE_MULT, mode_slots4face = config.MODE_SLOTS4FACE,
                mode_slots4face_all = config.MODE_SLOTS4FACE_ALL, mode_not_vertical = config.MODE_NOT_VERTICAL)
        elif os.environ["ALGORITHM"] == "GRN":
            modular_robot = genotype.develop(include_bias = config.CPPNBIAS, max_parts = config.MAX_PARTS, mode_core_mult = config.MODE_CORE_MULT)
        else:
            raise ValueError("ALGORITHM must be either GRN or CPPN")
        logging.info(f"Fitness: {fitness}")
        logging.info(f"Energy used: {energy_used}")
        logging.info(f"Efficiency: {efficiency}")
        logging.info(f"X distance: {x_distance}")
        logging.info(f"Y distance: {y_distance}")


        # Create the evaluator.
        headless = os.environ["HEADLESS"] == "True"
        evaluator = Evaluator(headless = headless, num_simulators = 1, terrain = config.TERRAIN, fitness_function = config.FITNESS_FUNCTION,
                              simulation_time = config.SIMULATION_TIME, sampling_frequency = config.SAMPLING_FREQUENCY,
                              simulation_timestep = config.SIMULATION_TIMESTEP, control_frequency = config.CONTROL_FREQUENCY,
                              record = not headless, video_path = os.getcwd() + f"/MuJoCo_videos/MuJoCo_{irow}")

        # Show the robot.
        fitnesses, behavioral_measures = evaluator.evaluate([modular_robot])
        logging.info(f"Fitness Measured: {fitnesses[0]}")
        print("-----------------------------------------------")
    
if __name__ == "__main__":
    # run with arguments <algo> <mode> <file_name> !!!
    main()
