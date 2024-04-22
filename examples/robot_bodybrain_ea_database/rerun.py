"""Rerun the best robot between all experiments."""
import logging
import os
import sys

# Set algorithm, mode and file name from command line arguments.
algo = sys.argv[1]
mode = sys.argv[2]
file_name = sys.argv[3]
headless = sys.argv[4]
writefiles = sys.argv[5]
writevideos = sys.argv[6]

assert headless in ["True", "False"], "HEADLESS must be either True or False"
if (writefiles == "True"):
    assert (writevideos == "False"), "WRITEVIDEOS must be False if WRITEFILES is True"
    assert (headless == "True"), "HEADLESS must be True if WRITEFILES is True"
assert writefiles in ["True", "False"], "WRITEFILES must be either True or False"
assert writevideos in ["True", "False"], "WRITEVIDEOS must be either True or False"
assert algo in ["GRN", "GRN_system", "CPPN"], "ALGORITHM must be either GRN, 'GRN_system' or CPPN"
assert mode in ["random search", "evolution"], "MODE must be either random search or evolution"
assert type(file_name) == str, "FILE_NAME must be a string"
assert file_name.endswith(".sqlite"), "FILE_NAME must end with sqlite"
os.environ["ALGORITHM"] = algo
os.environ["MODE"] = mode
os.environ["DATABASE_FILE"] = file_name
os.environ["HEADLESS"] = headless
os.environ["WRITEFILES"] = writefiles
os.environ["WRITEVIDEOS"] = writevideos

if os.environ["WRITEFILES"] == "True":
    os.environ["RERUN"] = "True"
else:
    os.environ["RERUN"] = "False"

# Import parameters
import config
os.environ['MAXPARTS'] = str(config.MAX_PARTS)

# Import the genotype
if os.environ["ALGORITHM"] == "CPPN":
    from genotype import Genotype
elif os.environ["ALGORITHM"] in ["GRN", "GRN_system"]:
    from genotype_grn import Genotype
else:
    raise ValueError("ALGORITHM must be either GRN or CPPN")

# Import other modules
from evaluator import Evaluator
from individual import Individual
import shutil
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
            .order_by(Individual.fitness.desc()).limit(1000)
        ).all()
    
    for irow, row in enumerate(rows[0:1]):
        genotype = row[0]
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
        elif os.environ["ALGORITHM"] in ["GRN", "GRN_system"]:
            modular_robot = genotype.develop(include_bias = config.CPPNBIAS, max_parts = config.MAX_PARTS, mode_core_mult = config.MODE_CORE_MULT)
        else:
            raise ValueError("ALGORITHM must be either GRN or CPPN")
        logging.info(f"Fitness: {fitness}")
        logging.info(f"Energy used: {energy_used}")
        logging.info(f"Efficiency: {efficiency}")
        logging.info(f"X distance: {x_distance}")
        logging.info(f"Y distance: {y_distance}")


        # Create the evaluator.
        evaluator = Evaluator(headless = headless, num_simulators = 1, terrain = config.TERRAIN, fitness_function = config.FITNESS_FUNCTION,
                              simulation_time = config.SIMULATION_TIME, sampling_frequency = config.SAMPLING_FREQUENCY,
                              simulation_timestep = config.SIMULATION_TIMESTEP, control_frequency = config.CONTROL_FREQUENCY,
                              writefiles = writefiles, record = writevideos, video_path = os.getcwd() + f"/MuJoCo_videos/MuJoCo_{irow}")

        # Show the robot.
        fitnesses, behavioral_measures = evaluator.evaluate([modular_robot])
        logging.info(f"Fitness Measured: {fitnesses[0]}")
        logging.info(f"X_distance Measured: {behavioral_measures[0]['x_distance']}")
        print("-----------------------------------------------")
    
if __name__ == "__main__":
    # run with arguments <algo> <mode> <file_name> <headless> <writefiles> <writevideos>!!!
    # --- Create/Empty directories for XMLs and PKLs
    for directory_path in ["MuJoCo_videos"]: # "RERUN\\XMLs", "RERUN\\PKLs", 
        if not os.path.exists(directory_path):
            # Create the directory and its parents if they don't exist
                os.makedirs(directory_path)
        else:
            for filename in os.listdir(directory_path):
                # Construct the full path
                file_path = os.path.join(directory_path, filename)

                # Check if it's a file
                if os.path.isfile(file_path):
                    # Remove the file
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    # Remove the directory
                    shutil.rmtree(file_path)
    # --- Rerun
    main()
