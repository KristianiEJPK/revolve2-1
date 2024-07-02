"""Rerun a chosen robot between all experiments."""
import logging
import os
import sys
import config

# Add these imports
from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from sqlalchemy.orm import Session
from sqlalchemy import select

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set algorithm, mode, file name, and individual identifier from command line arguments.
algo = sys.argv[1]
mode = sys.argv[2]
file_name = sys.argv[3]
headless = sys.argv[4]
writefiles = sys.argv[5]
writevideos = sys.argv[6]
individual_id = int(sys.argv[7])  # Changed to individual_id and converted to int

# ... (keep the rest of the imports and assertions)

def main() -> None:
    """Perform the rerun."""
    setup_logging()

    try:
        # Load the specified individual from the database.
        dbengine = open_database_sqlite(
            config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS
        )
        logging.info(f"Successfully opened database: {config.DATABASE_FILE}")

        with Session(dbengine) as ses:
            try:
                row = ses.execute(
                    select(Individual, Generation.experiment_id, Generation.generation_index)
                    .join(Generation, Individual.generation_id == Generation.id)
                    .where(Individual.id == individual_id)
                ).first()
                
                if row is None:
                    logging.error(f"No individual found with id: {individual_id}")
                    return
                
                logging.info(f"Successfully retrieved individual with id: {individual_id}")
            except Exception as e:
                logging.error(f"Error executing database query: {str(e)}")
                return

        individual, exp_id, gen_index = row

        logging.info(f"Experiment ID: {exp_id}")
        logging.info(f"Generation Index: {gen_index}")

        # Parse the genotype from the stored string
        genotype = Genotype.parse(individual.genotype)

        if os.environ["ALGORITHM"] == "CPPN":
            modular_robot = genotype.develop(zdirection=config.ZDIRECTION, include_bias=config.CPPNBIAS,
                                             include_chain_length=config.CPPNCHAINLENGTH, include_empty=config.CPPNEMPTY,
                                             max_parts=config.MAX_PARTS, mode_collision=config.MODE_COLLISION,
                                             mode_core_mult=config.MODE_CORE_MULT, mode_slots4face=config.MODE_SLOTS4FACE,
                                             mode_slots4face_all=config.MODE_SLOTS4FACE_ALL, mode_not_vertical=config.MODE_NOT_VERTICAL)
        elif os.environ["ALGORITHM"] in ["GRN", "GRN_system", "GRN_system_adv"]:
            modular_robot = genotype.develop(include_bias=config.CPPNBIAS, max_parts=config.MAX_PARTS, mode_core_mult=config.MODE_CORE_MULT)
        else:
            raise ValueError("ALGORITHM must be either GRN or CPPN")

        logging.info(f"Fitness: {individual.fitness}")
        logging.info(f"Energy used: {individual.energy_used}")
        logging.info(f"Efficiency: {individual.efficiency}")
        logging.info(f"X distance: {individual.x_distance}")
        logging.info(f"Y distance: {individual.y_distance}")
        logging.info(f"Body ID: {individual.body_id}")

        # Create the evaluator.
        evaluator = Evaluator(headless=headless, num_simulators=1, terrain=config.TERRAIN, fitness_function=config.FITNESS_FUNCTION,
                              simulation_time=config.SIMULATION_TIME, sampling_frequency=config.SAMPLING_FREQUENCY,
                              simulation_timestep=config.SIMULATION_TIMESTEP, control_frequency=config.CONTROL_FREQUENCY,
                              writefiles=writefiles, record=writevideos, video_path=os.getcwd() + f"/MuJoCo_videos/MuJoCo_{individual_id}")

        # Show the robot.
        fitnesses, behavioral_measures, ids = evaluator.evaluate([modular_robot])
        logging.info(f"Fitness Measured: {fitnesses[0]}")
        logging.info(f"X_distance Measured: {behavioral_measures[0]['x_distance']}")

        assert ids[0] == individual.body_id, "Body ID measured does not match the one in the database"
        logging.info(f"Body ID Measured: {ids[0]}")
        print("-----------------------------------------------")

    except Exception as e:
        logging.error(f"Error opening or querying database: {str(e)}")
        return

if __name__ == "__main__":
    # run with arguments <algo> <mode> <file_name> <headless> <writefiles> <writevideos> <individual_id>!!!
    # --- Create/Empty directories for XMLs and PKLs
    for directory_path in ["MuJoCo_videos"]:  # "RERUN\\XMLs", "RERUN\\PKLs",
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
