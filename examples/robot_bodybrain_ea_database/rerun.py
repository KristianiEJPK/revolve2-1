"""Rerun the best robot between all experiments."""

import logging

import config
from evaluator import Evaluator
from genotype import Genotype
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
            select(Genotype, Individual.fitness)
            .join_from(Genotype, Individual, Genotype.id == Individual.genotype_id)
            .order_by(Individual.fitness.desc()).limit(200)
        ).fetchall() #.one()
        #assert row is not None
        
    for row in rows[0:]:
        genotype = row[0]
        fitness = row[1]

        modular_robot = genotype.develop(include_bias = config.CPPNBIAS)

        logging.info(f"Best fitness: {fitness}")

        # Create the evaluator.
        evaluator = Evaluator(headless = False, num_simulators = 1, terrain = config.TERRAIN, fitness_function = config.FITNESS_FUNCTION,
                              simulation_time = config.SIMULATION_TIME, sampling_frequency = config.SAMPLING_FREQUENCY,
                              simulation_timestep = config.SIMULATION_TIMESTEP, control_frequency = config.CONTROL_FREQUENCY)

        # Show the robot.
        evaluator.evaluate([modular_robot])

        


if __name__ == "__main__":
    main()
