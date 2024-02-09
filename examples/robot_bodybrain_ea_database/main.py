"""Main script for the example."""

import logging
import time

import config
import multineat
import numpy as np
import numpy.typing as npt
from base import Base
from evaluator import Evaluator
from experiment import Experiment
from generation import Generation
from genotype import Genotype
from individual import Individual
from population import Population
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from threading import Thread
from queue import Queue

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.optimization.ea import population_management, selection
from revolve2.experimentation.rng import make_rng, seed_from_time


class DatabaseHandler:
    """A class to handle database commits."""
    _commit_queue: Queue
    _dbengine: Engine
    _commit_thread: Thread

    _is_active: bool
    _terminating: bool

    def __init__(self, dbengine: Engine) -> None:
        """
        Initialize the Handler.

        :param dbengine: The database engine.
        """
        self._is_active = True
        self._terminating = False
        self._dbengine = dbengine
        self._commit_queue = Queue()

        self._commit_thread = Thread(target=self._commit)
        self._commit_thread.start()

    def _commit(self) -> None:
        """Bulk save objects to database."""
        while self._is_active:
            if not self._commit_queue.empty():
                logging.info("Saving element.")
                with Session(self._dbengine, expire_on_commit=False) as session:
                    session.add(self._commit_queue.get())
                    session.commit()
            if self._terminating:
                self.disable()
        else:
            logging.info("Database operations concluded.")

    def add_element(self, element: Base) -> None:
        """
        Add an element to the queue.

        :param element: The element to add.
        """

        self._commit_queue.put(element)

    def disable(self) -> None:
        """Disable the Database Handler."""
        if self._commit_queue.empty():
            self._is_active = False
        else:
            self._terminating = True



def select_parents(
        rng: np.random.Generator,
        population: Population,
        offspring_size: int,
) -> npt.NDArray[np.float_]:
    """
    Select pairs of parents using a tournament.

    :param rng: Random number generator.
    :param population: The population to select from.
    :param offspring_size: The number of parent pairs to select.
    :returns: Pairs of indices of selected parents. offspring_size x 2 ints.
    """
    return np.array(
        [
            selection.multiple_unique(
                2,
                [individual.genotype for individual in population.individuals],
                [individual.fitness for individual in population.individuals],
                lambda _, fitnesses: selection.tournament(rng, fitnesses, k=1),
            )
            for _ in range(offspring_size)
        ],
    )


def select_survivors(
        rng: np.random.Generator,
        original_population: Population,
        offspring_population: Population,
) -> Population:
    """
    Select survivors using a tournament.

    :param rng: Random number generator.
    :param original_population: The population the parents come from.
    :param offspring_population: The offspring.
    :returns: A newly created population.
    """
    original_survivors, offspring_survivors = population_management.steady_state(
        [i.genotype for i in original_population.individuals],
        [i.fitness for i in original_population.individuals],
        [i.genotype for i in offspring_population.individuals],
        [i.fitness for i in offspring_population.individuals],
        lambda n, genotypes, fitnesses: selection.multiple_unique(
            n,
            genotypes,
            fitnesses,
            lambda _, fitnesses: selection.tournament(rng, fitnesses, k=2),
        ),
    )

    return Population(
        individuals=[
                        Individual(
                            genotype=original_population.individuals[i].genotype,
                            fitness=original_population.individuals[i].fitness,
                        )
                        for i in original_survivors
                    ]
                    + [
                        Individual(
                            genotype=offspring_population.individuals[i].genotype,
                            fitness=offspring_population.individuals[i].fitness,
                        )
                        for i in offspring_survivors
                    ]
    )


def find_best_robot(
        current_best: Individual | None, population: list[Individual]
) -> Individual:
    """
    Return the best robot between the population and the current best individual.

    :param current_best: The current best individual.
    :param population: The population.
    :returns: The best individual.
    """
    return max(
        population + [] if current_best is None else [current_best],
        key=lambda x: x.fitness,
    )


def run_experiment(database_handler) -> None:
    """
    Run an experiment.

    :param dbengine: An openened database with matching initialize database structure.
    """


    logging.info("----------------")
    logging.info("Start experiment")

    # Set up the random number generator.
    rng_seed = seed_from_time()
    rng = make_rng(rng_seed)

    # Create and save the experiment instance.
    experiment = Experiment(rng_seed=rng_seed)
    database_handler.add_element(experiment)
    #with Session(dbengine, expire_on_commit=False) as session:
    #    session.add(experiment)
    #    session.commit()

    # Intialize the evaluator that will be used to evaluate robots.
    evaluator = Evaluator(headless=True, num_simulators=config.NUM_SIMULATORS)

    # CPPN innovation databases.
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    # Create an initial population.
    logging.info("Generating initial population.")
    initial_genotypes = [
        Genotype.random(
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            rng=rng,
        )
        for _ in range(config.POPULATION_SIZE)
    ]

    # Evaluate the initial population.
    logging.info("Evaluating initial population.")
    initial_fitnesses = evaluator.evaluate(
        [genotype.develop() for genotype in initial_genotypes]
    )

    # Create a population of individuals, combining genotype with fitness.
    population = Population(
        individuals=[
            Individual(genotype=genotype, fitness=fitness)
            for genotype, fitness in zip(
                initial_genotypes, initial_fitnesses, strict=True
            )
        ]
    )

    # Finish the zeroth generation and save it to the database.
    generation = Generation(
        experiment=experiment,
        generation_index=0,
        population=population
    )

    database_handler.add_element(generation)
    #with Session(dbengine, expire_on_commit=False) as session:
     #   session.add(generation)
    #    session.commit()

    # Start the actual optimization process.
    logging.info("Start optimization process.")
    while generation.generation_index < config.NUM_GENERATIONS:
        logging.info(
            f"Generation {generation.generation_index + 1} / {config.NUM_GENERATIONS}."
        )

        # Create offspring.
        parents = select_parents(rng, population, config.OFFSPRING_SIZE)
        offspring_genotypes = [
            Genotype.crossover(
                population.individuals[parent1_i].genotype,
                population.individuals[parent2_i].genotype,
                rng,
            ).mutate(innov_db_body, innov_db_brain, rng)
            for parent1_i, parent2_i in parents
        ]

        # Evaluate the offspring.
        offspring_fitnesses = evaluator.evaluate(
            [genotype.develop() for genotype in offspring_genotypes]
        )

        # Make an intermediate offspring population.
        offspring_population = Population(
            individuals=[
                Individual(genotype=genotype, fitness=fitness)
                for genotype, fitness in zip(offspring_genotypes, offspring_fitnesses)
            ]
        )

        # Create the next population by selecting survivors.
        population = select_survivors(
            rng,
            population,
            offspring_population,
        )

        # Make it all into a generation and save it to the database.
        generation = Generation(
            experiment=experiment,
            generation_index=generation.generation_index + 1,
            population=population,
        )
        #with Session(dbengine, expire_on_commit=False) as session:
         #   session.add(generation)
        #    session.commit()
        database_handler.add_element(generation)


def main() -> None:
    """Run the program."""
    # Set up logging.
    setup_logging(file_name="log.txt")

    # Open the database, only if it does not already exists.
    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.NOT_EXISTS_AND_CREATE
    )
    # Create the structure of the database.
    Base.metadata.create_all(dbengine)

    database_handler = DatabaseHandler(dbengine)
    times = []
    # Run the experiment several times.
    for _ in range(config.NUM_REPETITIONS):
        start = time.time()
        run_experiment(database_handler)
        times.append(time.time() - start)
    database_handler.disable()
    print(f"AVG Time: {sum(times) / len(times)}")


if __name__ == "__main__":
    main()
