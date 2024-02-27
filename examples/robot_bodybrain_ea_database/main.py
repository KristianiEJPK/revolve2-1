"""Main script for the example."""

import logging

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

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.optimization.ea import population_management, selection
from revolve2.experimentation.rng import make_rng, seed_from_time


def select_parents(
    rng: np.random.Generator,
    population: Population,
    offspring_size: int,
    Nparents: int,
    parent_tournament_size: int,
) -> npt.NDArray[np.float_]:
    """
    Goal:
        Select pairs of parents using a tournament.
    -------------------------------------------------------------------------------------------
    Input:
        rng: Random number generator.
        population: The population to select from.
        offspring_size: The number of parent pairs to select.
        Nparents: The number of parents to select.
        parent_tournament_size: The size of the tournament to select parents.
    -------------------------------------------------------------------------------------------
    Output:
        Pairs of indices of selected parents. offspring_size x 2 ints.
    """
    return np.array(
        [
            selection.multiple_unique(
                Nparents,
                [individual.genotype for individual in population.individuals],
                [individual.fitness for individual in population.individuals],
                lambda _, fitnesses: selection.tournament(rng, fitnesses, k = parent_tournament_size),
            )
            for _ in range(offspring_size)
        ],
    )


def select_survivors(
    rng: np.random.Generator,
    original_population: Population,
    offspring_population: Population,
    survivor_tournament_size: int,
) -> Population:
    """
    Goal:
        Select survivors using a tournament.
    -------------------------------------------------------------------------------------------
    Input:
        rng: Random number generator.
        original_population: The population the parents come from.
        offspring_population: The offspring.
    -------------------------------------------------------------------------------------------
    Output: 
        A newly created population.
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
            lambda _, fitnesses: selection.tournament(rng, fitnesses, k = survivor_tournament_size),
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
    Goal:
        Return the best robot between the population and the current best individual.
    -------------------------------------------------------------------------------------------
    Input:
        current_best: The current best individual.
        population: The population.
    -------------------------------------------------------------------------------------------
    Output:
        The best individual.
    """
    return max(
        population + [] if current_best is None else [current_best],
        key=lambda x: x.fitness,
    )


def run_experiment(dbengine: Engine) -> None:
    """
    Goal:
        Run an experiment.
    -------------------------------------------------------------------------------------------
    Input:
        dbengine: An openened database with matching initialize database structure.
    """
    logging.info("----------------")
    logging.info("Start experiment")

    # ---- Set up the random number generator.
    rng_seed = seed_from_time()
    rng = make_rng(rng_seed)

    # ---- Create and save the experiment instance.
    experiment = Experiment(rng_seed=rng_seed)
    logging.info("Saving experiment configuration.")
    with Session(dbengine) as session:
        session.add(experiment)
        session.commit()

    # ---- Intialize the evaluator that will be used to evaluate robots.
    evaluator = Evaluator(headless=True, num_simulators = config.NUM_SIMULATORS, 
                          terrain = config.TERRAIN, fitness_function = config.FITNESS_FUNCTION,
                          simulation_time = config.SIMULATION_TIME, sampling_frequency = config.SAMPLING_FREQUENCY, 
                          simulation_timestep = config.SIMULATION_TIMESTEP, control_frequency = config.CONTROL_FREQUENCY)

    # ---- CPPN innovation databases.
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    # ---- Create an initial population.
    logging.info("Generating initial population.")
    initial_genotypes = [
        Genotype.random(
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            rng=rng, zdirection = config.ZDIRECTION, include_bias = config.CPPNBIAS,
            include_chain_length = config.CPPNCHAINLENGTH, include_empty = config.CPPNEMPTY,

        )
        for _ in range(config.POPULATION_SIZE)
    ]
    
    # Evaluate the initial population.
    logging.info("Evaluating initial population.")
    initial_fitnesses = evaluator.evaluate(
        [genotype.develop(zdirection = config.ZDIRECTION, include_bias = config.CPPNBIAS,
            include_chain_length = config.CPPNCHAINLENGTH, include_empty = config.CPPNEMPTY,
            max_parts = config.MAX_PARTS, mode_collision = config.MODE_COLLISION,
            mode_core_mult = config.MODE_CORE_MULT, mode_slots4face = config.MODE_SLOTS4FACE,
            mode_slots4face_all = config.MODE_SLOTS4FACE_ALL, mode_not_vertical = config.MODE_NOT_VERTICAL
            ) for genotype in initial_genotypes],
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
        experiment=experiment, generation_index=0, population=population
    )
    logging.info("Saving generation.")
    with Session(dbengine, expire_on_commit=False) as session:
        session.add(generation)
        session.commit()

    # Start the actual optimization process.
    logging.info("Start optimization process.")
    while generation.generation_index < config.NUM_GENERATIONS:
        logging.info(
            f"Generation {generation.generation_index + 1} / {config.NUM_GENERATIONS}."
        )

        # Create offspring.
        parents = select_parents(rng, population, config.OFFSPRING_SIZE, config.NPARENTS, config.PARENT_TOURNAMENT_SIZE)
        offspring_genotypes = [
            Genotype.crossover(
                population.individuals[parent1_i].genotype,
                population.individuals[parent2_i].genotype,
                rng, config.CROSSOVER_PROBABILITY	
            ).mutate(innov_db_body, innov_db_brain, rng, config.MUTATION_PROBABILITY)
            for parent1_i, parent2_i in parents
        ]

        # Evaluate the offspring.
        offspring_fitnesses = evaluator.evaluate(
            [genotype.develop(zdirection = config.ZDIRECTION, include_bias = config.CPPNBIAS,
            include_chain_length = config.CPPNCHAINLENGTH, include_empty = config.CPPNEMPTY,
            max_parts = config.MAX_PARTS, mode_collision = config.MODE_COLLISION,
            mode_core_mult = config.MODE_CORE_MULT, mode_slots4face = config.MODE_SLOTS4FACE,
            mode_slots4face_all = config.MODE_SLOTS4FACE_ALL, mode_not_vertical = config.MODE_NOT_VERTICAL
            ) for genotype in offspring_genotypes]
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
            offspring_population, config.SURVIVOR_TOURNAMENT_SIZE
        )

        # Make it all into a generation and save it to the database.
        generation = Generation(
            experiment=experiment,
            generation_index=generation.generation_index + 1,
            population=population,
        )
        logging.info("Saving generation.")
        with Session(dbengine, expire_on_commit=False) as session:
            session.add(generation)
            session.commit()

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

    # Run the experiment several times.
    for _ in range(config.NUM_REPETITIONS):
        run_experiment(dbengine)


if __name__ == "__main__":
    main()
