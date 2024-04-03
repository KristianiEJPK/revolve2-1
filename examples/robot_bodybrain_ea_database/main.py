import logging

# Import the configuration and os.
import config
import os

# Set algorithm
os.environ['ALGORITHM'] = config.ALGORITHM
os.environ['MAXPARTS'] = str(config.MAX_PARTS)
if os.environ["ALGORITHM"] == "CPPN":
    from genotype import Genotype
elif os.environ["ALGORITHM"] == "GRN":
    from genotype_grn import Genotype
else:
    raise ValueError("ALGORITHM must be either GRN or CPPN")

# Get other packages
from base import Base
from evaluator import Evaluator
from experiment import Experiment
from generation import Generation
from individual import Individual
import multineat
import numpy as np
import numpy.typing as npt
from population import Population
from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.optimization.ea import population_management, selection
from revolve2.experimentation.rng import make_rng, seed_from_time
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

def select_parents(
    rng: np.random.Generator,
    population: Population,
    offspring_size: int,
    Nparents: int,
    parent_tournament_size: int, random: bool = False
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
        random: Whether to select parents without selection pressure.
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
                lambda _, fitnesses: selection.tournament(rng, fitnesses, k = parent_tournament_size, random = random),
            )
            for _ in range(offspring_size)
        ],
    )


def select_survivors(
    rng: np.random.Generator,
    original_population: Population,
    offspring_population: Population,
    survivor_tournament_size: int,
    random: bool = False
) -> Population:
    """
    Goal:
        Select survivors using a tournament.
    -------------------------------------------------------------------------------------------
    Input:
        rng: Random number generator.
        original_population: The population the parents come from.
        offspring_population: The offspring.
        survivor_tournament_size: The size of the tournament to select survivors.
    -------------------------------------------------------------------------------------------
    Output: 
        A newly created population.
    """

    original_survivors, offspring_survivors, idx4selection = population_management.steady_state(
        [i.genotype for i in original_population.individuals],
        [i.fitness for i in original_population.individuals],
        [i.genotype for i in offspring_population.individuals],
        [i.fitness for i in offspring_population.individuals],
        lambda n, genotypes, fitnesses: selection.multiple_unique(
            n,
            genotypes,
            fitnesses,
            lambda _, fitnesses: selection.tournament(rng, fitnesses, k = survivor_tournament_size, random = random),
        ),
    )


    return Population(
        individuals=[
            Individual(
                genotype=original_population.individuals[i].genotype, fitness=original_population.individuals[i].fitness,
                energy_used = original_population.individuals[i].energy_used, 
                
                x_distance = original_population.individuals[i].x_distance, tot_xdistance = original_population.individuals[i].tot_xdistance, xmax = original_population.individuals[i].xmax,
                y_distance = original_population.individuals[i].y_distance, tot_ydistance = original_population.individuals[i].tot_ydistance, 
                
                min_dx = original_population.individuals[i].min_dx, dx25 = original_population.individuals[i].dx25, mean_dx = original_population.individuals[i].mean_dx,
                median_dx = original_population.individuals[i].median_dx, dx75 = original_population.individuals[i].dx75, max_dx = original_population.individuals[i].max_dx,
                std_dx = original_population.individuals[i].std_dx, 
                min_dy = original_population.individuals[i].min_dy, dy25 = original_population.individuals[i].dy25, mean_dy = original_population.individuals[i].mean_dy,
                median_dy = original_population.individuals[i].median_dy, dy75 = original_population.individuals[i].dy75, max_dy = original_population.individuals[i].max_dy,
                std_dy = original_population.individuals[i].std_dy,

                energy_used_min = original_population.individuals[i].energy_used_min, energy_used_25 = original_population.individuals[i].energy_used_25,
                energy_used_mean = original_population.individuals[i].energy_used_mean, energy_used_median = original_population.individuals[i].energy_used_median,
                energy_used_75 = original_population.individuals[i].energy_used_75, energy_used_max = original_population.individuals[i].energy_used_max,
                energy_used_std = original_population.individuals[i].energy_used_std,
                
                force_std_motor_min = original_population.individuals[i].force_std_motor_min, force_std_motor_25 = original_population.individuals[i].force_std_motor_25,
                force_std_motor_mean = original_population.individuals[i].force_std_motor_mean, force_std_motor_median = original_population.individuals[i].force_std_motor_median,
                force_std_motor_75 = original_population.individuals[i].force_std_motor_75, force_std_motor_max = original_population.individuals[i].force_std_motor_max,
                force_std_motor_std = original_population.individuals[i].force_std_motor_std,

                force_std_all_min = original_population.individuals[i].force_std_all_min, force_std_all_25 = original_population.individuals[i].force_std_all_25,
                force_std_all_mean = original_population.individuals[i].force_std_all_mean, force_std_all_median = original_population.individuals[i].force_std_all_median,
                force_std_all_75 = original_population.individuals[i].force_std_all_75, force_std_all_max = original_population.individuals[i].force_std_all_max,
                force_std_all_std = original_population.individuals[i].force_std_all_std,

                efficiency = original_population.individuals[i].efficiency, efficiency_min = original_population.individuals[i].efficiency_min,
                efficiency_25 = original_population.individuals[i].efficiency_25, efficiency_mean = original_population.individuals[i].efficiency_mean,
                efficiency_median = original_population.individuals[i].efficiency_median, efficiency_75 = original_population.individuals[i].efficiency_75,
                efficiency_max = original_population.individuals[i].efficiency_max, efficiency_std = original_population.individuals[i].efficiency_std,
                
                balance = original_population.individuals[i].balance
            )
            for i in original_survivors
        ]
        + [
            Individual(
                genotype=offspring_population.individuals[i].genotype, fitness=offspring_population.individuals[i].fitness,
                energy_used = offspring_population.individuals[i].energy_used,

                x_distance = offspring_population.individuals[i].x_distance, tot_xdistance = offspring_population.individuals[i].tot_xdistance, xmax = offspring_population.individuals[i].xmax,
                y_distance = offspring_population.individuals[i].y_distance, tot_ydistance = offspring_population.individuals[i].tot_ydistance,

                min_dx = offspring_population.individuals[i].min_dx, dx25 = offspring_population.individuals[i].dx25, mean_dx = offspring_population.individuals[i].mean_dx,
                median_dx = offspring_population.individuals[i].median_dx, dx75 = offspring_population.individuals[i].dx75, max_dx = offspring_population.individuals[i].max_dx,
                std_dx = offspring_population.individuals[i].std_dx,

                min_dy = offspring_population.individuals[i].min_dy, dy25 = offspring_population.individuals[i].dy25, mean_dy = offspring_population.individuals[i].mean_dy,
                median_dy = offspring_population.individuals[i].median_dy, dy75 = offspring_population.individuals[i].dy75, max_dy = offspring_population.individuals[i].max_dy,
                std_dy = offspring_population.individuals[i].std_dy,

                energy_used_min = offspring_population.individuals[i].energy_used_min, energy_used_25 = offspring_population.individuals[i].energy_used_25,
                energy_used_mean = offspring_population.individuals[i].energy_used_mean, energy_used_median = offspring_population.individuals[i].energy_used_median,
                energy_used_75 = offspring_population.individuals[i].energy_used_75, energy_used_max = offspring_population.individuals[i].energy_used_max,
                energy_used_std = offspring_population.individuals[i].energy_used_std,

                force_std_motor_min = offspring_population.individuals[i].force_std_motor_min, force_std_motor_25 = offspring_population.individuals[i].force_std_motor_25,
                force_std_motor_mean = offspring_population.individuals[i].force_std_motor_mean, force_std_motor_median = offspring_population.individuals[i].force_std_motor_median,
                force_std_motor_75 = offspring_population.individuals[i].force_std_motor_75, force_std_motor_max = offspring_population.individuals[i].force_std_motor_max,
                force_std_motor_std = offspring_population.individuals[i].force_std_motor_std,

                force_std_all_min = offspring_population.individuals[i].force_std_all_min, force_std_all_25 = offspring_population.individuals[i].force_std_all_25,
                force_std_all_mean = offspring_population.individuals[i].force_std_all_mean, force_std_all_median = offspring_population.individuals[i].force_std_all_median,
                force_std_all_75 = offspring_population.individuals[i].force_std_all_75, force_std_all_max = offspring_population.individuals[i].force_std_all_max,
                force_std_all_std = offspring_population.individuals[i].force_std_all_std,

                efficiency = offspring_population.individuals[i].efficiency, efficiency_min = offspring_population.individuals[i].efficiency_min,
                efficiency_25 = offspring_population.individuals[i].efficiency_25, efficiency_mean = offspring_population.individuals[i].efficiency_mean,
                efficiency_median = offspring_population.individuals[i].efficiency_median, efficiency_75 = offspring_population.individuals[i].efficiency_75,
                efficiency_max = offspring_population.individuals[i].efficiency_max, efficiency_std = offspring_population.individuals[i].efficiency_std,

                balance = offspring_population.individuals[i].balance
            )
            for i in offspring_survivors
        ]
    ), idx4selection


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

    print("Able to create random seed --> Test for DAS")

    # ---- Create and save the experiment instance.
    experiment = Experiment(rng_seed = rng_seed)
    logging.info("Saving experiment configuration.")
    with Session(dbengine) as session:
        session.add(experiment)
        session.commit()

    print("Able to create database --> Test for DAS")

    # ---- Intialize the evaluator that will be used to evaluate robots.
    evaluator = Evaluator(headless = True, num_simulators = config.NUM_SIMULATORS,
                          terrain = config.TERRAIN, fitness_function = config.FITNESS_FUNCTION,
                          simulation_time = config.SIMULATION_TIME, sampling_frequency = config.SAMPLING_FREQUENCY, 
                          simulation_timestep = config.SIMULATION_TIMESTEP, control_frequency = config.CONTROL_FREQUENCY)

    print("Able to evaluate --> Test for DAS")

    # ---- CPPN innovation databases.
    # Niet nodig voor GRN???
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    # ---- Create an initial population.
    logging.info("Generating initial population.")
    if os.environ["ALGORITHM"] == "CPPN":
        initial_genotypes = [
            Genotype.random(
                innov_db_body = innov_db_body,
                innov_db_brain = innov_db_brain,
                rng=rng, zdirection = config.ZDIRECTION, include_bias = config.CPPNBIAS,
                include_chain_length = config.CPPNCHAINLENGTH, include_empty = config.CPPNEMPTY,

            )
            for _ in range(config.POPULATION_SIZE)
        ]
    elif os.environ["ALGORITHM"] == "GRN":
        initial_genotypes = [
            Genotype.random(
                innov_db_brain = innov_db_brain,
                rng = rng, include_bias = config.CPPNBIAS,
            )
            for _ in range(config.POPULATION_SIZE)
        ]
    else:
        raise ValueError("ALGORITHM must be either GRN or CPPN")
    # Evaluate the initial population.
    logging.info("Evaluating initial population.")
    if os.environ["ALGORITHM"] == "CPPN":
        initial_fitnesses, behavioral_measures = evaluator.evaluate(
            [genotype.develop(zdirection = config.ZDIRECTION, include_bias = config.CPPNBIAS,
                include_chain_length = config.CPPNCHAINLENGTH, include_empty = config.CPPNEMPTY,
                max_parts = config.MAX_PARTS, mode_collision = config.MODE_COLLISION,
                mode_core_mult = config.MODE_CORE_MULT, mode_slots4face = config.MODE_SLOTS4FACE,
                mode_slots4face_all = config.MODE_SLOTS4FACE_ALL, mode_not_vertical = config.MODE_NOT_VERTICAL
                ) for genotype in initial_genotypes],
        )
    elif os.environ["ALGORITHM"] == "GRN":
        initial_fitnesses, behavioral_measures = evaluator.evaluate(
            [genotype.develop(include_bias = config.CPPNBIAS,
                max_parts = config.MAX_PARTS, mode_core_mult = config.MODE_CORE_MULT
                ) for genotype in initial_genotypes],
        )
    else:
        raise ValueError("ALGORITHM must be either GRN or CPPN")

    # Create a population of individuals, combining genotype with fitness.
    population = Population(
        individuals=[
            Individual(genotype=genotype, fitness=fitness, energy_used = behave_measure["energy_used"], 
                       
                x_distance = behave_measure["x_distance"], tot_xdistance = behave_measure["tot_xdistance"], xmax = behave_measure["xmax"],
                y_distance = behave_measure["y_distance"], tot_ydistance = behave_measure["tot_ydistance"],

                min_dx = behave_measure["min_dx"], dx25 = behave_measure["dx25"], mean_dx = behave_measure["mean_dx"],
                median_dx = behave_measure["median_dx"], dx75 = behave_measure["dx75"], max_dx = behave_measure["max_dx"],
                std_dx = behave_measure["std_dx"],

                min_dy = behave_measure["min_dy"], dy25 = behave_measure["dy25"], mean_dy = behave_measure["mean_dy"],
                median_dy = behave_measure["median_dy"], dy75 = behave_measure["dy75"], max_dy = behave_measure["max_dy"],
                std_dy = behave_measure["std_dy"],

                energy_used_min = behave_measure["energy_used_min"], energy_used_25 = behave_measure["energy_used_25"],
                energy_used_mean = behave_measure["energy_used_mean"], energy_used_median = behave_measure["energy_used_median"],
                energy_used_75 = behave_measure["energy_used_75"], energy_used_max = behave_measure["energy_used_max"],
                energy_used_std = behave_measure["energy_used_std"],

                force_std_motor_min = behave_measure["force_std_motor_min"], force_std_motor_25 = behave_measure["force_std_motor_25"],
                force_std_motor_mean = behave_measure["force_std_motor_mean"], force_std_motor_median = behave_measure["force_std_motor_median"],
                force_std_motor_75 = behave_measure["force_std_motor_75"], force_std_motor_max = behave_measure["force_std_motor_max"],
                force_std_motor_std = behave_measure["force_std_motor_std"],

                force_std_all_min = behave_measure["force_std_all_min"], force_std_all_25 = behave_measure["force_std_all_25"],
                force_std_all_mean = behave_measure["force_std_all_mean"], force_std_all_median = behave_measure["force_std_all_median"],
                force_std_all_75 = behave_measure["force_std_all_75"], force_std_all_max = behave_measure["force_std_all_max"],
                force_std_all_std = behave_measure["force_std_all_std"],

                efficiency = behave_measure["efficiency"], efficiency_min = behave_measure["efficiency_min"],
                efficiency_25 = behave_measure["efficiency_25"], efficiency_mean = behave_measure["efficiency_mean"],
                efficiency_median = behave_measure["efficiency_median"], efficiency_75 = behave_measure["efficiency_75"],
                efficiency_max = behave_measure["efficiency_max"], efficiency_std = behave_measure["efficiency_std"],
                
                balance = behave_measure["balance"]
                )
            for genotype, fitness, behave_measure in zip(
                initial_genotypes, initial_fitnesses, behavioral_measures, strict=True
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
    # Initialize random search boolean to True
    random_search = True
    while generation.generation_index < (config.NUM_GENERATIONS * 2): 
        # Log the generation number.
        logging.info(f"Generation {int(generation.generation_index / 2 + 1)}")

        # Check if random search should be performed
        if generation.generation_index == (config.NUM_RANDOM_SEARCH * 2):
            random_search = False

        # Create offspring.
        parents = select_parents(rng, population, config.OFFSPRING_SIZE, config.NPARENTS, config.PARENT_TOURNAMENT_SIZE, random = random_search)
        if os.environ["ALGORITHM"] == "CPPN":
            offspring_genotypes = [
                Genotype.crossover(
                    population.individuals[parent1_i].genotype,
                    population.individuals[parent2_i].genotype,
                    rng, config.CROSSOVER_PROBABILITY	
                ).mutate(innov_db_body, innov_db_brain, rng, config.MUTATION_PROBABILITY)
                for parent1_i, parent2_i in parents
            ]
        elif os.environ["ALGORITHM"] == "GRN":
            offspring_genotypes = [
                Genotype.crossover(
                    population.individuals[parent1_i].genotype,
                    population.individuals[parent2_i].genotype,
                    rng, config.CROSSOVER_PROBABILITY	
                ).mutate(innov_db_brain, rng, config.MUTATION_PROBABILITY)
                for parent1_i, parent2_i in parents
            ]
        else:
            raise ValueError("ALGORITHM must be either GRN or CPPN")
        # Evaluate the offspring.
        if os.environ["ALGORITHM"] == "CPPN":
            offspring_fitnesses, offspring_behavioral_measures = evaluator.evaluate(
                [genotype.develop(zdirection = config.ZDIRECTION, include_bias = config.CPPNBIAS,
                include_chain_length = config.CPPNCHAINLENGTH, include_empty = config.CPPNEMPTY,
                max_parts = config.MAX_PARTS, mode_collision = config.MODE_COLLISION,
                mode_core_mult = config.MODE_CORE_MULT, mode_slots4face = config.MODE_SLOTS4FACE,
                mode_slots4face_all = config.MODE_SLOTS4FACE_ALL, mode_not_vertical = config.MODE_NOT_VERTICAL
                ) for genotype in offspring_genotypes]
            )
        elif os.environ["ALGORITHM"] == "GRN":
            offspring_fitnesses, offspring_behavioral_measures = evaluator.evaluate(
                [genotype.develop(include_bias = config.CPPNBIAS,
                max_parts = config.MAX_PARTS, mode_core_mult = config.MODE_CORE_MULT
                ) for genotype in offspring_genotypes]
            )
        else:
            raise ValueError("ALGORITHM must be either GRN or CPPN")

        # Make an intermediate offspring population.
        offspring_population = Population(
            individuals=[
                Individual(genotype=genotype, fitness=fitness,  energy_used = behave_measure["energy_used"], 
                           
                    x_distance = behave_measure["x_distance"], tot_xdistance = behave_measure["tot_xdistance"], xmax = behave_measure["xmax"],
                    y_distance = behave_measure["y_distance"], tot_ydistance = behave_measure["tot_ydistance"],

                    min_dx = behave_measure["min_dx"], dx25 = behave_measure["dx25"], mean_dx = behave_measure["mean_dx"],
                    median_dx = behave_measure["median_dx"], dx75 = behave_measure["dx75"], max_dx = behave_measure["max_dx"],
                    std_dx = behave_measure["std_dx"],

                    min_dy = behave_measure["min_dy"], dy25 = behave_measure["dy25"], mean_dy = behave_measure["mean_dy"],
                    median_dy = behave_measure["median_dy"], dy75 = behave_measure["dy75"], max_dy = behave_measure["max_dy"],
                    std_dy = behave_measure["std_dy"],

                    energy_used_min = behave_measure["energy_used_min"], energy_used_25 = behave_measure["energy_used_25"],
                    energy_used_mean = behave_measure["energy_used_mean"], energy_used_median = behave_measure["energy_used_median"],
                    energy_used_75 = behave_measure["energy_used_75"], energy_used_max = behave_measure["energy_used_max"],
                    energy_used_std = behave_measure["energy_used_std"],

                    force_std_motor_min = behave_measure["force_std_motor_min"], force_std_motor_25 = behave_measure["force_std_motor_25"],
                    force_std_motor_mean = behave_measure["force_std_motor_mean"], force_std_motor_median = behave_measure["force_std_motor_median"],
                    force_std_motor_75 = behave_measure["force_std_motor_75"], force_std_motor_max = behave_measure["force_std_motor_max"],
                    force_std_motor_std = behave_measure["force_std_motor_std"],

                    force_std_all_min = behave_measure["force_std_all_min"], force_std_all_25 = behave_measure["force_std_all_25"],
                    force_std_all_mean = behave_measure["force_std_all_mean"], force_std_all_median = behave_measure["force_std_all_median"],
                    force_std_all_75 = behave_measure["force_std_all_75"], force_std_all_max = behave_measure["force_std_all_max"],
                    force_std_all_std = behave_measure["force_std_all_std"],

                    efficiency = behave_measure["efficiency"], efficiency_min = behave_measure["efficiency_min"],
                    efficiency_25 = behave_measure["efficiency_25"], efficiency_mean = behave_measure["efficiency_mean"],
                    efficiency_median = behave_measure["efficiency_median"], efficiency_75 = behave_measure["efficiency_75"],
                    efficiency_max = behave_measure["efficiency_max"], efficiency_std = behave_measure["efficiency_std"],
                    
                    balance = behave_measure["balance"])
                for genotype, fitness, behave_measure in zip(offspring_genotypes, offspring_fitnesses, offspring_behavioral_measures)
            ]
        )


        # Create 'fake' generation for offspring and save it to the database.
        generation = Generation(
            experiment = experiment,
            generation_index=generation.generation_index + 1,
            population = offspring_population,
        )

        logging.info("Saving offspring.")
        with Session(dbengine, expire_on_commit=False) as session:
            session.add(generation)
            session.commit()

        # Create the next population by selecting survivors.
        population, idx4selection = select_survivors(
            rng,
            population,
            offspring_population, config.SURVIVOR_TOURNAMENT_SIZE, random = random_search
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
