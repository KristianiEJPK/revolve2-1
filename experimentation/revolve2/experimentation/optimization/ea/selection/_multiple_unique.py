from typing import Callable, TypeVar

import numpy as np
import numpy.typing as npt

TIndividual = TypeVar("TIndividual")
TFitness = TypeVar("TFitness")


def multiple_unique(
    selection_size: int,
    population: list[TIndividual],
    fitnesses: list[TFitness],
    selection_function: Callable[[list[TIndividual], list[TFitness]], int],
) -> npt.NDArray[np.float_]:
    """
    Goal:
        Select multiple distinct individuals from a population using the provided selection function.
    -------------------------------------------------------------------------------------------
    Input:
        selection_size: Amount of of individuals to select.
        population: List of individuals to select from.
        fitnesses: Fitnesses of the population.
        selection_function: Function that select a single individual from a population. ([TIndividual], [TFitness]) -> index.
    -------------------------------------------------------------------------------------------
    Output:
        Indices of the selected individuals.
    """
    # Assertment statements
    assert len(population) == len(fitnesses), "Population and fitnesses must have the same length."
    assert selection_size <= len(population), "Selection size must be smaller or equal to the population size."

    # Initialize
    selected_individuals = []

    # Select individuals
    for _ in range(selection_size):
        new_individual = False
        while new_individual is False:
            selected_individual = selection_function(population, fitnesses)
            if selected_individual not in selected_individuals:
                selected_individuals.append(selected_individual)
                new_individual = True
    return np.array(selected_individuals)
