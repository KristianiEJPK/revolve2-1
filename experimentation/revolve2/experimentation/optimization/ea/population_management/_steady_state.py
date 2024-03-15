from typing import Callable, TypeVar

import numpy as np
import numpy.typing as npt

Genotype = TypeVar("Genotype")
Fitness = TypeVar("Fitness")


def steady_state(
    old_genotypes: list[Genotype],
    old_fitnesses: list[Fitness],
    new_genotypes: list[Genotype],
    new_fitnesses: list[Fitness],
    selection_function: Callable[
        [int, list[Genotype], list[Fitness]], npt.NDArray[np.float_]
    ],
) -> tuple[list[int], list[int]]:
    """
    Goal:
        Select `len(old_genotypes)` individuals using the provided selection function from 
            combined set of old and new individuals.
    -------------------------------------------------------------------------------------------
    Input:
        old_genotypes: Genotypes of the individuals in the parent population.
        old_fitnesses: Fitnesses of the individuals in the parent population.
        new_genotypes: Genotypes of the individuals from the offspring.
        new_fitnesses: Fitnesses of the individuals from the offspring.
        selection_function: Function that selects n individuals from a population based on their genotype and fitness. (n, genotypes, fitnesses) -> indices
    -------------------------------------------------------------------------------------------
    Output:
        Indices of selected individuals from parent population, indices of selected individuals from offspring.
    """
    # Assertment statements
    assert len(old_genotypes) == len(old_fitnesses), "Population and fitnesses must have the same length."
    assert len(new_genotypes) == len(new_fitnesses), "Population and fitnesses must have the same length."

    # Get population size
    population_size = len(old_genotypes)

    # Perform selection
    selection = selection_function(
        population_size, old_genotypes + new_genotypes, old_fitnesses + new_fitnesses
    )

    # Split selection into old and new individuals
    selected_old = [s for s in selection if s < len(old_fitnesses)]
    selected_new = [
        s - len(old_fitnesses) for s in selection if s >= len(old_fitnesses)
    ]
    return selected_old, selected_new, selection
