from typing import TypeVar

import numpy as np

Fitness = TypeVar("Fitness")


def tournament(rng: np.random.Generator, fitnesses: list[Fitness], k: int) -> int:
    """
    Goal:
        Perform tournament selection and return the index of the best individual.
    -------------------------------------------------------------------------------------------
    Input:
        rng: Random number generator.
        fitnesses: List of finesses of individuals that joint the tournamente.
        k: Amount of individuals to participate in tournament.
    -------------------------------------------------------------------------------------------
    Output:
        The index of te individual that won the tournament.
    """
    # Assert that there are enough individuals to perform a tournament
    assert len(fitnesses) >= k, "Not enough individuals to perform a tournament."

    # Select random participants
    participant_indices = rng.choice(range(len(fitnesses)), size=k)
    return max(participant_indices, key=lambda i: fitnesses[i])  # type: ignore[no-any-return]
