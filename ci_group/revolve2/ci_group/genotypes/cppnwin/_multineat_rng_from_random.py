import multineat
import numpy as np


def multineat_rng_from_random(rng: np.random.Generator) -> multineat.RNG:
    """
    Goal:
        Create a multineat rng object from a numpy rng state.
    -------------------------------------------------------------------------------------------
    Input:
        rng: The numpy rng.
    -------------------------------------------------------------------------------------------
    Output:
        The multineat rng.
    """
    # Create a multineat rng and seed it with the numpy rng state
    multineat_rng = multineat.RNG()
    multineat_rng.Seed(rng.integers(0, 2**31))
    return multineat_rng
