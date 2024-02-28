"""Genotype class."""

from __future__ import annotations

import multineat
import numpy as np
from base import Base

from revolve2.ci_group.genotypes.cppnwin.modular_robot import BrainGenotypeCpgOrm
from revolve2.ci_group.genotypes.cppnwin.modular_robot.v2 import BodyGenotypeOrmV2GRN
from revolve2.ci_group.genotypes.cppnwin.modular_robot.v2 import BodyMappingSeedOrmV2
from revolve2.experimentation.database import HasId
from revolve2.modular_robot import ModularRobot

class Genotype(Base, HasId, BodyGenotypeOrmV2GRN, BrainGenotypeCpgOrm, BodyMappingSeedOrmV2):
    """SQLAlchemy model for a genotype for a modular robot body and brain."""

    __tablename__ = "genotype"

    @classmethod
    def random(
        cls,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
        rng: np.random.Generator,
        zdirection: bool, include_bias: bool, include_chain_length: bool,
        include_empty: bool,
    ) -> Genotype:
        """
        Goal:
            Create a random genotype.
        -------------------------------------------------------------------------------------------
        Input:
            innov_db_body: Multineat innovation database for the body. See Multineat library.
            innov_db_brain: Multineat innovation database for the brain. See Multineat library.
            rng: Random number generator for CPPN.
            zdirection: Whether to include the z direction  as input for CPPN.
            include_bias: Whether to include the bias  as input for CPPN.
            include_chain_length: Whether to include the chain length as input for CPPN.
            include_empty: Whether to include the empty module output  as input for CPPN
        -------------------------------------------------------------------------------------------
        Output:
            The created genotype: (base class sqlalchemy, hashid sqlalchemy, body genotype, brain
                genotype, mapping seed).
        """
        # Set random body and brain
        body = cls.random_body(rng)
        brain = cls.random_brain(innov_db_brain, rng, include_bias)

        # Set random mapping seed
        mapping_seed = rng.integers(0, 2 ** 32)

        return Genotype(body = body.body, brain = brain.brain, mapping_seed = mapping_seed)

    def mutate(
        self,
        innov_db_brain: multineat.InnovationDatabase,
        rng: np.random.Generator,
        mutation_prob: float,
    ) -> Genotype:
        """
        Goal:
            Mutate the genotype. Note: The genotype will not be changed; a mutated 
            copy will be returned.
        -------------------------------------------------------------------------------------------
        Input:
            self: The genotype to mutate (base class sqlalchemy, hashid sqlalchemy, body genotype,
                brain genotype, mapping seed).
            innov_db_brain: Multineat innovation database for the brain. See Multineat library.
            rng: Random number generator for CPPN.
            mutation_prob: The probability of mutation.
        -------------------------------------------------------------------------------------------
        Output:
            A mutated copy of the provided genotype: (base class sqlalchemy, hashid sqlalchemy, 
                body genotype, brain genotype, mapping seed).
        """
        # Get random number
        random_number = rng.uniform(0, 1)

        if random_number > mutation_prob:
            return self
        else:
            # Mutate body and brain
            body = self.mutate_body(rng)
            brain = self.mutate_brain(innov_db_brain, rng)

            return Genotype(body=body.body, brain=brain.brain, mapping_seed = self.mapping_seed)

    @classmethod
    def crossover(
        cls,
        parent1: Genotype,
        parent2: Genotype,
        rng: np.random.Generator,
        crossover_prob: float,
    ) -> Genotype:
        """
        Goal:
            Perform crossover between two genotypes.
        -------------------------------------------------------------------------------------------
        Input:
            parent1: The first genotype.
            parent2: The second genotype.
            rng: Random number generator for CPPN.
            crossover_prob: The probability of crossover.
        -------------------------------------------------------------------------------------------
        Output:
            A newly created genotype: (base class sqlalchemy, hashid sqlalchemy, body genotype, 
                brain genotype, mapping seed).
        """
        # Get random number
        random_number = rng.uniform(0, 1)
        if random_number > crossover_prob:
            return parent1
        else:
            # Perform crossover for body and brain
            body = cls.crossover_body(parent1, parent2, rng)
            brain = cls.crossover_brain(parent1, parent2, rng)

            # Set mapping seed to the first parent's mapping seed
            mapping_seed = parent1.mapping_seed

            return Genotype(body=body.body, brain=brain.brain, mapping_seed = mapping_seed)

    def develop(self, zdirection, include_bias, include_chain_length, include_empty,
                max_parts, mode_collision, mode_core_mult, mode_slots4face,
            mode_slots4face_all, mode_not_vertical) -> ModularRobot:
        """
        Goal:
            Develop the genotype into a modular robot.
        -------------------------------------------------------------------------------------------
        Input:
            self: The genotype to develop (base class sqlalchemy, hashid sqlalchemy, body genotype,
                brain genotype, mapping seed).
            zdirection: Whether to include the z direction as input for CPPN.
            include_bias: Whether to include the bias as input for CPPN.
            include_chain_length: Whether to include the chain length as input for CPPN.
            include_empty: Whether to include the empty module output as input for CPPN.
        -------------------------------------------------------------------------------------------
        Output:
            The created robot: ModularRobot.
        """
        # Develop body and brain
        body = self.develop_body(max_parts, querying_seed = self.mapping_seed,)
        brain = self.develop_brain(body = body, include_bias = include_bias) # Deze al gedaan!

        return ModularRobot(body = body, brain = brain)
