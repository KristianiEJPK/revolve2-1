from __future__ import annotations

import multineat
import numpy as np
import sqlalchemy.orm as orm
from sqlalchemy import event
from sqlalchemy.engine import Connection
from typing_extensions import Self

from revolve2.modular_robot.body.v2 import BodyV2

from ..._multineat_rng_from_random import multineat_rng_from_random
from ..._random_multineat_genotype import random_multineat_genotype
from .._multineat_params import get_multineat_params
from ._body_develop import develop


class BodyGenotypeOrmV2(orm.MappedAsDataclass, kw_only=True):
    """Goal:
        SQLAlchemy model for a CPPNWIN body genotype."""

    # Initial number of mutations
    _NUM_INITIAL_MUTATIONS = 10
    # Multineat parameters
    _MULTINEAT_PARAMS = get_multineat_params()

    # Body genotype
    body: multineat.Genome

    # Serialized body
    _serialized_body: orm.Mapped[str] = orm.mapped_column(
        "serialized_body", init=False, nullable=False
    )

    @classmethod
    def random_body(
        cls,
        innov_db: multineat.InnovationDatabase,
        rng: np.random.Generator,
        zdirection: bool, include_bias: bool, include_chain_length: bool,
        include_empty: bool,
    ) -> BodyGenotypeOrmV2:
        """
        Goal:
            Create a random genotype.
        -------------------------------------------------------------------------------------------
        Input:
            innov_db: Multineat innovation database. See Multineat library.
            rng: Random number generator.
            zdirection: Whether to include the z direction  as input for CPPN.
            include_bias: Whether to include the bias as input for CPPN.
            include_chain_length: Whether to include the chain length as input for CPPN.
            include_empty: Whether to include the empty module output for CPPN.
        -------------------------------------------------------------------------------------------
        Output:
            The created genotype.
        """
        # Create a multineat rng and seed it with the numpy rng state
        multineat_rng = multineat_rng_from_random(rng)

        # Number of inputs
        num_inputs = 2 # pos_x, pos_y
        if zdirection: # pos_z
            num_inputs += 1 
        if include_bias: # bias(always 1)
            num_inputs += 1
        if include_chain_length: # chain_length
            num_inputs += 1

        # Number of outputs
        num_outputs = 4 # brick, activehinge, rot0, rot90
        if include_empty:
            num_outputs += 1

        # Create a random body
        body = random_multineat_genotype(
            innov_db=innov_db,
            rng=multineat_rng,
            multineat_params=cls._MULTINEAT_PARAMS,
            output_activation_func=multineat.ActivationFunction.TANH,
            num_inputs = num_inputs,
            num_outputs = num_outputs,
            num_initial_mutations=cls._NUM_INITIAL_MUTATIONS,
        )

        return BodyGenotypeOrmV2(body=body)

    def mutate_body(
        self,
        innov_db: multineat.InnovationDatabase,
        rng: np.random.Generator,
    ) -> BodyGenotypeOrmV2:
        """
        Mutate this genotype.

        This genotype will not be changed; a mutated copy will be returned.

        :param innov_db: Multineat innovation database. See Multineat library.
        :param rng: Random number generator.
        :returns: A mutated copy of the provided genotype.
        """
        multineat_rng = multineat_rng_from_random(rng)

        return BodyGenotypeOrmV2(
            body=self.body.MutateWithConstraints(
                False,
                multineat.SearchMode.BLENDED,
                innov_db,
                self._MULTINEAT_PARAMS,
                multineat_rng,
            )
        )

    @classmethod
    def crossover_body(
        cls,
        parent1: Self,
        parent2: Self,
        rng: np.random.Generator,
    ) -> BodyGenotypeOrmV2:
        """
        Perform crossover between two genotypes.

        :param parent1: The first genotype.
        :param parent2: The second genotype.
        :param rng: Random number generator.
        :returns: A newly created genotype.
        """
        multineat_rng = multineat_rng_from_random(rng)

        return BodyGenotypeOrmV2(
            body=parent1.body.MateWithConstraints(
                parent2.body,
                False,
                False,
                multineat_rng,
                cls._MULTINEAT_PARAMS,
            )
        )

    def develop_body(self: object, querying_seed = int) -> BodyV2:
        """
        Develop the genotype into a modular robot.

        :returns: The created robot.
        """
        return develop(self.body, querying_seed)


@event.listens_for(BodyGenotypeOrmV2, "before_update", propagate=True)
@event.listens_for(BodyGenotypeOrmV2, "before_insert", propagate=True)
def _update_serialized_body(
    mapper: orm.Mapper[BodyGenotypeOrmV2],
    connection: Connection,
    target: BodyGenotypeOrmV2,
) -> None:
    target._serialized_body = target.body.Serialize()
    pass

@event.listens_for(BodyGenotypeOrmV2, "load", propagate=True)
def _deserialize_body(target: BodyGenotypeOrmV2, context: orm.QueryContext) -> None:
    body = multineat.Genome()
    body.Deserialize(target._serialized_body)
    target.body = body
