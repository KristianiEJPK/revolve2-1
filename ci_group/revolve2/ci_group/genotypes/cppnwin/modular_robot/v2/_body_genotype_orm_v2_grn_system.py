from __future__ import annotations

from copy import deepcopy
import multineat
import numpy as np
import sqlalchemy.orm as orm
from sqlalchemy import event
from sqlalchemy.engine import Connection
from typing_extensions import Self

from revolve2.modular_robot.body.v2 import BodyV2
from ._body_develop_grn_system import DevelopGRN

class BodyGenotypeOrmV2GRN_system(orm.MappedAsDataclass, kw_only=True):
    """Goal:
        SQLAlchemy model for a CPPNWIN body genotype."""

    # Body genotype
    body: multineat.Genome

    # Serialized body
    _serialized_body: orm.Mapped[str] = orm.mapped_column(
        "serialized_body", init=False, nullable=False
    )

    @classmethod
    def random_body(self: object, rng: np.random.Generator) -> BodyGenotypeOrmV2GRN_system:
        # Set genome size
        genome_size = 150 + 1 + 6
        # Set random genotype
        genotype = [round(rng.uniform(0, 1), 2) for _ in range(genome_size)]

        return BodyGenotypeOrmV2GRN_system(body = genotype)
    
    def mutate_body(
        self,
        rng: np.random.Generator,
    ) -> BodyGenotypeOrmV2GRN_system: # Ask whether a copy should be provided or not!
        
        # Make deepcopy of the genotype
        genotype = deepcopy(self.body)
        # Get a random mutation position
        position = rng.choice(range(0, len(genotype)), 1)[0]

        # Get a random mutation type
        type = rng.choice(['perturbation', 'deletion', 'addition', 'swap'], 1)[0]

        # Mutate the genotype
        if type == 'perturbation':
            # Add or subtract a random value from the genotype
            newv = round(genotype[position] + rng.normal(0, 0.1), 2)
            # Protect the boundaries
            if newv > 1:
                genotype[position] = 1
            elif newv < 0:
                genotype[position] = 0
            else:
                genotype[position] = newv
        elif type == 'deletion':
            # Delete a value from the genotype
            genotype.pop(position)
        elif type == 'addition':
            # Add a value to the genotype
            genotype.insert(position, round(rng.uniform(0, 1), 2))
        elif type == 'swap':
            # Sample random second position
            position2 = rng.choice(range(0, len(genotype)), 1)[0]
            while position == position2:
                position2 = rng.choice(range(0, len(genotype)), 1)[0]
            # Get values
            position_v = genotype[position]
            position2_v = genotype[position2]
            # Swap values
            genotype[position] = position2_v
            genotype[position2] = position_v
        else:
            raise ValueError(f'Unknown mutation type {type}')
        
        return BodyGenotypeOrmV2GRN_system(body = genotype)

    @classmethod
    def crossover_body(
        cls,
        parent1: Self,
        parent2: Self,
        rng: np.random.Generator,
    ) -> BodyGenotypeOrmV2GRN_system:
        # Get genotypes
        genotype1 = parent1.body
        genotype2 = parent2.body

        # Set promoter threshold and number of nucleotypes 
        promoter_threshold = 0.8
        types_nucleotypes = 6

        # The first nucleotide is the concentration --> average of the parents
        new_genotype = []
        for ig in range(0, 7):
            new_genotype.append((genotype1[ig] + genotype2[ig])/2)

        # Get remaining nucleotides from parents
        p1 = genotype1[7:]
        p2 = genotype2[7:]

        # Get new genotype
        for parent in [p1, p2]:
            # Initialize nucleotide index and promotor sites
            nucleotide_idx = 0
            promotor_sites = []
            while nucleotide_idx < len(parent):
                # If the nucleotide value is less than the promoter threshold
                if parent[nucleotide_idx] < promoter_threshold:
                    # If there are nucleotides enough to compose a gene
                    if (len(parent) - 1 - nucleotide_idx) >= types_nucleotypes:
                        promotor_sites.append(nucleotide_idx)
                        nucleotide_idx += types_nucleotypes
                nucleotide_idx += 1

            # Sample a promotor site
            cutpoint = rng.choice(promotor_sites, 1)[0]
            # Get a subset of the parent genotype
            subset = parent[0:cutpoint+types_nucleotypes+1]
            # Append the subset to the new genotype
            new_genotype += subset

        return BodyGenotypeOrmV2GRN_system(body = new_genotype)
    
    def develop_body(self: object, max_parts: int, mode_core_mult: bool) -> BodyV2:
        """
        Goal:
            Develop the genotype into a modular robot.
        -------------------------------------------------------------------------------------------
        Output:
            The created robot.
        """
        return DevelopGRN(max_parts, mode_core_mult, self.body).develop()




@event.listens_for(BodyGenotypeOrmV2GRN_system, "before_update", propagate=True)
@event.listens_for(BodyGenotypeOrmV2GRN_system, "before_insert", propagate=True)
def _update_serialized_body(
    mapper: orm.Mapper[BodyGenotypeOrmV2GRN_system],
    connection: Connection,
    target: BodyGenotypeOrmV2GRN_system,
) -> None:
    target._serialized_body = ','.join([str(gen) for gen in target.body])
    pass

@event.listens_for(BodyGenotypeOrmV2GRN_system, "load", propagate=True)
def _deserialize_body(target: BodyGenotypeOrmV2GRN_system, context: orm.QueryContext) -> None:
    body = [float(gen) for gen in target._serialized_body.split(",")]
    target.body = body