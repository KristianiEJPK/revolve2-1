from copy import deepcopy
import multineat
import numpy as np
import sqlalchemy.orm as orm
from sqlalchemy import event
from sqlalchemy.engine import Connection

from typing_extensions import Self

class BodyGenotypeOrmV2(orm.MappedAsDataclass, kw_only=True):
    """Goal:
        SQLAlchemy model for a CPPNWIN body genotype."""

    # Body genotype
    body: multineat.Genome

    # Serialized body
    _serialized_body: orm.Mapped[str] = orm.mapped_column(
        "serialized_body", init=False, nullable=False
    )

    @classmethod
    def random_body(self: object, rng: np.random.Generator) -> BodyGenotypeOrmV2:
        # Set genome size
        genome_size = 150 + 1
        # Set random genotype
        genotype = [round(rng.uniform(0, 1), 2) for _ in range(genome_size)]

        return BodyGenotypeOrmV2(body = genotype)
    
    def mutate_body(
        self,
        innov_db: multineat.InnovationDatabase,
        rng: np.random.Generator,
    ) -> BodyGenotypeOrmV2: # Ask whether a copy should be provided or not!
        
        # Make deepcopy of the genotype
        genotype = deepcopy(innov_db.body)
        # Get a random mutation position
        position = rng.sample(range(0, len(genotype)), 1)[0]

        # Get a random mutation type
        type = rng.sample(['perturbation', 'deletion', 'addition', 'swap'], 1)[0]

        # Mutate the genotype
        if type == 'perturbation':
            # Add or subtract a random value from the genotype
            newv = round(genotype[position] + rng.normalvariate(0, 0.1), 2)
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
            position2 = rng.sample(range(0, len(genotype)), 1)[0]
            while position == position2:
                position2 = rng.sample(range(0, len(genotype)), 1)[0]
            # Get values
            position_v = genotype.genotype[position]
            position2_v = genotype.genotype[position2]
            # Swap values
            genotype.genotype[position] = position2_v
            genotype.genotype[position2] = position_v
        else:
            raise ValueError(f'Unknown mutation type {type}')
        
        return BodyGenotypeOrmV2(body = genotype)

    @classmethod
    def crossover_body(
        cls,
        parent1: Self,
        parent2: Self,
        rng: np.random.Generator,
    ) -> BodyGenotypeOrmV2:

        # Set promoter threshold and number of nucleotypes 
        promoter_threshold = 0.8
        types_nucleotypes = 6

        # The first nucleotide is the concentration --> average of the parents
        new_genotype = [(parent1.genotype[0] + parent2.genotype[0])/2]

        # Get remaining nucleotides from parents
        p1 = parent1.genotype[1:]
        p2 = parent2.genotype[1:]

        # Get new genotype
        for parent in [p1, p2]:
            # Initialize nucleotide index and promotor sites
            nucleotide_idx = 0
            promotor_sites = []
            while nucleotide_idx < len(parent):
                # If the nucleotide value is less than the promoter threshold
                if parent[nucleotide_idx] < promoter_threshold:
                    # If there are nucleotides enough to compose a gene
                    if (len(parent) - 1 - nucleotide_idx) >= types_nucleotypes: # ???? (e.g. 6 - 1 - 0) = 5 != 6
                        promotor_sites.append(nucleotide_idx)
                        nucleotide_idx += types_nucleotypes
                nucleotide_idx += 1 # ???? Is there also a nucleotide skipped in between?

            # Sample a promotor site
            cutpoint = rng.sample(promotor_sites, 1)[0]
            # Get a subset of the parent genotype
            subset = parent[0:cutpoint+types_nucleotypes+1]
            # Append the subset to the new genotype
            new_genotype += subset

        return BodyGenotypeOrmV2(body = new_genotype)