"""Individual class."""

from dataclasses import dataclass

from base import Base
import os

if os.environ["Algorithm"] == "CPPN":
    from genotype import Genotype
elif os.environ["Algorithm"] == "GRN":
    from genotype_grn import Genotype

from revolve2.experimentation.optimization.ea import Individual as GenericIndividual


@dataclass
class Individual(
    Base, GenericIndividual[Genotype], population_table="population", kw_only=True
):
    """An individual in a population."""

    __tablename__ = "individual"
