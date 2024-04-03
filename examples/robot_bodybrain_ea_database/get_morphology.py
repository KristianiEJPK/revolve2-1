import config
import os
os.environ['ALGORITHM'] = config.ALGORITHM

if os.environ["ALGORITHM"] == "CPPN":
    from genotype import Genotype
elif os.environ["ALGORITHM"] == "GRN":
    from genotype_grn import Genotype
else:
    raise ValueError("ALGORITHM must be either GRN or CPPN")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment import Experiment
from generation import Generation
from individual import Individual
from population import Population
from sqlalchemy.orm import Session
from sqlalchemy import select

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging
from revolve2.ci_group.morphological_measures import MorphologicalMeasures

import uuid


def select_data(dbengine) -> pd.DataFrame:
    """Goal:
        Select the data of the column
    -------------------------------------------------------------
    Input:
        dbengine: ?
    --------------------------------------------------------------
    Output:
        df: pd.Dataframe"""
    
    with Session(dbengine) as ses:
        rows = ses.execute(
            select(Genotype, Experiment.id.label("experiment_id"), Generation.generation_index,
                   Individual.id.label("individual_index"))
            .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id)
            .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id),
        ).fetchall()

    return rows

# Setup logging
setup_logging()

# Initialize dataframe
df = pd.DataFrame([])

# Open database
dbengine = open_database_sqlite(config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS)

# Get pandas data
rows = select_data(dbengine)
rows = rows[40000:] #45000:46000
for irow, row in enumerate(rows):
    # Print progress
    print(f"Processing row {irow + 1} of {len(rows)}", end="\r")
    # Get body
    genotype = row[0]
    experiment_id = row[1]
    generation_index = row[2]
    individual_index = row[3]

    # Develop body
    if os.environ["ALGORITHM"] == "CPPN":
        modular_robot = genotype.develop(zdirection = config.ZDIRECTION, include_bias = config.CPPNBIAS,
                include_chain_length = config.CPPNCHAINLENGTH, include_empty = config.CPPNEMPTY,
                max_parts = config.MAX_PARTS, mode_collision = config.MODE_COLLISION,
                mode_core_mult = config.MODE_CORE_MULT, mode_slots4face = config.MODE_SLOTS4FACE,
                mode_slots4face_all = config.MODE_SLOTS4FACE_ALL, mode_not_vertical = config.MODE_NOT_VERTICAL)
    elif os.environ["ALGORITHM"] == "GRN":
        modular_robot = genotype.develop(include_bias = config.CPPNBIAS, max_parts = config.MAX_PARTS, mode_core_mult = config.MODE_CORE_MULT)
    else:
        raise ValueError("ALGORITHM must be either GRN or CPPN")

    # ---- Get morphological measures
    morphology = MorphologicalMeasures(body = modular_robot.body, brain = np.nan, max_modules = config.MAX_PARTS)
    
    id_string = morphology.id_string

    nbricks = morphology.num_bricks
    nhinges = morphology.num_active_hinges

    size = morphology.size
    proportion2d = morphology.proportion_2d
    proportionNiels = morphology.proportionNiels

    single_neighbour_brick_ratio = morphology.single_neighbour_brick_ratio
    single_neighbour_ratio = morphology.single_neighbour_ratio
    double_neighbour_brick_and_active_hinge_ratio = morphology.double_neigbour_brick_and_active_hinge_ratio
    maxrel_llimbs, meanrel_llimbs, stdrel_llimbs, nlimbs = morphology.length_of_limbsNiels
    
    joints = morphology.joints
    joint_brick_ratio = morphology.joint_brick_ratio

    symmetry_incl, symmetry_excl = morphology.symmetry
    sym1, sym2, sym3, sym4 = symmetry_incl
    syme1, syme2, syme3, syme4 = symmetry_excl

    coverage = morphology.coverage
    branching = morphology.branching
    surface_area = morphology.surface

    # ---- Add to dataframe
    df = pd.concat([df, pd.DataFrame([{"bricks": nbricks, "hinges": nhinges, "modules": nbricks + nhinges + 1, "size": size, 
                                        
                                        "proportion2d": proportion2d, "proportionNiels": proportionNiels,

                                    "single_neighbour_brick_ratio": single_neighbour_brick_ratio, 
                                    "single_neighbour_ratio": single_neighbour_ratio, "double_neighbour_brick_and_active_hinge_ratio": double_neighbour_brick_and_active_hinge_ratio, 
                                    "maxrel_llimbs": maxrel_llimbs, "meanrel_llimbs": meanrel_llimbs, "stdrel_llimbs": stdrel_llimbs,
                                    "nlimbs": nlimbs,
                                    
                                    "joints": joints, "joint_brick_ratio": joint_brick_ratio, 
                                    
                                    "symmetry_incl1": sym1, "symmetry_incl2": sym2, "symmetry_incl3": sym3, "symmetry_incl4": sym4,
                                    "symmetry_excl1": syme1, "symmetry_excl2": syme2, "symmetry_excl3": syme3, "symmetry_excl4": syme4,
                            
                                    "coverage": coverage, "branching": branching, "surface_area": surface_area,
                                    "id_string": id_string, "experiment_id": experiment_id, "generation_index": generation_index, "individual_index": individual_index}])],
                                        ignore_index=True)



# Create directory
path = f"C:\\Users\\niels\\OneDrive\\Documenten\\GitHub\\revolve2\\Test\\{os.environ["ALGORITHM"]}\\Morphologies"
if not os.path.exists(path):
    os.makedirs(path)
uuid = uuid.uuid4()
df.to_csv(path + f"\\morphological_measures_experiment_{uuid}.csv", index = False)
# # # Get max and mean fitness per experiment per generation
# # agg_per_experiment_per_generation = (
# #     df.groupby(["experiment_id", "generation_index"])
# #     .agg({column: ["max", "mean"]})
# #     .reset_index()
# # )

# # # Aggregate over experiments
# # agg_per_experiment_per_generation.columns = [
# #     "experiment_id",
# #     "generation_index",
# #     f"max_{column}",
# #     f"mean_{column}",
# # ]