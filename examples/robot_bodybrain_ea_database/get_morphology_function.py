
import numpy as np
import os
from revolve2.ci_group.morphological_measures import MorphologicalMeasures

def get_morphologies(row, ZDIRECTION, CPPNBIAS, CPPNCHAINLENGTH, CPPNEMPTY, MAX_PARTS, MODE_COLLISION, MODE_CORE_MULT, MODE_SLOTS4FACE, 
                     MODE_SLOTS4FACE_ALL, MODE_NOT_VERTICAL):
    # Get body
    genotype = row[0]
    experiment_id = row[1]
    generation_index = row[2]
    individual_index = row[3]

    # Develop body
    if os.environ["ALGORITHM"] == "CPPN":
        modular_robot = genotype.develop(zdirection = ZDIRECTION, include_bias = CPPNBIAS,
                include_chain_length = CPPNCHAINLENGTH, include_empty = CPPNEMPTY,
                max_parts = MAX_PARTS, mode_collision = MODE_COLLISION,
                mode_core_mult = MODE_CORE_MULT, mode_slots4face = MODE_SLOTS4FACE,
                mode_slots4face_all = MODE_SLOTS4FACE_ALL, mode_not_vertical = MODE_NOT_VERTICAL)
    elif os.environ["ALGORITHM"] == "GRN":
        modular_robot = genotype.develop(include_bias = CPPNBIAS, max_parts = MAX_PARTS, mode_core_mult = MODE_CORE_MULT)
    else:
        raise ValueError("ALGORITHM must be either GRN or CPPN")

    # ---- Get morphological measures
    morphology = MorphologicalMeasures(body = modular_robot.body, brain = np.nan, max_modules = MAX_PARTS)
    
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
    dict = {"bricks": nbricks, "hinges": nhinges, "modules": nbricks + nhinges + 1, "size": size, 
                                        
                                        "proportion2d": proportion2d, "proportionNiels": proportionNiels,

                                    "single_neighbour_brick_ratio": single_neighbour_brick_ratio, 
                                    "single_neighbour_ratio": single_neighbour_ratio, "double_neighbour_brick_and_active_hinge_ratio": double_neighbour_brick_and_active_hinge_ratio, 
                                    "maxrel_llimbs": maxrel_llimbs, "meanrel_llimbs": meanrel_llimbs, "stdrel_llimbs": stdrel_llimbs,
                                    "nlimbs": nlimbs,
                                    
                                    "joints": joints, "joint_brick_ratio": joint_brick_ratio, 
                                    
                                    "symmetry_incl1": sym1, "symmetry_incl2": sym2, "symmetry_incl3": sym3, "symmetry_incl4": sym4,
                                    "symmetry_excl1": syme1, "symmetry_excl2": syme2, "symmetry_excl3": syme3, "symmetry_excl4": syme4,
                            
                                    "coverage": coverage, "branching": branching, "surface_area": surface_area,
                                    "id_string": id_string, "experiment_id": experiment_id, "generation_index": generation_index, "individual_index": individual_index}
    
    return dict