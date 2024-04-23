import json
import numpy as np
import os
import pandas as pd
from revolve2.ci_group.morphological_measures import MorphologicalMeasures
from develop_from_string import get_body

def main_morph(path2file, max_parts, destination_path):
    print(f"Processing {max_parts} parts")
    # Initialize dataframe and strings
    if not os.path.exists(destination_path):
        df = pd.DataFrame([])
        strings = []
    else:
        df = pd.read_csv(destination_path)
        strings = df["id_string"].values.tolist()

    # Get counts
    with open(path2file.split(".")[0] + "_counts.json", 'r') as f:
        dict_counts = json.load(f)
        # Get new dict
        new_dict = {}
        for key, value in dict_counts.items():
            for nbricks, brickdict in value.items():
                new_dict[int(nbricks)] = {}
                for nactivehinges, gridlist in brickdict.items():
                    new_dict[int(nbricks)][int(nactivehinges)] = gridlist
        # Initialize counts
        counts = new_dict
    
    # Initialize samples and counter
    samples = []
    counter = 0

    # Open file for max_parts
    with open(path2file, 'r') as f:
        # Load data
        data = json.load(f)

        # Bodies are stored in a certain way, so we need to loop through the data
        # ---> max_parts: {nbricks: {nhinges: [string]}}
        for max_parts, dictparts in data.items():
            for bricks, dictbricks in dictparts.items():
                for hinges, dicthinges in dictbricks.items():
                    for string in dicthinges:
                        counter += 1
                        if counter % 1000 == 0:
                            print(f"{max_parts}: Processed {counter} strings")
                        # New string?
                        if df.empty or (string not in strings):
                            # Get max_parts
                            splitted = string.split("|")
                            max_parts = int(splitted[0])
                            # Only core? or should we get the building plan?
                            if max_parts == 1:
                                dict_coord = None
                            elif len(splitted) == 2:
                                # ---- Get coordinate data
                                substring = splitted[1]
                                substring_split = substring.split("-")
                                # Fill dictionary with building plan
                                # --> {poslin: ["B" or "H", {attachment_point: rotation_index}]}
                                dict_coord = {}
                                i = 0
                                while (i != len(substring_split)) and (substring_split[i] != ""):
                                    # Linear coordinate
                                    coord = int(substring_split[i])
                                    # Information for that coordinate (type, attachment points and orientations)
                                    info = substring_split[i + 1]
                                    # Set type of module (Brick or Hinge)
                                    dict_coord[coord] = []
                                    dict_coord[coord].append(info[0])
                                    # Set attachment points and orientations
                                    if len(info[1:]) > 1:
                                        dict_coord[coord].append({})
                                        for j in range(int((len(info) - 1) / 2)):
                                            dict_coord[coord][1][int(info[1 + int(j * 2)])] = int(info[1 + (int(j * 2) + 1)])
                                    else:
                                        dict_coord[coord].append({})
                                    # Increase i
                                    i += 2
                            else:
                                raise ValueError("No substring found")
                            
                            # ---- Develop body
                            body = get_body(max_parts, dict_coord)
                            
                            # ---- Get morphological measures
                            morphology = MorphologicalMeasures(body = body, brain = np.nan, max_modules = 30)
                            
                            id_string = morphology.id_string
                            assert id_string == string, "ID string not equal to original string!"

                            nbricks = morphology.num_bricks
                            nhinges = morphology.num_active_hinges
                            assert (nhinges + nbricks + 1) == max_parts, "Number of parts not equal to max_parts!"

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
                            strings.append(id_string)
                            samples.append(pd.DataFrame([{"bricks": nbricks, "hinges": nhinges, "modules": max_parts, "size": size, 
                                                               
                                                               "proportion2d": proportion2d, "proportionNiels": proportionNiels,

                                                            "single_neighbour_brick_ratio": single_neighbour_brick_ratio, 
                                                            "single_neighbour_ratio": single_neighbour_ratio, "double_neighbour_brick_and_active_hinge_ratio": double_neighbour_brick_and_active_hinge_ratio, 
                                                            "maxrel_llimbs": maxrel_llimbs, "meanrel_llimbs": meanrel_llimbs, "stdrel_llimbs": stdrel_llimbs,
                                                            "nlimbs": nlimbs,
                                                            
                                                            "joints": joints, "joint_brick_ratio": joint_brick_ratio, 
                                                            
                                                            "symmetry_incl1": sym1, "symmetry_incl2": sym2, "symmetry_incl3": sym3, "symmetry_incl4": sym4,
                                                            "symmetry_excl1": syme1, "symmetry_excl2": syme2, "symmetry_excl3": syme3, "symmetry_excl4": syme4,
                                                    
                                                            "coverage": coverage, "branching": branching, "surface_area": surface_area,
                                                            "id_string": id_string, "count": counts[int(bricks)][int(hinges)][id_string]}]))
                

                        else:
                            # Increase count
                            old_count = df.loc[df["id_string"] == string, "count"].values[0]
                            idx = df.loc[df["id_string"] == string].index[0]
                            df.loc[idx, "count"] = counts[int(bricks)][int(hinges)][string]
                            assert old_count >= df.loc[idx, "count"], "Count not increased!"
    
    # Concatenate data
    if samples != []:
        df = pd.concat(samples + [df], ignore_index = True)
    else:
        pass
    
    # Write to file
    df.to_csv(destination_path, index = False)

    print(f"Finished with {max_parts} parts")
    return df