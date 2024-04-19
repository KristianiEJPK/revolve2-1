import json
import numpy as np
import os
import pandas as pd
from pyrr import Quaternion, Vector3

from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2
from revolve2.ci_group.morphological_measures import MorphologicalMeasures


def rotate(a: Vector3, b: Vector3, rotation: Quaternion) -> Vector3:
    """
    Goal:
        Rotates vector a a given angle around b.
    --------------------------------------------------------
    Input:
        a:
            Vector a.
        b:
            Vector b.
        rotation:
            The quaternion for rotation.
    --------------------------------------------------------
    Output:
        A copy of a, rotated.
    """
    cos_angle: int = int(round(np.cos(rotation.angle)))
    sin_angle: int = int(round(np.sin(rotation.angle)))

    vec: Vector3 = (
        a * cos_angle + sin_angle * b.cross(a) + (1 - cos_angle) * b.dot(a) * b
    )
    return vec


def vec3_int(vector: Vector3) -> Vector3[np.int_]:
    """
    Goal:
        Cast a Vector3 object to an integer only Vector3.
    --------------------------------------------------------
    Input:
        vector:
            The vector.
    --------------------------------------------------------
    Output:
        The integer vector.
    """
    x, y, z = map(lambda v: int(round(v)), vector)
    return Vector3([x, y, z], dtype=np.int64)

def pos2lin(position: Vector3[int], shape0: int) -> int:
    """Goal:
        Position to linear index.
    --------------------------------------------------------
    Input:
        position:
            The position of the module.
        shape0:
            The number of modules in the first dimension.
    --------------------------------------------------------
    Output:
       The linear index of the module."""
    return int(position[0] * shape0 + position[1])

def get_body(max_parts: int, dict_coord: dict):
    """Goal:
        Create a body from a given dictionary.
    ------------------------------------------------------------------
    Input:
        max_parts: int
            The maximum number of parts.
        dict_coord: dict
            The dictionary that contains the building plan.
            {poslin: ["B" or "H", {attachment_point: rotation_index}]}
    ------------------------------------------------------------------
    Output:
        BodyV2
            The body that is created.
    """
    # ---- Initialize body
    body = BodyV2()
    v2_core = body.core_v2

    # ---- Return core if building plan is empty
    if dict_coord == None:
        return body
    
    # ---- Get the maximum number of parts and the core position in the grid
    max4grid = max(max_parts, 1)
    position_core = Vector3([max4grid + 2, max4grid + 2, 0], dtype = np.int_)

    # ---- Loop through all attachment faces
    for idx_attachment, attachment_face in v2_core.attachment_faces.items():
        # -- Initialize
        # Get forward --> No idea why but this works properly!
        if idx_attachment in [0, 2]:
            forward = Vector3([1, 0, 0])
        else:
            forward = Vector3([-1, 0, 0])
        # Get upward
        up = Vector3([0, 0, 1])
        
        # -- Get all attachment points
        attachment_point_tuples_all = list(attachment_face.attachment_points.items())
        ## Get the min and max values of the attachment points --> used to adapt the core position later on!
        # Offset of attachment points
        att_arr = [] 
        for att in attachment_point_tuples_all:
            transf_off = rotate(att[1].offset,up, att[1].orientation)
            att_arr.append(transf_off)
        att_arr = np.array(att_arr)
        # Min and max values of the attachment points
        min_values = att_arr.min(axis = 0)
        max_values = att_arr.max(axis = 0)

        # -- Get fourth attachment point --> only middle now
        for attup in attachment_point_tuples_all:
            if attup[0] == 4:
                break
        
        # -- Get relative location of slot within face
        # Rotate forward vector 
        forward4slot = rotate(forward, up, attup[1].orientation)
        # Get middle and divider
        middle = np.mean([min_values, max_values], axis = 0)
        divider = (max_values - middle) # maximum slot location - middle, both are transformed already
        divider[divider == 0] = 1 # To prevent division by zero
        # Transform offset
        offset_pos = rotate(attup[1].offset, up, attup[1].orientation) # Transform offset
        # Get relative location of slot within face
        rellocslot_raw = (offset_pos - middle)
        rellocslot = (rellocslot_raw / divider) # to -1, 0, 1
        # Add 1 additional for forward position --> 3 x 3 x 3 core instead of 1 x 1 x 1
        rellocslot = forward4slot + rellocslot

        # -- Get the linear index of the current position
        poslin_current = pos2lin(position_core + rellocslot, max4grid * 2 + 4)
        # Skip if the position is not in the dictionary
        if poslin_current not in dict_coord.keys():
            continue

        # -- Initialize modules
        modules = {poslin_current: {"module": attachment_face, "position": position_core, "forward": forward, "up": up}}
        tuples2cons = [att for att in attachment_point_tuples_all if att[0] in list(dict_coord[poslin_current][1].keys())]
        modules[poslin_current]["tuples"] = tuples2cons
        current_module = modules[poslin_current]
        
        # -- Build the body from current face
        while len(modules) != 0:
            # ---- Get attachment tuple
            attup = current_module["tuples"][0]

            # ---- Rotate vector a given angle around b, required due to change of orientation
            forward = rotate(current_module["forward"], current_module["up"], attup[1].orientation)

            # ---- Calculate new position
            position = vec3_int(current_module["position"] + forward + rellocslot) 
            # Get linear index
            poslin = pos2lin(position, max4grid * 2 + 4)

            # Get child type and rotation index
            child_type = BrickV2 if dict_coord[poslin][0] == "B" else ActiveHingeV2
            child_rotation_index = dict_coord[poslin_current][1][attup[0]]

            # ---- Set the new child
            angle = child_rotation_index * (np.pi / 2.0) # Index is 0 | 1 --> 0 pi | 0.5 pi
            child = child_type(angle)
            current_module["module"].set_child(child, attup[0])

            # ---- Rotate the up vector around the forward vector
            up = rotate(current_module["up"], forward, Quaternion.from_eulers([angle, 0, 0]))

            # ---- Store foreward and up vector
            tuples2cons = [att for att in list(child.attachment_points.items())
                        if att[0] in list(dict_coord[poslin][1].keys())]
            # If there are occupied attachment points, store the module
            if tuples2cons != []:
                modules[poslin] = {"module": child, "position": position, 
                                   "forward": forward, "up": up, "tuples": tuples2cons}
            else:
                del dict_coord[poslin]

            # ---- Delete the adressed attachment point
            current_module["tuples"] = current_module["tuples"][1:]

            # ---- Change module if all attachment points have been adressed
            if (len(current_module["tuples"]) == 0) and (len(modules) != 1):
                del modules[poslin_current]
                module_list = list(modules.values())
                current_module = module_list[np.random.choice(np.arange(len(module_list)))]
                poslin_current = pos2lin(current_module["position"], max4grid * 2 + 4)
                rellocslot = np.zeros(3, dtype = np.int32)
            elif len(current_module["tuples"]) == 0:
                break
            else: pass

    return body

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