from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import os
import pandas as pd
from pyrr import Quaternion, Vector3

from revolve2.modular_robot.body import AttachmentPoint, Module
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2


def rotate(a: Vector3, b: Vector3, rotation: Quaternion) -> Vector3:
    """
    Rotates vector a a given angle around b.

    :param a: Vector a.
    :param b: Vector b.
    :param rotation: The quaternion for rotation.
    :returns: A copy of a, rotated.
    """
    cos_angle: int = int(round(np.cos(rotation.angle)))
    sin_angle: int = int(round(np.sin(rotation.angle)))

    vec: Vector3 = (
        a * cos_angle + sin_angle * b.cross(a) + (1 - cos_angle) * b.dot(a) * b
    )
    return vec


def vec3_int(vector: Vector3) -> Vector3[np.int_]:
    """
    Cast a Vector3 object to an integer only Vector3.

    :param vector: The vector.
    :return: The integer vector.
    """
    x, y, z = map(lambda v: int(round(v)), vector)
    return Vector3([x, y, z], dtype=np.int64)


@dataclass
class __Module:
    """"Goal:
        Class to hold some values for the functions in this file.
    ----------------------------------------------------------------------
    Input:
        position: The position of the module.
        forward: Identifies what the forward direction is for the module.
        up: Identifies what the up direction is for the module.
        chain_length: The distance (in blocks) from the core.
        module_type: The type of the module.
        rotation_index: The index of the rotation.
        _absolute_rotation: The absolute rotation index of the module.
        module_reference: The module."""
    position: Vector3[np.int_]
    forward: Vector3[np.int_]
    up: Vector3[np.int_]
    chain_length: int
    module_type: object
    rotation_index: int
    _absolute_rotation: int
    module_reference: Module


def add_child(
    module: __Module,
    attachment_point_tuple: tuple[int, AttachmentPoint],
    grid: NDArray[np.uint8], relllocslot: Vector3[np.int_], core_pos: Vector3[np.int_], 
    bool_core: str, id_string: dict
) -> __Module | None:
    """"Goal:
        Add a child to the body.
    ----------------------------------------------------------------------
    Input:
        module: The module
        
    ----------------------------------------------------------------------------------------------------
    Output:
        The new module or None if no child can be set."""
    # ---- Unpack attachment point tuple
    attachment_index, attachment_point = attachment_point_tuple

    # ---- Rotate vector a given angle around b, required due to change of orientation
    forward = rotate(module.forward, module.up, attachment_point.orientation)
    
    # ---- Calculate new position
    position = vec3_int(module.position + forward + relllocslot) 

    # Do a check for the position if we have a module directly following the core
    if bool_core:
        lwb_core, ub_core = core_pos - 1, core_pos + 1
        bool_pos = (position[0] >= lwb_core[0]) and (position[1] >= lwb_core[1]) and (position[2] >= lwb_core[2])
        bool_pos = bool_pos and (position[0] <= ub_core[0]) and (position[1] <= ub_core[1]) and (position[2] <= ub_core[2])
        assert not bool_pos, f"Error: The position is within the CORE, something went wrong! Position: {position}, Module Position: {module.position}, Module Forward: {module.forward}, Forward: {forward}, rellocslot: {relllocslot}, idx_slot: {attachment_index}."
    else:
        assert sum(np.array(relllocslot) == 0) == 3, f"Error: correction for grid is applied, but should not be performed, rellocslot: {relllocslot}, idx_slot: {attachment_index}."
    
    # ---- If grid cell is occupied, don't make a child else: set cell as occupied
    if grid[tuple(position)] > 0:
        return False, None, None, id_string # False means that the cell is occupied
    else:
        # Occupy cell
        # Increase chain length
        chain_length = module.chain_length + 1
        
    # ---- Evaluate CPPN
    child_type, child_rotation_index = np.random.choice([ActiveHingeV2, BrickV2]), np.random.choice([0, 1])
    
    # ---- Set grid cell as occupied
    if child_type is None:
        grid[tuple(position)] = 2
    elif child_type == ActiveHingeV2:
        grid[tuple(position)] = 3
    elif child_type == BrickV2:
        grid[tuple(position)] = 4

    # ---- Adapt rotation
    absolute_rotation = 0
    # Rotation always 0 for brick
    if (child_type != ActiveHingeV2):
        child_rotation_index = 0

    # Adapt absolute rotation  --> ?? idkn copied it
    if (child_type == ActiveHingeV2) and (child_rotation_index == 1):
        if (module.module_type == ActiveHingeV2) and (module._absolute_rotation == 1):
            absolute_rotation = 0
        else:
            absolute_rotation = 1
    else:
        if (module.module_type == ActiveHingeV2) and (module._absolute_rotation == 1):
            absolute_rotation = 1
    
    # Adapt rotation --> ?? idkn copied it
    if (child_type == BrickV2) and (module.module_type == ActiveHingeV2) and (module._absolute_rotation == 1):
        child_rotation_index = 1

    angle = child_rotation_index * (np.pi / 2.0) # Index is 0 | 1 --> 0 pi | 0.5 pi
    
    # ---- Is setting the child possible?
    bool_None = (child_type is None)
    if bool_None: # Empty block is queried --> No child can be set
        bool_set_child = False
    else:
        # Here we call the queried module type to get the desired type of module and rotation
        child = child_type(angle)
        if child_type == ActiveHingeV2:
            assert len(child.attachment_points.keys()) == 1, f"Error: The number of attachment points is not 1. Length: {len(child.attachment_points.keys())}."
        # Here we need an exception as for the core module the function is different!
        try:
            bool_set_child, slots2close = module.module_reference.can_set_child(child, attachment_index, flag = "condition")
            bool_set_child = (not bool_set_child)
        except TypeError:
            slots2close = []
            bool_set_child = (not module.module_reference.can_set_child(child, attachment_index))
    
    # If not possible, return None
    if bool_None or bool_set_child:
        return None, None, None, id_string
    
    # ---- Rotate the up vector around the forward vector
    up = rotate(module.up, forward, Quaternion.from_eulers([angle, 0, 0]))

    # ---- Set the new child
    module.module_reference.set_child(child, attachment_index)

    # ---- Store info in id_string
    # Set parent
    parent_position = module.position + relllocslot
    linear_index_parent = str(int(parent_position[0] * grid.shape[0] + parent_position[1]))

    if linear_index_parent in id_string.keys():
        if len(id_string[linear_index_parent]) == 2:
            id_string[linear_index_parent][1][str(attachment_index)] = str(child_rotation_index)
        else:
            id_string[linear_index_parent].append({str(attachment_index): str(child_rotation_index)})
    else:
        id_string[linear_index_parent] = ["C", {str(attachment_index): str(child_rotation_index)}]
    
    # Set child
    linear_index_child = str(int(position[0] * grid.shape[0] + position[1]))
    child_char = "A" if child_type == ActiveHingeV2 else "B"
    id_string[linear_index_child] = [child_char]
    
    return __Module(
        position,
        forward,
        up,
        chain_length,
        child_type,
        child_rotation_index,
        absolute_rotation,
        child,
    ), slots2close, child_type, id_string

def get_new_robot(max_parts):

    # Initialize
    part_count = 0
    nactivehinges, nbricks = 0, 0
    to_explore = [] # The modules which can be explored for adding more modules. Each item is a module's specific attachment face
    explored_modules = {} # The modules which have been explored already
    id_string = {}

    # Get Body
    body = BodyV2()

    # Get core
    v2_core = body.core_v2

    # Set core position
    max4grid = max(max_parts, 1)
    core_position = Vector3([max4grid + 2, max4grid + 2, 0], dtype = np.int_)

    # Increase body part count
    part_count += 1

    # Initialize Grid
    grid = np.zeros(shape=(max4grid * 2 + 4, max4grid * 2 + 4, 1), dtype=np.uint8)
    grid[max4grid + 1:max4grid + 4, max4grid + 1:max4grid + 4, 0] = 1
    assert np.sum(grid) == (3 * 3), f"Error: The core is not placed correctly in the grid. Sum: {np.sum(grid)}."

    # Add all attachment faces to the 'explore_list'
    for idx_attachment, attachment_face in v2_core.attachment_faces.items():
        if idx_attachment in [0, 2]:
            forward = Vector3([1, 0, 0])
        else:
            forward = Vector3([-1, 0, 0])
        to_explore.append(
            __Module(
                core_position,
                forward, # Here I changed the forward direction, because it seems not to be in line with the face coordinates
                Vector3([0, 0, 1]),
                0, "Core", 0, 0, attachment_face,))
        
    while part_count < max_parts:
        # Get parent module and id from "to_explore"
        module = np.random.choice(to_explore)
        module_id = module.module_reference.uuid

        # Get all attachment points of the module
        attachment_point_tuples_all = list(module.module_reference.attachment_points.items())
        
        # Initialize explored_modules if it is not initialized yet
        if module_id not in explored_modules.keys():
            explored_modules[module_id] = [[], []]
            if module.position == core_position: # Core module
                assert len(attachment_point_tuples_all) == 9, f"Error: The core module does not have 9 attachment points. Length: {len(attachment_point_tuples_all)}."
                # Eliminate attachment points?
                for att_tup in attachment_point_tuples_all:
                    if att_tup[0] in [0, 1, 2, 3, 5, 6, 7, 8]:
                        explored_modules[module_id][0].append(att_tup)
                
                ## Get the min and max values of the attachment points --> used to adapt the core position later on!
                # Offset of attachment points
                att_arr = [] 
                for att in attachment_point_tuples_all:
                    transf_off = rotate(att[1].offset, module.up, att[1].orientation)
                    att_arr.append(transf_off)
                att_arr = np.array(att_arr)
                # Min and max values of the attachment points
                explored_modules[module_id][1].append(att_arr.min(axis = 0))
                explored_modules[module_id][1].append(att_arr.max(axis = 0))

        # Get random attachment points which have not been explored yet
        attachment_point_tuples = [attach for attach in attachment_point_tuples_all if attach not in explored_modules[module_id][0]]
        attachment_point_tuple_idx = np.random.choice(np.arange(0, len(attachment_point_tuples)))
        attachment_point_tuple = attachment_point_tuples[attachment_point_tuple_idx]
        explored_modules[module_id][0].append(attachment_point_tuple) # Append to explored!

        # Check if forward direction is not vertical
        forward = rotate(module.forward, module.up, attachment_point_tuple[1].orientation)

        if forward[2] == 0: # Not vertical
            # Get slot location
            bool_core = (module.position == core_position)
            if bool_core: # If the module is a core module --> get relative location of slot as we have 3 x 3 x 3 core
                # Get relative location of slot within face
                middle = np.mean(explored_modules[module_id][1], axis = 0)
                divider = (explored_modules[module_id][1][1] - middle) # maximum slot location - middle, both are transformed already
                divider[divider == 0] = 1 # To prevent division by zero
                offset_pos = rotate(attachment_point_tuple[1].offset, module.up, attachment_point_tuple[1].orientation) # Transform offset
                rellocslot_raw = (offset_pos - middle)
                rellocslot = (rellocslot_raw / divider) # to -1, 0, 1
                # Add 1 additional for forward position --> 3 x 3 x 3 core instead of 1 x 1 x 1
                rellocslot = forward + rellocslot
            else:
                rellocslot = np.zeros(3, dtype = np.int64)

            # Add a child to the body
            child, slots2close, child_type, id_string = add_child(
                module, attachment_point_tuple, grid, rellocslot, core_position, bool_core, id_string)

            # Check some conditions
            if child == False: # Collision mode is off
                pass
            elif child is not None: # New module is not left as an empty cell
                to_explore.append(child)
                part_count += 1
                if child_type == ActiveHingeV2:
                    nactivehinges += 1
                elif child_type == BrickV2:
                    nbricks += 1

                # Remove closed slots
                for attachpointup in deepcopy(attachment_point_tuples):
                    if (attachpointup[0] in slots2close) and (attachpointup not in explored_modules[module_id][0]):
                        explored_modules[module_id][0].append(attachpointup)
                        attachment_point_tuples.remove(attachpointup)
        else:
            pass

        # Remove module from to_explore if it has no attachment points left after current
        if len(attachment_point_tuples) == 1:
            to_explore.remove(module)

        # Nothing left anymore or body collided with itself --> then stop the development
        if (to_explore == []):
            break


    # # ---- Plot
    # # Create a custom colormap with 4 colors
    # cmap = plt.cm.colors.ListedColormap(['grey', 'red', 'black', 'white', 'blue'])

    # # Create a normalized color map
    # norm = plt.cm.colors.Normalize(vmin=0, vmax=4)

    # # Create an array of colors based on the values
    # plt.imshow(grid[:, :, 0], cmap = cmap, norm = norm)
    # plt.xticks(np.arange(0, grid.shape[0], 1))
    # plt.yticks(np.arange(0, grid.shape[1], 1))
    # plt.grid(True, which='both')
    # plt.show()
        
    # ---- Complete id_string
    id_string = dict(sorted(id_string.items()))
    new_id_string = f"{max_parts}|"
    for key, value in id_string.items():
        new_id_string += f"{key}-{value[0]}"
        if len(value) > 1:
            for att, rot in dict(sorted(value[1].items())).items():
                new_id_string += att + rot
        new_id_string += "-"
        
    return body, new_id_string, part_count, nactivehinges, nbricks


bodies = []
grids, ids = {}, pd.DataFrame([], columns = ["id", "count"])
ngrids = [] # 1030 total
compute2spend = {1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10, 9: 10, 10: 10,
                 11: 10, 12: 10, 13: 10, 14: 10, 15: 10, 16: 10, 17: 10, 18: 10, 19: 10, 20: 10}
max_solutions = {1: np.inf, 2: np.inf, 3: np.inf, 4: np.inf, 5: np.inf, 6: np.inf, 7: np.inf, 8: np.inf, 9: np.inf, 10: np.inf,
                 11: np.inf, 12: np.inf, 13: np.inf, 14: np.inf, 15: np.inf, 16: np.inf, 17: np.inf, 18: np.inf, 19: np.inf, 20: np.inf}

for max_parts in range(1, 21):
    if ngrids != []:
        old_ngrids = np.cumsum(ngrids)[-1]
    else: old_ngrids = 0

    # Look if file exists for all unique ids
    if os.path.exists(f'C:\\Users\\niels\\OneDrive\\Documenten\\GitHub\\revolve2\\{max_parts}.json'):
        with open(f'C:\\Users\\niels\\OneDrive\\Documenten\\GitHub\\revolve2\\{max_parts}.json', 'r') as f:
            dict_grid = json.load(f)
            new_dict = {}
            for key, value in dict_grid.items():
                for nbricks, brickdict in value.items():
                    new_dict[int(nbricks)] = {}
                    for nactivehinges, gridlist in brickdict.items():
                        new_dict[int(nbricks)][int(nactivehinges)] = gridlist
            
                grids[max_parts] = new_dict
                ngrids += int(key) * [1]
            del value
    else:
        pass
    # Look if file exists for unique id counts
    if os.path.exists(f'C:\\Users\\niels\\OneDrive\\Documenten\\GitHub\\revolve2\\idcounts.csv'):
        ids = pd.read_csv(f'C:\\Users\\niels\\OneDrive\\Documenten\\GitHub\\revolve2\\idcounts.csv')
    else: pass


    # Get new robots
    for _ in range(compute2spend[max_parts] * 1):
        # Get new robot
        body, new_id_string, part_count, nactivehinges, nbricks = get_new_robot(max_parts = max_parts)

        #grid = grid.tolist()
        # Store id?
        if part_count not in grids.keys():
            grids[part_count] = {}
            grids[part_count][nbricks] = {}
            #grids[part_count][nbricks][nactivehinges] = [grid]
            grids[part_count][nbricks][nactivehinges] = [new_id_string]
            #bodies.append(body)
            ngrids.append(1)
        elif nbricks not in grids[part_count].keys():
            grids[part_count][nbricks] = {}
            #grids[part_count][nbricks][nactivehinges] = [grid]
            grids[part_count][nbricks][nactivehinges] = [new_id_string]
            #bodies.append(body)
            ngrids.append(1)
        elif nactivehinges not in grids[part_count][nbricks].keys():
            #grids[part_count][nbricks][nactivehinges] = [grid]
            grids[part_count][nbricks][nactivehinges] = [new_id_string]
            #bodies.append(body)
            ngrids.append(1)
        else:
            exists = False
            # for array in grids[part_count][nbricks][nactivehinges]:
            #     exists = np.array_equal(array, grid)
            #     if exists: break
            for string in grids[part_count][nbricks][nactivehinges]:
                exists = (string == new_id_string)
                if exists: break
            if not exists: 
                #grids[part_count][nbricks][nactivehinges].append(grid)
                grids[part_count][nbricks][nactivehinges].append(new_id_string)
                #bodies.append(body)
                ngrids.append(1)
            else:
                ngrids.append(0)
        # Increase id count
        if new_id_string not in ids["id"].values:
            ids = pd.concat([ids, pd.DataFrame([[new_id_string, 1]], columns = ["id", "count"])])
        else:
            ids.loc[ids["id"] == new_id_string, "count"] += 1

        # Stop early
        if np.cumsum(ngrids)[-1] >= max_solutions[max_parts]:
            break

    # ---- Dump
    # Ids
    new_grids = int(np.cumsum(ngrids)[-1] - old_ngrids)
    with open(f'{part_count}.json', 'w') as f:
        json.dump({new_grids: grids[part_count]}, f)
    # Id counts
    ids.to_csv(f'idcounts.csv', index = False)

    # ---- Print
    print("\tNumber of unique bodies: ", new_grids)

plt.plot(np.cumsum(ngrids))
plt.axis('square')
plt.show()
        
        





