from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import multineat
import numpy as np
from numpy.typing import NDArray
from pyrr import Quaternion, Vector3

import random
from revolve2.modular_robot.body import AttachmentPoint, Module
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2


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
    
def develop(
    genotype: multineat.Genome, querying_seed: int, zdirection: bool, 
        include_bias: bool, include_chain_length: bool, include_empty: bool,
        max_parts: int, mode_collision: bool, mode_core_mult: bool, 
        mode_slots4face: bool, mode_slots4face_all: bool, mode_not_vertical: bool
) -> BodyV2:
    """
    Goal:
        Develop a CPPNWIN genotype into a modular robot body. It is important that the genotype was created using 
        a compatible function.
    -------------------------------------------------------------------------------------------------------------
    Input:
        genotype: The genotype to create the body from.
        querying_seed: The seed for the random number generator.
        zdirection: Whether to include the z direction as input for CPPN.
        include_bias: Whether to include the bias as input for CPPN.
        include_chain_length: Whether to include the chain length as input for CPPN.
        include_empty: Whether to include the empty module output for CPPN.
        max_parts: Maximum number of parts in the body.
        mode_collision: Whether to stop if collision occurs.
        mode_core_mult: Whether to allow multiple core slots.
        mode_slots4face: Whether multiple slots can be used for a single face for the core module.
        mode_slots4face_all: Whether slots can be set for all 9 attachments, or only 3, 4, 5.
        mode_not_vertical: Whether to disable vertical expansion of the body.
    -------------------------------------------------------------------------------------------------------------
    Output:
        The createD body."""
    # ---- Initialize
    rng = random.Random(querying_seed) # Random number generator
    collision = False # If the body has collided with itself
    part_count = 0 # Number of body parts
    to_explore = [] # The modules which can be explored for adding more modules. Each item is a module's specific attachment face
    explored_modules = {} # The modules which have been explored already
    
    # ---- CPPN
    # Initialize cppn
    body_net = multineat.NeuralNetwork()

    # Construct phenotype from genotype
    genotype.BuildPhenotype(body_net)

    # ---- Body
    # Initialize body
    body = BodyV2() # Get body, which only contains the core right now
    v2_core = body.core_v2
    # Set core position
    if mode_core_mult:
        # 3 x 3 x 3 core block
        core_position = Vector3([max_parts + 2, max_parts + 2, max_parts + 2], dtype = np.int_)
    else:
        core_position = Vector3([max_parts + 1, max_parts + 1, max_parts + 1], dtype = np.int_)
    # Increase body part count
    part_count += 1

    # Initialize grid
    if mode_core_mult:
        # 3 x 3 x 3 block for core instead of 1 x 1 x 1
        grid = np.zeros(shape=(max_parts * 2 + 4, max_parts * 2 + 4, max_parts * 2 + 4), dtype=np.uint8)
        grid[max_parts + 1:max_parts + 4, max_parts + 1:max_parts + 4, max_parts + 1:max_parts + 4] = 1
        assert np.sum(grid) == (3**3), f"Error: The core is not placed correctly in the grid. Sum: {np.sum(grid)}."
    else:
        grid = np.zeros(shape=(max_parts * 2 + 1, max_parts * 2 + 1, max_parts * 2 + 1), dtype=np.uint8)
        grid[max_parts + 1, max_parts + 1, max_parts + 1] = 1
        assert np.sum(grid) == 1, f"Error: The core is not placed correctly in the grid. Sum: {np.sum(grid)}."

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

    # ---- Explore all attachment points for development --> recursive
    for _ in range(0, (max_parts - 1)):
        # Get parent module and id from "to_explore"
        module = rng.choice(to_explore)
        module_id = module.module_reference.uuid

        # Get all attachment points of the module
        attachment_point_tuples_all = list(module.module_reference.attachment_points.items())

        # Initialize explored_modules if it is not initialized yet
        if module_id not in explored_modules.keys():
            explored_modules[module_id] = [[], []]
            if module.position == core_position: # Core module
                assert len(attachment_point_tuples_all) == 9, f"Error: The core module does not have 9 attachment points. Length: {len(attachment_point_tuples_all)}."
                # Eliminate attachment points?
                if (mode_core_mult == False) or (mode_slots4face == False): # Keep only middle attachment point
                    for att_tup in attachment_point_tuples_all:
                        if att_tup[0] in [0, 1, 2, 3, 5, 6, 7, 8]:
                            explored_modules[module_id][0].append(att_tup)
                elif (mode_slots4face_all == False): # Keep only middle (row's) attachment points
                    for att_tup in attachment_point_tuples_all:
                        if att_tup[0] in [0, 1, 2, 6, 7, 8]:
                            explored_modules[module_id][0].append(att_tup)
                else:
                    assert mode_slots4face, "Error: The mode for slots4face is not set correctly."
                # Get the min and max values of the attachment points --> used to adapt the core position later on!
                if mode_core_mult:
                    # Offset of attachment points
                    att_arr = [] 
                    for att in attachment_point_tuples_all:
                        transf_off = __rotate(att[1].offset, module.up, att[1].orientation)
                        att_arr.append(transf_off)
                    att_arr = np.array(att_arr)
                    # Min and max values of the attachment points
                    explored_modules[module_id][1].append(att_arr.min(axis = 0))
                    explored_modules[module_id][1].append(att_arr.max(axis = 0))

        # Get random attachment points which have not been explored yet
        attachment_point_tuples = [attach for attach in attachment_point_tuples_all if attach not in explored_modules[module_id][0]]
        attachment_point_tuple = tuple(rng.choice(attachment_point_tuples))
        explored_modules[module_id][0].append(attachment_point_tuple) # Append to explored!

        # Check if forward direction is not vertical
        forward = __rotate(module.forward, module.up, attachment_point_tuple[1].orientation)

        if (mode_not_vertical and (forward[2] == 0)) or (not mode_not_vertical):
            # Get slot location
            bool_core = (module.position == core_position)
            if bool_core and mode_core_mult: # If the module is a core module and multiple slots --> get relative location of slot as we have 3 x 3 x 3 core
                # Get relative location of slot within face
                middle = np.mean(explored_modules[module_id][1], axis = 0)
                divider = (explored_modules[module_id][1][1] - middle) # maximum slot location - middle, both are transformed already
                divider[divider == 0] = 1 # To prevent division by zero
                offset_pos = __rotate(attachment_point_tuple[1].offset, module.up, attachment_point_tuple[1].orientation) # Transform offset
                rellocslot_raw = (offset_pos - middle)
                rellocslot = (rellocslot_raw / divider) # to -1, 0, 1
                # Add 1 additional for forward position --> 3 x 3 x 3 core instead of 1 x 1 x 1
                rellocslot = forward + rellocslot
            else:
                rellocslot = np.zeros(3, dtype = np.int64)

            # Add a child to the body
            child, slots2close = __add_child(body_net, module, attachment_point_tuple, grid, rellocslot, core_position, mode_core_mult,
                                             mode_slots4face_all, bool_core, zdirection, include_bias, include_chain_length, include_empty)

            # Check some conditions
            if (child == False) and (mode_collision): # If the cell is occupied
                collision = True
            elif child == False: # Collision mode is off
                pass
            elif child is not None: # New module is not left as an empty cell
                to_explore.append(child)
                part_count += 1
                # Remove closed slots
                for attachpointup in deepcopy(attachment_point_tuples):
                    if (attachpointup[0] in slots2close) and (attachpointup not in explored_modules[module_id][0]):
                        explored_modules[module_id][0].append(attachpointup)
                        attachment_point_tuples.remove(attachpointup)
        else:
            pass

        # Remove module from to_explore if it has no attachment points left after current
        if not mode_slots4face: # Or immediately if mode_slots4face is off
            to_explore.remove(module)
        elif len(attachment_point_tuples) == 1:
            to_explore.remove(module)

        # Nothing left anymore or body collided with itself --> then stop the development
        if (to_explore == []) or (collision):
            break
    
    # # ---- Plot
    # # Create a custom colormap with 4 colors
    # cmap = plt.cm.colors.ListedColormap(['grey', 'red', 'black', 'white', 'blue'])

    # # Create a normalized color map
    # norm = plt.cm.colors.Normalize(vmin=0, vmax=4)

    # # Create an array of colors based on the values
    # plt.imshow(grid[:, :, core_position[2]], cmap = cmap, norm = norm)
    # plt.xticks(np.arange(0, grid.shape[0], 1))
    # plt.yticks(np.arange(0, grid.shape[1], 1))
    # plt.grid(True, which='both')
    # plt.show()
    
    return body


def __evaluate_cppn(
    body_net: multineat.NeuralNetwork,
    position: Vector3[np.int_],
    chain_length: int,
    zdirection: bool, include_bias: bool, include_chain_length: bool, include_empty: bool
) -> tuple[Any, int]:
    """
    Goal:
        Get module type and orientation from a multineat CPPN network.
    -----------------------------------------------------------------------
    Input:
        body_net: The CPPN network.
        position: Position of the module.
        chain_length: Tree distance of the module from the core.
        zdirection: Whether to include the z direction as input for CPPN.
        include_bias: Whether to include the bias as input for CPPN.
        include_chain_length: Whether to include the chain length as input for CPPN.
        include_empty: Whether to include the empty module output for CPPN.
    -----------------------------------------------------------------------
    Output:
        (module type, rotation_index)
    """
    # Unpack tuple
    x, y, z = position

    # Get inputs
    inputs = []
    if include_bias:
        inputs.append(1)
    inputs.append(x)
    inputs.append(y)
    if zdirection:
        inputs.append(z)
    if include_chain_length:
        inputs.append(chain_length)

    # Set inputs
    body_net.Input(inputs)

    # Activate all layers
    body_net.ActivateAllLayers()

    # Get outputs
    outputs = body_net.Output()

    # Get module type and rotation from output probabilities
    if include_empty:
        assert len(outputs) == 5, f"Error: The number of outputs is not 5. Length: {len(outputs)}."
        type_probs = list(outputs[:3])
        rotation_probs = list(outputs[3:5])
        types = [None, BrickV2, ActiveHingeV2]
    else:
        assert len(outputs) == 4, f"Error: The number of outputs is not 4. Length: {len(outputs)}."
        type_probs = list(outputs[:2])
        rotation_probs = list(outputs[2:4])
        types = [BrickV2, ActiveHingeV2]
    module_type = types[type_probs.index(max(type_probs))]
    rotation_index = rotation_probs.index(max(rotation_probs))

    return module_type, rotation_index


def __add_child(
    body_net: multineat.NeuralNetwork,
    module: __Module,
    attachment_point_tuple: tuple[int, AttachmentPoint],
    grid: NDArray[np.uint8], relllocslot: Vector3[np.int_], core_pos: Vector3[np.int_], 
    mode_core_mult: bool, mode_slots4face_all: bool, bool_core: str,
    zdirection: bool, include_bias: bool, include_chain_length: bool, include_empty: bool
) -> __Module | None:
    """"Goal:
        Add a child to the body.
    ----------------------------------------------------------------------
    Input:
        body_net: The CPPN network.
        module: The parent module.
        attachment_point_tuple: The attachment point tuple --> index and attachment point.
        grid: The grid of the body.
        relllocslot: The relative location of the slot --> only used if multiple slots are considered
        core_pos: The position of the core.
        mode_core_mult: Whether to allow multiple core slots.
        mode_slots4face_all: Whether slots can be set for all 9 attachments, or only 3, 4, 5
        bool_core: Whether the module is a core module.
        zdirection: Whether to include the z direction as input for CPPN.
        include_bias: Whether to include the bias as input for CPPN.
        include_chain_length: Whether to include the chain length as input for CPPN.
        include_empty: Whether to include the empty module output for CPPN.
    ----------------------------------------------------------------------------------------------------
    Output:
        The new module or None if no child can be set."""
    # ---- Unpack attachment point tuple
    attachment_index, attachment_point = attachment_point_tuple

    # ---- Rotate vector a given angle around b, required due to change of orientation
    forward = __rotate(module.forward, module.up, attachment_point.orientation)
    
    # ---- Calculate new position
    position = __vec3_int(module.position + forward + relllocslot) 

    # ---- Checks 
    # Do a check for z-position
    if (mode_slots4face_all == False) or (mode_core_mult == False):
        try:
            assert (position[2] == core_pos[2]), f"Error: The z-position is not the same as the core. Position: {position}, Module Position: {module.position}, Module Forward: {module.forward}, Forward: {forward}, rellocslot: {relllocslot}, idx_slot: {attachment_index}."
        except AssertionError as e:
            print(e)
            plt.imshow(grid[:, :, core_pos[2]])
            plt.xticks(np.arange(0, 20 * 2 + 3, 1))
            plt.yticks(np.arange(0, 20 * 2 + 3, 1))
            plt.grid(True, which='both')
            plt.show()
            raise e

    # Do a check for the position if we have a module directly following the core
    if bool_core:
        if mode_core_mult: # 3 x 3 x 3 core
            lwb_core, ub_core = core_pos - 1, core_pos + 1
        else: # 1 x 1 x 1 core
            lwb_core, ub_core = core_pos, core_pos
        bool_pos = (position[0] >= lwb_core[0]) and (position[1] >= lwb_core[1]) and (position[2] >= lwb_core[2])
        bool_pos = bool_pos and (position[0] <= ub_core[0]) and (position[1] <= ub_core[1]) and (position[2] <= ub_core[2])
        assert not bool_pos, f"Error: The position is within the CORE, something went wrong! Position: {position}, Module Position: {module.position}, Module Forward: {module.forward}, Forward: {forward}, rellocslot: {relllocslot}, idx_slot: {attachment_index}."
    else:
        assert sum(np.array(relllocslot) == 0) == 3, f"Error: correction for grid is applied, but should not be performed, rellocslot: {relllocslot}, idx_slot: {attachment_index}."
    
    # ---- If grid cell is occupied, don't make a child else: set cell as occupied
    if grid[tuple(position)] > 0:
        return False, None # False means that the cell is occupied
    else:
        # Occupy cell
        # Increase chain length
        chain_length = module.chain_length + 1
        
    # ---- Evaluate CPPN
    child_type, child_rotation_index = __evaluate_cppn(body_net, position, chain_length, zdirection, include_bias, include_chain_length, include_empty)
    
    # ---- Set grid cell as occupied
    if child_type is None:
        grid[tuple(position)] = 2
    elif child_type == ActiveHingeV2:
        grid[tuple(position)] = 3
    elif child_type == BrickV2:
        grid[tuple(position)] = 4

    # ---- Adapt rotation
    absolute_rotation = 0
    if zdirection == False:
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
        return None, None
    
    # ---- Rotate the up vector around the forward vector
    up = __rotate(module.up, forward, Quaternion.from_eulers([angle, 0, 0]))

    # ---- Set the new child
    module.module_reference.set_child(child, attachment_index)
    
    return __Module(
        position,
        forward,
        up,
        chain_length,
        child_type,
        child_rotation_index,
        absolute_rotation,
        child,
    ), slots2close  


def __rotate(a: Vector3, b: Vector3, rotation: Quaternion) -> Vector3:
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


def __vec3_int(vector: Vector3) -> Vector3[np.int_]:
    """
    Cast a Vector3 object to an integer only Vector3.

    :param vector: The vector.
    :return: The integer vector.
    """
    x, y, z = map(lambda v: int(round(v)), vector)
    return Vector3([x, y, z], dtype=np.int64)
