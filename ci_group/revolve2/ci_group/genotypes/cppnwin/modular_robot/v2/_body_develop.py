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
    """"position: The position of the module.
        forward: Identifies what the forward direction is for the module.
        up: Identifies what the up direction is for the module.
        chain_length: The distance (in blocks) from the core.
        module_reference: The module."""
    position: Vector3[np.int_]
    forward: Vector3[np.int_]
    up: Vector3[np.int_]
    chain_length: int
    module_reference: Module

def develop(
    genotype: multineat.Genome, querying_seed: int
) -> BodyV2:
    """
    Develop a CPPNWIN genotype into a modular robot body.

    It is important that the genotype was created using a compatible function.

    :param genotype: The genotype to create the body from.
    :returns: The create body.
    """
    # ---- Initialize
    # Modifiable parameters
    max_parts = 20 # Maximum number of parts in the body --> better pass as parameter????
    part_count = 0 # Number of body parts
    mode_collision = True # Whether to stop if collision occurs
    mode_slots4face = True # Whether multiple slots can be used for a single face for the core module
    mode_slots4face_all = False # Whether slots can be set for all 9 attachments, or only 3, 4, 5
    mode_not_vertical = True # Whether to disable vertical expansion of the body
    # Internal parameters
    rng = random.Random(querying_seed)
    collision = False # If the body has collided with itself
    to_explore = [] # The modules which can be explored for adding more modules. Each item is a module's specific attachment face
    
    # ---- CPPN
    # Initialize cppn
    body_net = multineat.NeuralNetwork()

    # Construct phenotype from genotype
    genotype.BuildPhenotype(body_net)

    # ---- Body
    # Initialize body
    body = BodyV2() # Get body, which only contains the core right now
    v2_core = body.core_v2
    # Set core position, which is in the middle of the 3 x 3 x 3 core block
    core_position = Vector3([max_parts + 2, max_parts + 2, max_parts + 2], dtype=np.int_)
    # Increase body part count
    part_count += 1

    # Initialize grid --> 3 x 3 x 3 block for core instead of 1 x 1 x 1
    grid = np.zeros(shape=(max_parts * 2 + 3, max_parts * 2 + 3, max_parts * 2 + 3), dtype=np.uint8)
    grid[max_parts + 1:max_parts + 4, max_parts + 1:max_parts + 4, max_parts + 1:max_parts + 4] = 1
    assert np.sum(grid) == (3**3), f"Error: The core is not placed correctly in the grid. Sum: {np.sum(grid)}."

    # Add all attachment faces to the 'explore_list' --> core position is kept the same as I do not know whether I can change it or not
    for attachment_face in v2_core.attachment_faces.values():
        to_explore.append(
            __Module(
                core_position,
                Vector3([1, 0, 0]),
                Vector3([0, 0, 1]),
                0, attachment_face,))

    # ---- Explore all attachment points for development --> recursive
    explored_modules = {}
    while part_count < max_parts:
        # Get parent module (and direction) from "to_explore" and remove it afterwards
        module = rng.choice(to_explore)
        module_id = module.module_reference.uuid

        # Get attachment points of the module
        attachment_point_tuples_all = list(module.module_reference.attachment_points.items())

        # Initialize explored_modules if it is not initialized yet
        if module_id not in explored_modules.keys():
            explored_modules[module_id] = [[], []]
            if module.position == core_position: # Core module
                assert len(attachment_point_tuples_all) == 9, f"Error: The core module does not have 9 attachment points. Length: {len(attachment_point_tuples_all)}."
                # Eliminate attachment points that are not on middle row?
                if mode_slots4face_all == False:
                    for att_tup in attachment_point_tuples_all:
                        if att_tup[0] in [0, 1, 2, 6, 7, 8]:
                            explored_modules[module_id][0].append(att_tup)
                ## Get the min and max values of the attachment points
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
        explored_modules[module_id][0].append(attachment_point_tuple)

        # Check if forward direction is not vertical
        forward = __rotate(module.forward, module.up, attachment_point_tuple[1].orientation)
        if (mode_not_vertical and (forward[2] == 0)) or (not mode_not_vertical):
            # Get slot location
            bool_core = (module.position == core_position)
            if bool_core: # If the module is a core module  --> get relative location of slot as we have 3 x 3 x 3 core
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
            child, slots2close = __add_child(body_net, module, attachment_point_tuple, grid, rellocslot, core_position, mode_slots4face_all, bool_core)

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
    
    # plt.imshow(grid[:, :, core_position[2]])
    # plt.xticks(np.arange(0, 20 * 2 + 3, 1))
    # plt.yticks(np.arange(0, 20 * 2 + 3, 1))
    # plt.grid(True, which='both')
    # plt.show()
    # print("----------------------------")

    return body


def __evaluate_cppn(
    body_net: multineat.NeuralNetwork,
    position: Vector3[np.int_],
    chain_length: int,
) -> tuple[Any, int]:
    """
    Get module type and orientation from a multineat CPPN network.

    :param body_net: The CPPN network.
    :param position: Position of the module.
    :param chain_length: Tree distance of the module from the core.
    :returns: (module type, rotation_index)
    """
    x, y, z = position
    # assert isinstance(
    #     x, np.int64
    # ), f"Error: The position is not of type int. Type: {type(x)}."
    body_net.Input([1.0, x, y, z, chain_length])  # 1.0 is the bias input
    body_net.ActivateAllLayers()
    outputs = body_net.Output()

    # get module type from output probabilities
    type_probs = list(outputs[:3])
    types = [None, BrickV2, ActiveHingeV2]
    module_type = types[type_probs.index(min(type_probs))]

    # get rotation from output probabilities
    rotation_probs = list(outputs[3:5])
    rotation_index = rotation_probs.index(min(rotation_probs))

    return module_type, rotation_index


def __add_child(
    body_net: multineat.NeuralNetwork,
    module: __Module,
    attachment_point_tuple: tuple[int, AttachmentPoint],
    grid: NDArray[np.uint8], relllocslot: Vector3[np.int_], core_pos: Vector3[np.int_], mode_slots4face_all: bool,
    bool_core: str
) -> __Module | None:
    
    # ---- Unpack attachment point tuple
    attachment_index, attachment_point = attachment_point_tuple

    # ---- Rotate vector a given angle around b, required due to change of orientation
    forward = __rotate(module.forward, module.up, attachment_point.orientation)
    
    # ---- Calculate new position
    position = __vec3_int(module.position + forward + relllocslot) 

    # ---- Checks 
    # Do a check for z-position
    if mode_slots4face_all == False:
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
        lwb_core, ub_core = core_pos - 1, core_pos + 1
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
        grid[tuple(position)] += 1
        # Increase chain length
        chain_length = module.chain_length + 1
    
    # plt.imshow(grid[:, :, core_pos[2]])
    # plt.xticks(np.arange(0, 20 * 2 + 3, 1))
    # plt.yticks(np.arange(0, 20 * 2 + 3, 1))
    # plt.grid(True, which='both')
    # plt.show()
    
    # ---- Get new position including the attachment point offset --> Why an int?
    if not bool_core:
        rotate_off = np.array(__rotate(attachment_point.offset, module.up, attachment_point.orientation))
        rotate_off = Vector3(np.where(rotate_off != 0, np.sign(rotate_off) * 0.5, 0))
    else: # For core we already corrected the position completely by switching from a 3 x 3 x 3 core to a 1 x 1 x 1 core
        rotate_off = 0
    new_pos = np.array(position + rotate_off, dtype=np.int64)#np.array(np.round(position + rotate_off), dtype=np.int64)

    # ---- Evaluate CPPN
    child_type, child_rotation = __evaluate_cppn(body_net, new_pos, chain_length)
    angle = child_rotation * (np.pi / 2.0)

    # ---- Is setting the child possible?
    bool_None = (child_type is None)
    if bool_None: # Empty block is queried --> No child can be set
        bool_set_child = False
    else:
        # .....?
        child = child_type(angle)
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
