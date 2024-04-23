
import numpy as np
from pyrr import Quaternion, Vector3
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2


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