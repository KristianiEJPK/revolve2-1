from copy import deepcopy
from develop_from_string import get_body
from itertools import product
import numpy as np
import pandas as pd
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BrickV2
from revolve2.modular_robot.body.base._core import Core
from revolve2.modular_robot.body.v2._attachment_face_core_v2 import AttachmentFaceCoreV2
import sys


def create_plot(grid, z = 0):
    """Goal:
        Create a plot of the robot.
    -------------------------------------------------------
    Input:
        grid: The grid of the robot
    -------------------------------------------------------
    Output:
        A plot of the robot."""
    # Create a copy
    newgrid = deepcopy(grid)
    
    # Fill the new grid
    for x, y in product(range(grid.shape[0]), range(grid.shape[1])):
        if type(grid[x, y, z + 1]) == BrickV2:
            newgrid[x, y, z] = 3
        elif type(grid[x, y, z + 1]) == ActiveHingeV2:
            newgrid[x, y, z] = 2
        elif type(grid[x, y, z + 1]) == AttachmentFaceCoreV2:
            newgrid[x, y, z] = 1
        elif type(grid[x, y, z + 1]) == Core:
            newgrid[x, y, z] = 1
        else:
            newgrid[x, y, z] = 0
    
    return newgrid[:, :, z].astype(int)

def string2grid(string):
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
    
    # ---- Develop body
    body = get_body(max_parts, dict_coord)

    # ---- Get Grid
    grid, core_grid_position, id_string = body.to_grid(ActiveHingeV2, BrickV2)
    assert string == id_string, "Error in string to grid conversion"

    # ---- Grid to image
    grid = create_plot(grid)
            
    return grid, core_grid_position

def calculate_symmetry(grid, core_grid_position):
    # Initialize
    results = []

    # How much on each side from center
    s1 = (grid.shape[0] - 1) - core_grid_position[0]
    s2 = (grid.shape[1] - 1) - core_grid_position[1]
    s3 = core_grid_position[0]
    s4 = core_grid_position[1]

    # Pad new grid with maximum of (s1, s2, s3, s4) at every side
    maxside = max(s1, s2, s3, s4)
    new_grid = np.zeros((maxside * 2 + 1 , maxside * 2 + 1))

    # Place old grid with core at center
    assert (new_grid.shape[0] - 1) % 2 == 0, "New grid should have an odd number of rows"
    new_grid[maxside - s3:maxside - s3 + grid.shape[0], 
            maxside - s4:maxside - s4 + grid.shape[1]] = grid


    for mode in ["diag1", "diag2", "horizontal", "vertical"]:
        if mode == "diag1":
            # ---- Left top to right bottom diagonal
            # Get lower and upper triangle
            lower = np.tril(new_grid, k = -1)
            upper = np.triu(new_grid, k = 1)
            #upper = np.rot90(np.flipud(upper), 1).astype(int)
            upper = np.transpose(upper)

        elif mode == "diag2":
            # ---- Left bottom to right top diagonal
            # Get lower and upper triangle
            new_grid = np.rot90(new_grid, -1).astype(int)
            lower = np.rot90(np.tril(new_grid, k = -1), 1).astype(int)
            upper = np.rot90(np.triu(new_grid, k = 1), 1).astype(int)
            upper = np.transpose(upper)
            upper = np.fliplr(np.flipud(upper))

        elif mode == "vertical":
            # Around vertical axis
            upper = new_grid[0:maxside, :]
            lower = new_grid[maxside + 1:, :]
            upper = np.flipud(upper)
        elif mode == "horizontal":
            # Around horizontal axis
            upper = new_grid[:, 0:maxside]
            lower = new_grid[:, maxside + 1:]
            upper = np.fliplr(upper)

        # ---- Symmetry including type
        # Get sum of lower and upper triangle
        equality = np.logical_and((upper == lower), (upper != 0))
        # Calculate symmetry
        counttot = np.sum(upper > 0) + np.sum(lower > 0)
        symmetry = np.sum(equality) * 2 / counttot
        results.append(symmetry)

        # ---- Symmetry excluding type
        equality = np.logical_and((upper > 0), (lower > 0))
        # Calculate symmetry
        symmetry = np.sum(equality) * 2 / counttot
        results.append(symmetry)
    
    return results



def main():
    start_string, end_string = int(sys.argv[1]), int(sys.argv[2])
    print(f"Start string: {start_string}, end string: {end_string}")
    # Initialize storage
    storage = {}
    # Read data
    strings = pd.read_csv("C:\\Users\\niels\\OneDrive\\Documenten\\GitHub\\revolve2\\examples\\robot_bodybrain_ea_database\\id_strings.csv")
    strings = strings["id_string"][start_string:end_string]
    for istring, string in enumerate(strings):
        # Print progress
        print(istring)

        # Get grid
        grid, core_grid_position = string2grid(string)

        # Initialize
        storage[string] = calculate_symmetry(grid, core_grid_position)

        # Save data every 1000 strings
        if ((istring % 10000 == 0) and (istring != 0)) or (istring == (len(strings) - 1)):
            pd.DataFrame.from_dict(storage, orient = "index").to_csv(f"C:\\Users\\niels\\OneDrive\\Documenten\\GitHub\\revolve2\\examples\\robot_bodybrain_ea_database\\symmetry_data\\symmetry_{start_string + istring}.csv")
            storage = {}

if __name__ == "__main__":
    main()