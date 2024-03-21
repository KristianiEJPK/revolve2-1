"""MorphologicalMeasures class."""
from copy import deepcopy
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from revolve2.modular_robot.body import Module
from revolve2.modular_robot.body.base import ActiveHinge, Body, Brick
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BrickV2
from revolve2.modular_robot.body.base._core import Core
from revolve2.modular_robot.body.v2._attachment_face_core_v2 import AttachmentFaceCoreV2
from typing import Generic, TypeVar

TModule = TypeVar("TModule", bound=np.generic)


class MorphologicalMeasures(Generic[TModule]):

    def __init__(self, body: Body, brain: object, max_modules: int) -> None:
        # Initialize
        self.max_modules = max_modules
        self.core = body.core

        # Get number of modules
        self.bricks = body.find_modules_of_type(Brick)
        self.active_hinges = body.find_modules_of_type(ActiveHinge)
        self.num_modules = 1 + len(self.bricks) + len(self.active_hinges)

        # n neighbours
        self.single_neighbour_bricks = len(self.__calculate_single_neighbour(Brick))
        self.single_neighbour_active_hinges = len(self.__calculate_single_neighbour(ActiveHinge))
        
        # Other measures
        self.grid, self.core_grid_position, self.id_string = body.to_grid(ActiveHingeV2, BrickV2)

        # Check if 2D
        assert self.__calculate_is_2d_recur(self.grid), "Body is not 2D."

    @property
    def size(self) -> float:
        """Goal:
            Get the relative size of the robot (Miras, 2018).
        ----------------------------------------------------
        Input:
            num_modules:
                The number of modules.
            max_modules:
                The maximum number of modules.
        ----------------------------------------------------
        Output:
            The relative size of the robot."""
        return self.num_modules / self.max_modules
    
    @property
    def proportion_2d(self) -> float:
        """
        Goal:
            Get the 'proportion' measurement, which is 
            the ratio between the depth and width of the 
            bounding box around the body. Miras (2018).	
        ----------------------------------------------------
        Input:
            bounding_box_depth: 
                The depth of the bounding box around the body.
            bounding_box_width: 
                The width of the bounding box around the body.
        ----------------------------------------------------
        Output:
            Proportion measurement.
        """
        return min(self.bounding_box_depth, self.bounding_box_width) / max(
            self.bounding_box_depth, self.bounding_box_width)
    

    @property
    def single_neighbour_brick_ratio(self) -> float:
        """Goal:
            Get the ratio between single neighbour bricks and the 
            maximum number of potential single neighbours. Miras (2018).
        -----------------------------------------------------------------
        Input:
            num_bricks:
                The number of bricks of the robot.
        -----------------------------------------------------------------
        Output:
            The proportion.
        """
        # Get maximum number of potential single neighbours
        max_potential_single_neighbour_bricks = self.num_bricks - max(0, (self.num_bricks - 2) // 3)

        # Get the proportion
        if max_potential_single_neighbour_bricks == 0:
            return 0.0
        return (self.single_neighbour_bricks / max_potential_single_neighbour_bricks)
    

    @property
    def single_neighbour_ratio(self) -> float:
        """
        Goal:
            Get the single neighbour ratio.
        ----------------------------------------------------
        Input:
            num_bricks:
                The number of bricks of the robot.
        ----------------------------------------------------
        Output:
            Limbs measurement.
        """
        # ---- Calculate the maximum potential single neighbours based on bricks
        max_potential_single_neighbour_bricks = self.num_bricks - max(0, (self.num_bricks - 2) // 3)

        # ---- Adapt for active hinges
        # Around core --> 4 spots available
        spotsleft = max(0, 4 - self.num_bricks)
        spots2fill = min(spotsleft, self.num_active_hinges)

        # At each brick --> 3 spots available
        addattachments =  min(max_potential_single_neighbour_bricks * 3, (self.num_active_hinges - spots2fill)) # 3 spots available
        spots2fill += addattachments + (((addattachments - 1) // 3) * - 1) - 1 # Each brick attachment will initially coincide with no increase, then next two will lead to increase

        # ---- Calculate the maximum potential single neighbours for all modules
        max_potential_single_neighbour_modules = max_potential_single_neighbour_bricks + spots2fill

        # ------ Calculate the proportion
        if max_potential_single_neighbour_modules == 0:
            return 0.0
        else:
            return ((self.single_neighbour_bricks + 
                 self.single_neighbour_active_hinges) / max_potential_single_neighbour_modules)

    @property
    def double_neigbour_brick_and_active_hinge_ratio(self) -> float:
        """Goal:
            Ratio for double attached modules from Miras (2018).
        --------------------------------------------------------------
        Input:
            num_bricks:
                The number of bricks of the robot.
            num_active_hinges:
                The number of active hinges of the robot.
        --------------------------------------------------------------
        Output:
            The proportion."""
        
        # ---- Get the maximum possible double neighbour bricks and active hinges
        potential_double_neighbour_bricks_and_active_hinges = max(0, self.num_bricks + self.num_active_hinges - 1)
        # ---- Get double neighbour bricks and active hinges
        num_double_neighbour_bricks = len(self.__calculate_double_neighbour(Brick))
        num_double_neighbour_active_hinges = len(self.__calculate_double_neighbour(ActiveHinge))

        # Return the proportion
        if potential_double_neighbour_bricks_and_active_hinges == 0:
            return 0.0

        return (num_double_neighbour_bricks + num_double_neighbour_active_hinges
            ) / potential_double_neighbour_bricks_and_active_hinges
    

    def get_limb(self, module):
        """Goal:
            Get the limb length of the robot. Hereby, the limb length
            is defined as the maximum distance from the core to the
            furthest module in a certain direction.
        -------------------------------------------------------------
        Input:
            module:
                The module.
        -------------------------------------------------------------
        Output:
            The limb length of the robot in each direction.
        """
        # Initialize lengths
        lengths = [0, 0, 0, 0]
        # Get the limb length
        for ichild, child in enumerate(module.children.values()):
            if child != None:
                lengths[ichild] = self.get_limb_length_recur(child, [0, 0, 0, 0])
            else:
                lengths[ichild] = 0
        return lengths
    
    def get_limb_length_recur(self, module, lengths2add):
        """Goal:
            Get the limb length of the robot. Hereby, the limb length
            is defined as the maximum distance from the core to the
            furthest module in a certain direction.
        -----------------------------------------------------------------
        Input:
            module:
                The module.
            lengths2add:
                The lengths to add.
        ------------------------------------------------------------------
        Output:
            The limb length of the robot in each direction from that point.
        """
        # Children present?
        child_bool = list(module.children.values()) != []

        # If children present, get limb length
        if child_bool:
            for ichild, child in enumerate(module.children.values()):
                lengths2add[ichild] += self.get_limb_length_recur(child, [1, 1, 1, 1])
        else:
            pass
        return max(lengths2add)
            
    @property
    def length_of_limbsNiels(self) -> float:
        """
        Goal:
            Get 'length of limbs' measurements.
        ------------------------------------------------------------------
        Input:
            num_bricks:
                The number of bricks of the robot.
            num_active_hinges:
                The number of active hinges of the robot.
            ndouble_neighbour_bricks:
                The number of bricks that are connected to exactly two
                other modules.
            ndouble_neighbour_active_hinges:
                The number of active hinges that are connected to exactly 
                two other modules.
        ------------------------------------------------------------------
        Output:
            Length of limbs measurements: ratio based on mean, ratio based
            on max and .
        """
        # ---- Get the maximum possible limb length
        potential_length_of_limb = max(0, self.num_modules - 1)

        # ---- Get limb length (per core direction)
        limb_length = self.get_limb(self.core)

        # ---- Max as fraction of potential length of limb
        if (potential_length_of_limb == 0) or (self.num_modules == 1):
            maxrel = 0
        else: maxrel = max(limb_length) / (potential_length_of_limb)

        # ---- Mean as fraction of potential length of limb
        if (potential_length_of_limb == 0) or (self.num_modules == 1):
            meanrel = 0
        else: meanrel = np.mean(limb_length) / (potential_length_of_limb)

        # ---- Standard deviation as fraction of maximal unbalanced limb length
        samples = [self.num_modules - 1, 0, 0, 0]
        std_max = np.sqrt(np.mean([((x - np.mean(samples)) ** 2) for x in samples]))
        if (std_max == 0) or (self.num_modules == 1):
            stdrel = 0
        else:
            stdrel = np.sqrt(np.mean([((x - np.mean(limb_length)) ** 2) for x in limb_length])) / std_max

        return maxrel, meanrel, stdrel
    
    @property
    def joints(self):
        """Goal:
            Get a measure for how movable the body is.
        ----------------------------------------------------
        Input:
            num_modules:
                The number of modules.
            num_active_hinges:
                The number of active hinges.
        ----------------------------------------------------
        Output:
            Joints measurement."""
        # ---- Calculate the maximum potential joints
        jmax = np.floor((self.num_modules - 1) / 2)
        # ---- Calculate the proportion
        if self.num_modules < 3:
            return 0
        else:
            # Get hinges that are fully connected, but not to other hinges
            new_hinges = []
            for hinge in self.__calculate_filled(ActiveHinge):
                if all([type(hinge.children.get(child_index)) != ActiveHingeV2
                    for child_index in hinge.attachment_points.keys()]):
                    new_hinges.append(hinge)

            return len(new_hinges) / jmax
    
    @property
    def joint_brick_ratio(self):
        """Goal:
            Get a measure for the ratio between joints and bricks.
        ----------------------------------------------------------
        Input:
            num_bricks:
                The number of bricks of the robot.
            num_active_hinges:
                The number of active hinges of the robot.
        ----------------------------------------------------------
        Output:
            Joints/Bricks ratio."""
        # ---- Calculate the proportion
        if (self.num_modules - 1) == 0:
            return 0
        else:
            ratio = (self.num_active_hinges - self.num_bricks) / (self.num_modules - 1) # (-1, 1)
            return (ratio + 1) / 2 # (0, 1)
        
    @property
    def symmetry(self) -> float:
        """
        Goal: 
            Get the 'symmetry' measurement from the paper.
        -----------------------------------------------------------------------
        Input:
            xz_symmetry: 
                The X/Z-plane symmetry.
            yz_symmetry: 
                The Y/Z-plane symmetry.
        -----------------------------------------------------------------------
        Output:
            Symmetry measurement.
        """
        # Pad the grid
        self.__pad_grid()
        # Get symmetry scores for both 2D planes
        self.xz_symmetry = self.__calculate_xz_symmetry()
        self.yz_symmetry = self.__calculate_yz_symmetry()
        # Return the maximum symmetry score
        return max(self.xz_symmetry, self.yz_symmetry)
    

    @property
    def coverage(self) -> float:
        """
        Goal:
            Get the 'coverage' measurement from the paper.
        ----------------------------------------------------
        Input:
            num_modules:
                The number of modules.
            bounding_box_width:
                The width of the bounding box around the body.
            bounding_box_height:
                The height of the bounding box around the body.
            bounding_box_depth:
                The depth of the bounding box around the body.
        ----------------------------------------------------
        Output:
            Coverage measurement.
        """
        
        bounding_box_volume = self.bounding_box_width * self.bounding_box_height * self.bounding_box_depth

        return (self.num_modules) / (bounding_box_volume - 8) # - 8 because of 3 x 3 core block
    
    @property
    def branching(self) -> float:
        """
        Goal:
            Get the ratio between filled cores and bricks and how many that potentially 
            could have been if this set of modules was rearranged in an optimal way.
            This calculates 'branching' from the paper.
        ----------------------------------------------------------------------------
        Input:
            num_modules:
                The number of modules.
            filled_bricks: 
                The bricks which have all slots filled with other modules.
            max_potentionally_filled_core_and_bricks: 
                The maximum number of core and bricks that could potentially be filled 
                with this set of modules if rearranged in an optimal way.
        ----------------------------------------------------------------------------
        Output:
            The proportion.
        """
        # bmax
        max_filled = max(0, (self.num_modules - 2) // 3)
        max_filled = min(max_filled, 1 + self.num_bricks) # Correct for number of modules with four attachments!

        # Calculate the branching variable
        if self.num_modules < 5:
            return 0
        else:
            return (len(self.__calculate_filled(Brick)) + (
                1 if self.__calculate_core_is_filled() else 0)) / max_filled
        
    @property   
    def surface(self):
        """Goal:
            Calculate the surface area of the robot.
        ----------------------------------------------------
        Input:
            grid:
                The grid of the robot.
            num_modules:
                The number of modules.
        ----------------------------------------------------
        Output:
            The ratio between the minimum surface area of the robot
            and the actual surface area of the robot."""

        # Get the valid values
        valid_values = [tuple(arg) for arg in np.argwhere(self.grid[:, :, [1]] != None)]

        # Get the surface area
        surf = 0
        for valid in valid_values:
            xnext = tuple(valid + np.array([1, 0, 0]))
            xback = tuple(valid + np.array([-1, 0, 0]))
            ynext = tuple(valid + np.array([0, 1, 0]))
            yback = tuple(valid + np.array([0, -1, 0]))
            # Top and Bottom
            surf += 2
            # X forward
            if (xnext[0] == (self.grid.shape[0])) or (xnext not in valid_values):
                surf += 1
            # X back
            if (xback[0] < 0) or (xback not in valid_values):
                surf += 1
            # Y next
            if (ynext[1] == (self.grid.shape[1])) or (ynext not in valid_values):
                surf += 1
            # Y back
            if (yback[1] < 0) or (yback not in valid_values):
                surf += 1

        # Calculate the minimum possible surface
        sqrt = np.sqrt(self.num_modules + 8) # 8 because of 3 x 3 core block
        floor_sqrt = np.floor(sqrt)
        if sqrt % 1 == 0:
            minsurf = (4 * sqrt) + 2 * (self.num_modules + 8) # 4 sides + top and bottom
        else:
            # Blocks remaining
            blocks = (self.num_modules + 8) - (floor_sqrt ** 2)
            # Current lengths
            lengths = [floor_sqrt, floor_sqrt]
            # Current surface
            minsurf = (2 * lengths[0]) + (2 * lengths[1]) + 2 * (self.num_modules + 8)
            while blocks > 0:
                # Maximum side
                side = np.max(lengths)
                # Index of minimum side
                iside = np.argmin(lengths)
                # First block + intermediate blocks and last block
                minsurf += (4 - 1 - 1) + (max(0, (side % 1) - 2) * -1) + ((((side % 1) - 1) > 0) * 0)
                # Increase length and decrease blocks
                lengths[iside] += 1
                blocks -= side
        
        return minsurf / surf

    @property
    def num_bricks(self) -> int:
        """
        Goal:
            Get the number of bricks.
        -------------------------------------------
        Input:
            bricks: 
                The number of bricks of the robot.
        -------------------------------------------
        Output:
            The number of bricks.
        """
        return len(self.bricks)

    @property
    def num_active_hinges(self) -> int:
        """
        Goal:
            Get the number of active hinges.
        ---------------------------------------
        Input:
            active_hinges: 
                The active hinges of the robot.
        ---------------------------------------
        Output:
            The number of active hinges.
        """
        return len(self.active_hinges)
    
    def __calculate_core_is_filled(self) -> bool:
        return all(
            [
                self.core.children.get(child_index) is not None
                for child_index in self.core.attachment_points.keys()
            ]
        )

    def __calculate_filled(self, type):
        if type == Brick:
            return [brick for brick in self.bricks
                if all([brick.children.get(child_index) is not None
                    for child_index in brick.attachment_points.keys()])]
        elif type == ActiveHinge:
            return [active_hinge for active_hinge in self.active_hinges
                if all([active_hinge.children.get(child_index) is not None
                    for child_index in active_hinge.attachment_points.keys()])]
        else:
            raise ValueError("Unknown type {}".format(type))


    def __calculate_single_neighbour(self, type):
        """Goal:
            Calculate the Hinges/bricks that are only connected to one other module.
        ----------------------------------------------------------------------------
        Input:
            bricks or active_hinges: 
                The Hinges/bricks of the robot.
        ----------------------------------------------------------------------------
        Output:
            The Hinges/bricks that are only connected to one other module."""
        if type == Brick:
            return [brick for brick in self.bricks
                if all([brick.children.get(child_index) is None
                        for child_index in brick.attachment_points.keys()])]
        elif type == ActiveHinge:
            return [hinge for hinge in self.active_hinges
            if all([hinge.children.get(child_index) is None
                    for child_index in hinge.attachment_points.keys()])]
        else:
            raise ValueError("Unknown type {}".format(type))
        
    def  __calculate_double_neighbour(self, type):
        """Goal:
            Calculate the Hinges/bricks that are connected to exactly two other modules.
        ---------------------------------------------------------------------------------
        Input:
            type:
                The type of module.
            bricks or active_hinges: 
                The Hinges/bricks of the robot.
        ---------------------------------------------------------------------------------
        Output:
            The Hinges/bricks that are connected to exactly two other modules."""
        if type == Brick:
            return [brick for brick in self.bricks if sum([0 if child is None else 1 for child in brick.children]) == 1]
        elif type == ActiveHinge:
            return [active_hinge for active_hinge in self.active_hinges if sum(
                [0 if child is None else 1 for child in active_hinge.children]) == 1]
        else:
            raise ValueError("Unknown type {}".format(type))
    
    def __pad_grid(self) -> None:
        """Goal:
            Pad the grid with empty modules to create a symmetry grid.
        ---------------------------------------------------------------------------
        Input:
            grid: 
                The grid of the robot.
        ---------------------------------------------------------------------------
        Output:
            The symmetry grid."""
        # Get grid shape
        x, y, _ = self.grid.shape
        # Position of core
        xoffs, yoffs, _ = self.core_grid_position
        # Creat empty grid
        self.symmetry_grid = np.empty(
            shape=(x + max(0, xoffs), y + max(0, yoffs), 1), dtype=Module) # Hier nog even naar kijken!
        
        # Fill with None
        self.symmetry_grid.fill(None)
        # Fill with grid values
        self.symmetry_grid[:x, :y, [0]] = self.grid[:, :, [1]]
        # Plot?
        #self.create_plot(self.symmetry_grid, z = -1)

    def __calculate_xz_symmetry(self) -> float:
        """Goal:
            Calculate a X/Z-plane symmetry score.
        ----------------------------------------------------------------------------
        Input:
            symmetry_grid: 
                The symmetry grid.
            bounding_box_depth: 
                The depth of the bounding box around the body.
            core_grid_position: 
                The position of the core in 'body_as_grid'.
        ----------------------------------------------------------------------------
        Output:
            The X/Z-plane symmetry."""
        # Initialize
        num_along_plane = 0
        num_symmetrical = 0

        # For all coordinates
        for x in range(self.bounding_box_depth):
            # Check if module is present at core y location
            if self.symmetry_grid[x, self.core_grid_position[1], 0] is not None:
                num_along_plane += 1
            # Check if module is symmetrical
            for y in range(1, (self.bounding_box_width - 1) // 2 + 0): # Because highest possible is half of bounding box width
                # If module is symmetrical
                if self.symmetry_grid[x, self.core_grid_position[1] + y, 0] is not None and type(
                    self.symmetry_grid[x, self.core_grid_position[1] + y, 0]) is type(
                    self.symmetry_grid[x, self.core_grid_position[1] - y, 0]):
                    num_symmetrical += 2

        # Calculate difference between number of modules and number along plane
        difference = (self.num_modules + 8) - num_along_plane # + 8 because of 3 x 3 core block

        # Return the symmetry relative to the difference
        assert num_symmetrical <= difference
        return num_symmetrical / difference if (difference > 0.0) else difference

    def __calculate_yz_symmetry(self) -> float:
        """Goal:
            Calculate a Y/Z-plane symmetry score.
        ----------------------------------------------------------------------------
        Input:
            symmetry_grid: 
                The symmetry grid.
            bounding_box_width: 
                The width of the bounding box around the body.
            core_grid_position: 
                The position of the core in 'body_as_grid'.
        ----------------------------------------------------------------------------
        Output:
            The Y/Z-plane symmetry."""
        # Initialize
        num_along_plane = 0
        num_symmetrical = 0

        # For all coordinates
        for y in range(0, self.bounding_box_width):
            # Check module is present at core x location
            if self.symmetry_grid[self.core_grid_position[0], y, 0] is not None:
                num_along_plane += 1
            # Check if module is symmetrical
            for x in range(1, (self.bounding_box_depth - 1) // 2 + 1): # Because highest possible is half of bounding box depth
                if self.symmetry_grid[self.core_grid_position[0] + x, y, 0] is not None and type(
                    self.symmetry_grid[self.core_grid_position[0] + x, y, 0]) is type(
                    self.symmetry_grid[self.core_grid_position[0] - x, y, 0]):
                    num_symmetrical += 2

        # Calculate difference
        difference = (self.num_modules + 8) - num_along_plane # + 8 because of 3 x 3 core block

        # Return the symmetry relative to the difference
        assert num_symmetrical <= difference
        return num_symmetrical / difference if (difference > 0.0) else difference


    @property
    def bounding_box_depth(self) -> int:
        """
        Goal:
            Get the depth of the bounding box around the body.
            Forward/backward axis for the core module.
        -------------------------------------------------------
        Input:
            None
        -------------------------------------------------------
        Output:
            The depth.
        """
        return self.grid.shape[0]

    @property
    def bounding_box_width(self) -> int:
        """
        Goal:
            Get the width of the bounding box around the body.
            Right/left axis for the core module.
        -------------------------------------------------------
        Input:
            None
        -------------------------------------------------------
        Output:
            The width.
        """
        return self.grid.shape[1]
    
    @property
    def bounding_box_height(self) -> int:
        """
        Get the height of the bounding box around the body.

        Up/down axis for the core module.

        :returns: The height.
        """
        return self.grid.shape[2]
    
    @classmethod
    def __calculate_is_2d_recur(cls, grid) -> bool:
        """Goal:
            Calculate if the robot is two dimensional.
        -------------------------------------------------------
        Input:
            grid: The grid of the robot.
        -------------------------------------------------------
        Output:
            If the robot is two dimensional.
        """
        return (grid.shape[2] == 3) and ((grid[:, :, 2] != None).sum() == 8) and ((grid[:, :, 0] != None).sum() == 8)

    def create_plot(self, grid, z = 0, movable = None):
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
                
        # Create a custom colormap with 4 colors
        cmap = plt.cm.colors.ListedColormap(['grey', 'red', 'white', 'blue'])

        # Create a normalized color map
        norm = plt.cm.colors.Normalize(vmin = 0, vmax = 3)

        # Plot
        plt.imshow(newgrid[:, :, z].astype(int), cmap = cmap, norm = norm)

        if isinstance(movable, np.ndarray):
            plt.scatter(movable[:, 1], movable[:, 0], c = 'black', s = 100, marker = 'o')
        
        return plt.show()
    