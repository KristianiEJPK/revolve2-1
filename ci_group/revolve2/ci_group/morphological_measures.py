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
from typing import Generic, TypeVar, Union

# Type variable
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
        #self.create_plot(self.grid, z = 0)

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
    def proportion_2d(self) -> float: # Maybe improve the proportion metric, e.g. get std across 4 directions!
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
    def proportionNiels(self) -> float:
        """
        Goal:
            Improved version of proportion measurement.
        -------------------------------------------------------------------
        """
        # Get positions
        positions = [self.core_grid_position[0], self.core_grid_position[1]]
        positions.append(self.bounding_box_depth - self.core_grid_position[0] - 1) # 0, 1, 2 --> 3 - 1 - 1 = 1
        positions.append(self.bounding_box_width - self.core_grid_position[1] - 1)
        

        # Get max possible std
        samples = [self.num_modules, 1, 1, 1] # num_modules = 1 + num_bricks + num_active_hinges
        std_max = np.sqrt(np.mean([((x - np.mean(samples)) ** 2) for x in samples]))

        # Get std
        std = np.sqrt(np.mean([((x - np.mean(positions)) ** 2) for x in positions]))

        # Return the proportion
        if (std_max == 0) or (self.num_modules == 1):
            return 0.0
        else:
            return std / std_max
        

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
        -------------------------------------------------------------------------------------
        Input:
            num_bricks:
                The number of bricks of the robot.
            num_active_hinges:
                The number of active hinges of the robot.
            single_neighbour_bricks:
                The number of bricks that are connected to exactly one other module.
            single_neighbour_active_hinges:
                The number of active hinges that are connected to exactly one other module.
        -------------------------------------------------------------------------------------
        Output:
            Limbs measurement.
        """
        # ---- Calculate the maximum potential single neighbours based on bricks
        max_potential_single_neighbour_bricks = self.num_bricks - max(0, (self.num_bricks - 2) // 3)
        
        # ---- Calculate the spots left for ActiveHinges
        ## Around core & left-over attachment of non-single neighbour bricks
        if self.num_bricks <= 4:
            # Around core --> 4 spots available for modules
            spotsleft = max(0, 4 - self.num_bricks) # Number of spots left
            spots2fill_core = min(spotsleft, self.num_active_hinges) # How many can we fill by hinges?
            # Spots on non-single neighbour bricks
            spots2fill_nonsingle = 0
        else:
            # Around core
            spots2fill_core = 0
            # Spots on non-single neighbour bricks
            # e.g. 5 - 2 = 3 --> 3 % 3 = 0 --> 2 - 0 = 2 spots left (5:3, 6:4, 7:5, (8:6))
            spots2fill_nonsingle = min(2 - max(0, (self.num_bricks - 2) % 3), self.num_active_hinges)
        # Assert
        assert spots2fill_core >= 0, "Number of spots left on core is negative."
        assert spots2fill_nonsingle >= 0, "Number of spots left on non-single bricks is negative."
        
        ## Single neighbour brick attachments
        # At each brick --> 3 spots available
        left_over_hinges = self.num_active_hinges - spots2fill_core - spots2fill_nonsingle # How many hinges are left?
        addattachments = min(max_potential_single_neighbour_bricks * 3, left_over_hinges) # 3 spots available
        # Each hinge-brick attachment will initially coincide with no increase, then next two will lead to increase
        spots2add = spots2fill_core + spots2fill_nonsingle + addattachments + (((addattachments - 1) // 3) * - 1) - 1 

        # ---- Calculate the maximum potential single neighbours for all modules
        max_potential_single_neighbour_modules = max_potential_single_neighbour_bricks + spots2add

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
        
        # # ---- Get the maximum possible double neighbour bricks and active hinges
        potential_double_neighbour_bricks_and_active_hinges = max(0, self.num_bricks + self.num_active_hinges - 1)
        # # ---- Get double neighbour bricks and active hinges
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
            on max and based on std.
        """
        # ---- Get the maximum possible limb length
        potential_length_of_limb = max(0, self.num_modules - 1)

        # ---- Get limb length (per core direction)
        limb_length = np.array(self.get_limb(self.core))

        # # ---- Max as fraction of potential length of limb
        # if (potential_length_of_limb == 0) or (self.num_modules == 1):
        #     maxrel = 0
        # else: maxrel = max(limb_length) / (potential_length_of_limb)

        # # ---- Mean as fraction of potential length of limb
        # if (potential_length_of_limb == 0) or (self.num_modules == 1):
        #     meanrel = 0
        # else: meanrel = np.mean(limb_length) / (potential_length_of_limb)

        # # ---- Standard deviation as fraction of maximal unbalanced limb length
        # samples = [self.num_modules - 1, 0, 0, 0]
        # std_max = np.sqrt(np.mean([((x - np.mean(samples)) ** 2) for x in samples]))
        # if (std_max == 0) or (self.num_modules == 1):
        #     stdrel = 0
        # else:
        #     stdrel = np.sqrt(np.mean([((x - np.mean(limb_length)) ** 2) for x in limb_length])) / std_max

        # Max
        if (potential_length_of_limb == 0):
            maxrel = 0
        else:
            maxrel = np.max(limb_length) / potential_length_of_limb

        # Mean, std and number of limbs
        if sum(limb_length) == 0:
            meanrel, stdrel, nlimbs = 0, 0, 0
        else:
            # Get limbs
            limbs = limb_length[limb_length > 0]
            # Mean length
            meanrel = np.mean(limbs) / potential_length_of_limb
            # Standard deviation
            samples = [self.num_modules - 1] + [0] * (len(limbs) - 1)
            std_max = np.sqrt(np.mean([((x - np.mean(samples)) ** 2) for x in samples]))
            if (std_max == 0) or (self.num_modules == 1):
                stdrel = 0
            else:
                stdrel = np.sqrt(np.mean([((x - np.mean(limbs)) ** 2) for x in limbs])) / std_max
                
            # Number of limbs
            nlimbs = len(limbs)

        return maxrel, meanrel, stdrel, nlimbs
    
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
    def symmetry(self) -> Union[list[float], list[float]]:
        """
        Goal: 
            Get symmetry along x-axis, y-axis and diagonals.
        -----------------------------------------------------------------------
        Input:
            ...
        -----------------------------------------------------------------------
        Output:
            * Symmetry measurements: including module type
            * Symmetry measurements: excluding module type
        """
        # Pad the grid
        self.__pad_grid()
        # Get symmetry scores for both 2D planes
        self.symmetry_xax = self.__calculate_symmetry_xax()
        self.symmetry_yax = self.__calculate_symmetry_yax()
        self.symmetry_diag, self.symmetry_diag_excl = self.__calculate_symmetry_diag()

        # Return the maximum symmetry score
        incl = [self.symmetry_xax[0], self.symmetry_yax[0], self.symmetry_diag[0], self.symmetry_diag[1]]
        excl = [self.symmetry_xax[1], self.symmetry_yax[1], self.symmetry_diag_excl[0], self.symmetry_diag_excl[1]]

        return incl, excl
    

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
            bounding_box_depth:
                The depth of the bounding box around the body.
        ----------------------------------------------------
        Output:
            Coverage measurement.
        """
        
        bounding_box_volume = self.bounding_box_width * self.bounding_box_depth * 1

        return self.num_modules / bounding_box_volume
    
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
        children = []
        for child_index in self.core.attachment_points.keys():
            att = self.core.children.get(child_index)
            if att.children != {}:
                children.append(True)
        # Core filled?
        return sum(children) == 4

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
        row, col, _ = self.grid.shape
        # Position of core
        row_offs, col_offs, _ = self.core_grid_position
        # Get offsets
        row_off_top = row_offs - 1 # Minus 1 because of 3 x 3 core block
        row_off_bottom = (self.bounding_box_depth - 1) - row_offs - 1 # e.g. 0, 1, 2, 3 --> 4 - 1 - 1 - 1 = 1
        col_off_left = col_offs - 1
        col_off_right = (self.bounding_box_width - 1) - col_offs - 1

        # Creat empty grid
        diff_off_row = abs(row_off_top - row_off_bottom)
        diff_off_col = abs(col_off_left - col_off_right)
        self.symmetry_grid = np.empty(shape = (row + diff_off_row, col + diff_off_col, 1), dtype = Module)
        
        # Fill with None
        self.symmetry_grid.fill(None)
        # Fill with grid values
        start_row = 0 if row_off_top >= row_off_bottom else diff_off_row
        end_row = row if row_off_top >= row_off_bottom else row + diff_off_row
        start_col = 0 if col_off_left >= col_off_right else diff_off_col
        end_col = col if col_off_left >= col_off_right else col + diff_off_col
        self.symmetry_grid[start_row:end_row, start_col:end_col, [0]] = self.grid[:, :, [1]]

        # Save new core position
        self.symmetry_core_coordinates = (start_row + row_offs, start_col + col_offs) 

        ## Plot?
        #self.create_plot(self.symmetry_grid, z = -1)

    def __calculate_symmetry_xax(self) -> Union[float, float]:
        """Goal:
            Calculate symmetry along the x-axis.
        ----------------------------------------------------------------------------
        Input:
            symmetry_grid: 
                The symmetry grid.
            symmetry_core_coordinates:
                The core coordinates in the symmetry grid.
        ----------------------------------------------------------------------------
        Output:
            * Symmetry along the x-axis including module type
            * Symmetry along the x-axis excluding module type."""
        # ---- Initialize
        num_along_plane = 0
        num_symmetrical = 0 # Including module type
        num_symmetrical_excl = 0 # Excluding module type

        # ---- Get y position of core
        ycore_position = self.symmetry_core_coordinates[0]
        
        # ---- For all coordinates
        for x in range(0, self.symmetry_grid.shape[1]): # For all x-coordinates
            # Check if module is present at (fixed) core y location
            if self.symmetry_grid[ycore_position, x, 0] is not None: 
                num_along_plane += 1
                
            # Check if module is symmetrical in y direction
            for y in range(2, ycore_position + 1): # Exclude core block
                # If module is symmetrical
                if self.symmetry_grid[ycore_position + y, x, 0] is not None and type(
                    self.symmetry_grid[ycore_position + y, x, 0]) is type(
                    self.symmetry_grid[ycore_position - y, x, 0]):
                    num_symmetrical += 2
                if (self.symmetry_grid[ycore_position + y, x, 0] is not None) and (
                    self.symmetry_grid[ycore_position - y, x, 0] is not None
                ):
                    num_symmetrical_excl += 2
        
        # Calculate difference between number of modules and number along plane
        difference = (self.num_modules + 2) - num_along_plane # + 2 because of 3 x 3 core block

        # Assert
        assert num_symmetrical <= difference
        assert num_symmetrical_excl <= difference
        assert num_symmetrical_excl >= num_symmetrical

        # Calculate the symmetry
        symmetries = []
        symmetries.append(num_symmetrical / difference if (difference > 0.0) else difference)
        symmetries.append(num_symmetrical_excl / difference if (difference > 0.0) else difference)

        return symmetries

    def __calculate_symmetry_yax(self) -> Union[float, float]:
        """Goal:
            Calculate symmetry along the y-axis.
        ----------------------------------------------------------------------------
        Input:
            symmetry_grid: 
                The symmetry grid.
            symmetry_core_coordinates:
                The core coordinates in the symmetry grid.
        ----------------------------------------------------------------------------
        Output:
            * Symmetry along the y-axis including module type
            * Symmetry along the y-axis excluding module type."""
        # Initialize
        num_along_plane = 0
        num_symmetrical = 0 # Including module type
        num_symmetrical_excl = 0 # Excluding module type

        # ---- Get x position of core
        xcore_position = self.symmetry_core_coordinates[1]

        # For all coordinates
        for y in range(0, self.symmetry_grid.shape[0]):
            # Check module is present at core x location
            if self.symmetry_grid[y, xcore_position, 0] is not None:
                num_along_plane += 1
            # Check if module is symmetrical
            for x in range(2, xcore_position + 1):
                if self.symmetry_grid[y, xcore_position + x, 0] is not None and type(
                    self.symmetry_grid[y, xcore_position + x, 0]) is type(
                    self.symmetry_grid[y, xcore_position - x, 0]):
                    num_symmetrical += 2
                if (self.symmetry_grid[y, xcore_position + x, 0] is not None) and (
                    self.symmetry_grid[y, xcore_position - x, 0] is not None
                ):
                    num_symmetrical_excl += 2

        # Calculate difference
        difference = (self.num_modules + 2) - num_along_plane # + 2 because of 3 x 3 core block

        # Assert
        assert num_symmetrical <= difference
        assert num_symmetrical_excl <= difference
        assert num_symmetrical_excl >= num_symmetrical

        # Calculate the symmetry
        symmetries = []
        symmetries.append(num_symmetrical / difference if (difference > 0.0) else difference)
        symmetries.append(num_symmetrical_excl / difference if (difference > 0.0) else difference)

        return symmetries

    def __calculate_symmetry_diag(self) -> Union[list[float], list[float]]:
        """Goal:'
            Calculate symmetry along the diagonals.
        ----------------------------------------------------------------------------
        Input:
            symmetry_grid: 
                The symmetry grid.
            symmetry_core_coordinates:
                The core coordinates in the symmetry grid.
        ----------------------------------------------------------------------------
        Output:
            * Symmetry along the diagonals including module type
            * Symmetry along the diagonals excluding module type"""
        # ---- Initialize
        diagsyms = [] # Including module type
        diagsyms_excl = [] # Excluding module type

        # ---- For all diagonals
        for diag in [1, -1]: # 1 is from right to left, -1 is from left to right
            # Initialize
            num_along_plane = 0
            num_symmetrical = 0 # Including module type
            num_symmetrical_excl = 0 # Excluding module type
        
            # ---- Find start
            xstart = self.symmetry_core_coordinates[1]
            ystart = self.symmetry_core_coordinates[0]
            while True:
                if (((diag == 1) and ((xstart == (self.symmetry_grid.shape[1] - 1)) or (ystart == (self.symmetry_grid.shape[0] - 1))))
                    or
                    ((diag == -1) and ((xstart == 0) or (ystart == (self.symmetry_grid.shape[0] - 1))))):
                    break
                else:
                    xstart += diag * 1 # + if from left to right!
                    ystart -= 1

            # ---- From core towards left top
            ypos = ystart
            xpos = xstart
            while True:
                if (((diag == 1) and ((xpos < 0) or (ypos < 0)))
                    or
                    ((diag == -1) and ((xpos >= self.symmetry_grid.shape[1]) or (ypos < 0)))):
                    break
                else:
                    # Num along plane
                    if self.symmetry_grid[ypos, xpos, 0] is not None:
                        num_along_plane += 1
                    # Num symmetrical
                    ypositions = [ypos + (diag * -1), ypos + (diag * 1)] # if from right to left, -1, 1
                    xpositions = [xpos - 1, xpos + 1]
                    while True:
                        if (ypositions[0] < 0) or (ypositions[1] >= self.symmetry_grid.shape[0]) or (
                            xpositions[0] < 0) or (xpositions[1] >= self.symmetry_grid.shape[1]):
                            break
                        else:
                            # Within core block?
                            bool_core = ((xpositions[0] in [self.symmetry_core_coordinates[1] - 1, self.symmetry_core_coordinates[1], 
                                    self.symmetry_core_coordinates[1] + 1])

                                    and

                                    (ypositions[0] in [self.symmetry_core_coordinates[0] - 1, self.symmetry_core_coordinates[0],
                                                       self.symmetry_core_coordinates[0] + 1]))
                            # Check if symmetrical
                            if (self.symmetry_grid[ypositions[0], xpositions[0], 0] is not None) and (
                                type(self.symmetry_grid[ypositions[0], xpositions[0], 0]) is type(
                                    self.symmetry_grid[ypositions[1], xpositions[1], 0])):
                                if bool_core:
                                    pass
                                else:
                                    num_symmetrical += 2
                            
                            if (self.symmetry_grid[ypositions[0], xpositions[0], 0] is not None) and (
                                self.symmetry_grid[ypositions[1], xpositions[1], 0] is not None):
                                if bool_core:
                                    pass
                                else:
                                    num_symmetrical_excl += 2

                            # Adapt positions
                            ypositions[0] -= (diag * 1) # - if from right to left!
                            ypositions[1] += (diag * 1) # + if from right to left!
                            xpositions[0] -= 1
                            xpositions[1] += 1
                    # Adapt positions
                    ypos += 1
                    xpos -= (diag * 1) # - if from right to left!

            # Calculate difference
            difference = (self.num_modules + 2) - num_along_plane # + 2 because of 3 x 3 core block

            # Assert
            assert num_symmetrical <= difference
            assert num_symmetrical_excl <= difference
            assert num_symmetrical_excl >= num_symmetrical

            # Calculate the symmetry
            if difference > 0.0:
                diagsyms.append(num_symmetrical / difference)
                diagsyms_excl.append(num_symmetrical_excl / difference)
            else:
                diagsyms.append(difference)
                diagsyms_excl.append(difference)
    
        return diagsyms, diagsyms_excl

            
            

        


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
    