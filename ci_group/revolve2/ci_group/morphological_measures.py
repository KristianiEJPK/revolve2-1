"""MorphologicalMeasures class."""
from itertools import product
from typing import Generic, TypeVar

from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
from pyrr import Vector3

from revolve2.modular_robot.body import Module
from revolve2.modular_robot.body.base import ActiveHinge, Body, Brick, Core
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BrickV2
from revolve2.modular_robot.body.v2._attachment_face_core_v2 import AttachmentFaceCoreV2

TModule = TypeVar("TModule", bound=np.generic)


class MorphologicalMeasures(Generic[TModule]):
    # """
    # Modular robot morphological measures.

    # Only works for robot with only right angle module rotations (90 degrees).
    # Some measures only work for 2d robots, which is noted in their docstring.

    # Some measures are based on the following paper:
    # Miras, K., Haasdijk, E., Glette, K., Eiben, A.E. (2018).
    # Search Space Analysis of Evolvable Robot Morphologies.
    # In: Sim, K., Kaufmann, P. (eds) Applications of Evolutionary Computation.
    # EvoApplications 2018. Lecture Notes in Computer Science(), vol 10784. Springer, Cham.
    # https://doi.org/10.1007/978-3-319-77538-8_47
    # """

    # """Represents the modules of a body in a 3D tensor."""
    # grid: NDArray[TModule]
    # symmetry_grid: NDArray[TModule]
    # """Position of the core in 'body_as_grid'."""
    # core_grid_position: Vector3[np.int_]

    # """If the robot is two dimensional, i.e. all module rotations are 0 degrees."""
    # is_2d: bool

    # core: Core
    # bricks: list[Brick]
    # active_hinges: list[ActiveHinge]

    # """If all slots of the core are filled with other modules."""
    # core_is_filled: bool

    # """Bricks which have all slots filled with other modules."""
    # filled_bricks: list[Brick]

    # """Active hinges which have all slots filled with other modules."""
    # filled_active_hinges: list[ActiveHinge]

    # """
    # Bricks that are only connected to one other module.

    # Both children and parent are counted.
    # """
    # single_neighbour_bricks: list[Brick]

    # """
    # Bricks that are connected to exactly two other modules.

    # Both children and parent are counted.
    # """
    # double_neighbour_bricks: list[Brick]

    # """
    # Active hinges that are connected to exactly two other modules.

    # Both children and parent are counted.
    # """
    # double_neighbour_active_hinges: list[ActiveHinge]

    # """
    # X/Y-plane symmetry according to the paper but in 3D.

    # X-axis is defined as forward/backward for the core module
    # Y-axis is defined as left/right for the core module.
    # """
    # xy_symmetry: float

    # """
    # X/Z-plane symmetry according to the paper but in 3D.

    # X-axis is defined as forward/backward for the core module
    # Z-axis is defined as up/down for the core module.
    # """
    # xz_symmetry: float

    # """
    # Y/Z-plane symmetry according to the paper but in 3D.

    # Y-axis is defined as left/right for the core module.
    # Z-axis is defined as up/down for the core module.
    # """
    # yz_symmetry: float

    def __init__(self, body: Body, max_modules: int) -> None:
    #     """
    #     Initialize this object.

    #     :param body: The body to measure.
    #     """
        
        # Initialize
        self.max_modules = max_modules

        # Get number of modules
        self.bricks = body.find_modules_of_type(Brick)
        self.active_hinges = body.find_modules_of_type(ActiveHinge)
        self.num_modules = 1 + len(self.bricks) + len(self.active_hinges)
        
        # Other measures
        self.grid, self.core_grid_position = body.to_grid()
        import matplotlib.pyplot as plt
        newgrid = deepcopy(self.grid)
        z = 1
        for x, y in product(range(self.grid.shape[0]), range(self.grid.shape[1])):
            print(type(self.grid[x, y, z]))
            if type(self.grid[x, y, z]) == BrickV2:
                newgrid[x, y, z] = 3
            elif type(self.grid[x, y, z]) == ActiveHingeV2:
                newgrid[x, y, z] = 2
            elif type(self.grid[x, y, z]) == AttachmentFaceCoreV2:
                newgrid[x, y, z] = 1
            elif type(self.grid[x, y, z]) == Core:
                newgrid[x, y, z] = 1
            else:
                newgrid[x, y, z] = 0
        # Create a custom colormap with 4 colors
        cmap = plt.cm.colors.ListedColormap(['grey', 'red', 'white', 'blue'])

        # Create a normalized color map
        norm = plt.cm.colors.Normalize(vmin = 0, vmax = 3)
        #print(newgrid)
        plt.imshow(newgrid[:, :, z].astype(int), cmap = cmap, norm = norm)
        plt.show()
        # Blauwe zit nog aan verkeerde kant? --> oplossen!!!!

        vmb
        self.core = body.core
        self.core_is_filled = self.__calculate_core_is_filled()
        self.filled_active_hinges = self.__calculate_filled(ActiveHinge)
        self.nsingle_neighbour_bricks = len(self.__calculate_single_neighbour(Brick))
        self.nsingle_neighbour_hinges = len(self.__calculate_single_neighbour(ActiveHinge))
        self.ndouble_neighbour_active_hinges = len(self.__calculate_double_neighbour_active_hinges())

        # Symmetry
        self.__pad_grid()
        self.xy_symmetry = self.__calculate_xy_symmetry()
        self.xz_symmetry = self.__calculate_xz_symmetry()
        self.yz_symmetry = self.__calculate_yz_symmetry()

    
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
    def size(self) -> float:
        """Goal:
            Get the relative size of the robot.
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
            bounding box around the body. Only for 2d robots.
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
        assert self.__calculate_is_2d_recur(self.core)

        return min(self.bounding_box_depth, self.bounding_box_width) / max(
            self.bounding_box_depth, self.bounding_box_width)

    @property
    def limbs(self) -> float:
        """
        Goal:
            Get a measure for limbs.
        ----------------------------------------------------
        Input:
            num_bricks:
                The number of bricks of the robot.
            nsingle_neighbour_bricks:
                The number of bricks that are only connected to 
                    one other module.
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
            return ((self.nsingle_neighbour_hinges + 
                 self.nsingle_neighbour_bricks) / max_potential_single_neighbour_modules)

    @property
    def length_of_limbs(self) -> float:
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
        if (potential_length_of_limb == 0) and (self.num_modules == 1):
            maxrel = 0
        else: maxrel = max(limb_length) / (potential_length_of_limb + 1)

        # ---- Mean as fraction of potential length of limb
        if (potential_length_of_limb == 0) and (self.num_modules == 1):
            meanrel = 0
        else: meanrel = np.mean(limb_length) / (potential_length_of_limb + 1)

        # ---- Standard deviation as fraction of maximal unbalanced limb length
        samples = [self.num_modules - 1, 0, 0, 0]
        std_max = np.sqrt(np.mean([((x - np.mean(samples)) ** 2) for x in samples]))
        if (std_max == 0) and (self.num_modules == 1):
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
        jmax = np.ceil((self.num_modules - 1) / 2)
        # ---- Calculate the proportion
        if self.num_modules < 3:
            return 0
        else:
            # Get hinges that are fully connected, but not to other hinges
            new_hinges = []
            for hinge in self.filled_active_hinges:
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
            Get the 'symmetry' measurement from the paper, but extended to 3d.
        -----------------------------------------------------------------------
        Input:
            xy_symmetry: 
                The X/Y-plane symmetry.
            xz_symmetry: 
                The X/Z-plane symmetry.
            yz_symmetry: 
                The Y/Z-plane symmetry.
        -----------------------------------------------------------------------
        Output:
            Symmetry measurement.
        """
        return max(self.xy_symmetry, self.xz_symmetry, self.yz_symmetry)
    

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
            core_is_filled: 
                If all slots of the core are filled with other modules.
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
            return (len(self.__calculate_filled(Brick)) + (1 if self.core_is_filled else 0)) / max_filled
        
    @property   
    def surface(self):
        assert self.__calculate_is_2d_recur(self.core)
        
        # Get None values
        none_values = np.argwhere(self.grid == None)

        # Get surface loss/gain due to None values
        add2surf = 0
        for none_value in none_values:
            # x-direction
            if ((none_value[0] + 1) < self.grid.shape[0]) and (self.grid[none_value[0] + 1, none_value[1], none_value[2]] != None):
                add2surf += 1
            elif ((none_value[0] + 1) >= self.grid.shape[0]):
                add2surf -= 1

            if ((none_value[0] - 1) >= 0) and (self.grid[none_value[0] - 1, none_value[1], none_value[2]] != None):
                add2surf += 1
            elif ((none_value[0] - 1) < 0):
                add2surf -= 1

            # y-direction
            if ((none_value[1] + 1) < self.grid.shape[1]) and (self.grid[none_value[0], none_value[1] + 1, none_value[2]] != None):
                add2surf += 1
            elif ((none_value[1] + 1) >= self.grid.shape[1]):
                add2surf -= 1

            if ((none_value[1] - 1) >= 0) and (self.grid[none_value[0], none_value[1] - 1, none_value[2]] != None):
                add2surf += 1
            elif ((none_value[1] - 1) < 0):
                add2surf -= 1
            # Loss in z-direction
            add2surf -= 2
        
        # Grid surface
        surf = (self.grid.shape[0] * self.grid.shape[1] * 2) + (2 * self.grid.shape[0]) + (2 * self.grid.shape[1]) + add2surf


        # Calculate the minimum possible surface
        sqrt = np.sqrt(self.num_modules)
        floor_sqrt = np.floor(sqrt)
        floor_square = floor_sqrt ** 2
        if sqrt % 1 == 0:
            minsurf = (4 * sqrt) + 2 * self.num_modules
        else:
            blocks = self.num_modules - floor_square
            lengths = [floor_sqrt, floor_sqrt]
            minsurf = (2 * lengths[0]) + (2 * lengths[1]) + 2 * self.num_modules
            while blocks > 0:
                side = np.max(lengths)
                iside = np.argmin(lengths) # Opposite
                # First block + intermediate blocks and last block
                minsurf += (4 - 1 - 1) + (max(0, (side % 1) - 2) * -1) + ((((side % 1) - 1) > 0) * 0)
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
            return [
                brick
                for brick in self.bricks
                if all([brick.children.get(child_index) is None
                        for child_index in brick.attachment_points.keys()])]
        elif type == ActiveHinge:
            return [hinge for hinge in self.active_hinges
            if all([hinge.children.get(child_index) is None
                    for child_index in hinge.attachment_points.keys()])]
        else:
            raise ValueError("Unknown type {}".format(type))
    

    def __calculate_double_neighbour_active_hinges(self) -> list[ActiveHinge]:
        return [
            active_hinge
            for active_hinge in self.active_hinges
            if sum(
                [
                    0 if active_hinge.children.get(child_index) is None else 1
                    for child_index in active_hinge.attachment_points.keys()
                ]
            )
            == 1
        ]

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
        x, y, z = self.grid.shape
        # Position of core
        xoffs, yoffs, zoffs = self.core_grid_position
        # Creat empty grid
        self.symmetry_grid = np.empty(
            shape=(x + xoffs, y + yoffs, z + zoffs), dtype=Module)
        # Fill with None
        self.symmetry_grid.fill(None)
        # Fill with grid values
        self.symmetry_grid[:x, :y, :z] = self.grid

    def __calculate_xy_symmetry(self) -> float:
        """Goal:
            Calculate the X/Y-plane symmetry according to the paper but in 3D.
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
            The X/Y-plane symmetry."""
        # Initialize
        num_along_plane = 0
        num_symmetrical = 0

        for x, y, z in product(
            range(self.bounding_box_depth),
            range(self.bounding_box_width),
            range(1, (self.bounding_box_height - 1) // 2),):
            if self.symmetry_grid[x, y, self.core_grid_position[2]] is not None:
                num_along_plane += 1
            if self.symmetry_grid[
                x, y, self.core_grid_position[2] + z
            ] is not None and type(
                self.symmetry_grid[x, y, self.core_grid_position[2] + z]) is type(
                self.symmetry_grid[x, y, self.core_grid_position[2] - z]):
                num_symmetrical += 2

        # Calculate difference
        difference = self.num_modules - num_along_plane
        return num_symmetrical / difference if difference > 0.0 else difference

    def __calculate_xz_symmetry(self) -> float:
        """Goal:
            Calculate the X/Z-plane symmetry according to the paper but in 3D.
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

        for x, y, z in product(
            range(self.bounding_box_depth),
            range(1, (self.bounding_box_width - 1) // 2),
            range(self.bounding_box_height),
        ):
            if self.symmetry_grid[x, self.core_grid_position[1], z] is not None:
                num_along_plane += 1
            if self.symmetry_grid[x, self.core_grid_position[1] + y, z] is not None and type(
                self.symmetry_grid[x, self.core_grid_position[1] + y, z]
            ) is type(
                self.symmetry_grid[x, self.core_grid_position[1] - y, z]):
                num_symmetrical += 2
        # Calculate difference
        difference = self.num_modules - num_along_plane
        return num_symmetrical / difference if difference > 0.0 else difference

    def __calculate_yz_symmetry(self) -> float:
        """Goal:
            Calculate the Y/Z-plane symmetry according to the paper but in 3D.
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

        for x, y, z in product(
            range(1, (self.bounding_box_depth - 1) // 2),
            range(self.bounding_box_width),
            range(self.bounding_box_height),):
            if self.symmetry_grid[self.core_grid_position[0], y, z] is not None:
                num_along_plane += 1
            if self.symmetry_grid[
                self.core_grid_position[0] + x, y, z] is not None and type(
                self.symmetry_grid[self.core_grid_position[0] + x, y, z]
            ) is type(
                self.symmetry_grid[self.core_grid_position[0] - x, y, z]):
                num_symmetrical += 2

        # Calculate difference
        difference = self.num_modules - num_along_plane
        return num_symmetrical / difference if difference > 0.0 else difference


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
    def __calculate_is_2d_recur(cls, module: Module) -> bool:
        """Goal:
            Calculate if the robot is two dimensional.
        -------------------------------------------------------
        Input:
            module: The module.
        -------------------------------------------------------
        Output:
            If the robot is two dimensional.
        """
        # return all(
        #     [np.isclose(module.rotation, 0.0)]
        #     + [cls.__calculate_is_2d_recur(child) for child in module.children.values()])
        return True
    

