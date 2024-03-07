"""MorphologicalMeasures class."""
from itertools import product
from typing import Generic, TypeVar

from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
from pyrr import Vector3

from revolve2.modular_robot.body import Module
from revolve2.modular_robot.body.base import ActiveHinge, Body, Brick, Core
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
        
        


    
        self.grid, self.core_grid_position = body.to_grid()

        self.core = body.core

        self.core_is_filled = self.__calculate_core_is_filled()
        self.nfilled_bricks = len(self.__calculate_filled_bricks())
        self.nfilled_active_hinges = len(self.__calculate_filled_active_hinges())
        self.nsingle_neighbour_bricks = len(self.__calculate_single_neighbour_bricks())
        self.nsingle_neighbour_hinges = len(self.__calculate_single_neighbour_hinges())
        #self.ndouble_neighbour_bricks = len(self.__calculate_double_neighbour_bricks())
        self.ndouble_neighbour_active_hinges = len(self.__calculate_double_neighbour_active_hinges())

        self.__pad_grid()
        self.xy_symmetry = self.__calculate_xy_symmetry()
        self.xz_symmetry = self.__calculate_xz_symmetry()
        self.yz_symmetry = self.__calculate_yz_symmetry()

    
    def get_limb(self, module):
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
        #potential_double_neighbour_bricks_and_active_hinges = max(0, self.num_bricks + self.num_active_hinges - 1)
        potential_length_of_limb = max(0, self.num_modules - 1)

        # ---- Get limb length (per core direction)
        limb_length = self.get_limb(self.core)
        # Calculate the proportion
        # if potential_double_neighbour_bricks_and_active_hinges == 0:
        #     return 0.0
        # else:
        #     return (self.ndouble_neighbour_bricks + 
        #             self.ndouble_neighbour_active_hinges) / potential_double_neighbour_bricks_and_active_hinges

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
            print(self.nfilled_active_hinges)
            print(jmax)
            return self.nfilled_active_hinges / jmax
    
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

    def __calculate_core_is_filled(self) -> bool:
        return all(
            [
                self.core.children.get(child_index) is not None
                for child_index in self.core.attachment_points.keys()
            ]
        )

    def __calculate_filled_bricks(self) -> list[Brick]:
        return [
            brick
            for brick in self.bricks
            if all(
                [
                    brick.children.get(child_index) is not None
                    for child_index in brick.attachment_points.keys()
                ]
            )
        ]

    def __calculate_filled_active_hinges(self) -> list[ActiveHinge]:
        return [
            active_hinge
            for active_hinge in self.active_hinges
            if all(
                [
                    active_hinge.children.get(child_index) is not None
                    for child_index in active_hinge.attachment_points.keys()
                ]
            )
        ]

    def __calculate_single_neighbour_bricks(self) -> list[Brick]:
        """Goal:
            Calculate the bricks that are only connected to one other module.
        ----------------------------------------------------------------------------
        Input:
            bricks: 
                The bricks of the robot.
        ----------------------------------------------------------------------------
        Output:
            The bricks that are only connected to one other module."""
        return [
            brick
            for brick in self.bricks
            if all(
                [
                    brick.children.get(child_index) is None
                    for child_index in brick.attachment_points.keys()
                ]
            )
        ]
    
    def __calculate_single_neighbour_hinges(self) -> list[Brick]:
        """Goal:
            Calculate the active hinges that are only connected a single module.
        ----------------------------------------------------------------------------
        Input:
            active_hinges: 
                The active hinges of the robot.
        ----------------------------------------------------------------------------
        Output:
            The active hinges that are only connected to a single module."""
        return [
            hinge
            for hinge in self.active_hinges
            if all(
                [
                    hinge.children.get(child_index) is None
                    for child_index in hinge.attachment_points.keys()
                ]
            )
        ]
    
    def __calculate_double_neighbour_bricks(self) -> list[Brick]:
        return [
            brick
            for brick in self.bricks
            if sum(
                [
                    0 if brick.children.get(child_index) is None else 1
                    for child_index in brick.attachment_points.keys()
                ]
            )
            == 1
        ]

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
        x, y, z = self.grid.shape
        xoffs, yoffs, zoffs = self.core_grid_position
        self.symmetry_grid = np.empty(
            shape=(x + xoffs, y + yoffs, z + zoffs), dtype=Module
        )
        self.symmetry_grid.fill(None)
        self.symmetry_grid[:x, :y, :z] = self.grid

    def __calculate_xy_symmetry(self) -> float:
        num_along_plane = 0
        num_symmetrical = 0
        for x, y, z in product(
            range(self.bounding_box_depth),
            range(self.bounding_box_width),
            range(1, (self.bounding_box_height - 1) // 2),
        ):
            if self.symmetry_grid[x, y, self.core_grid_position[2]] is not None:
                num_along_plane += 1
            if self.symmetry_grid[
                x, y, self.core_grid_position[2] + z
            ] is not None and type(
                self.symmetry_grid[x, y, self.core_grid_position[2] + z]
            ) is type(
                self.symmetry_grid[x, y, self.core_grid_position[2] - z]
            ):
                num_symmetrical += 2

        difference = self.num_modules - num_along_plane
        return num_symmetrical / difference if difference > 0.0 else difference

    def __calculate_xz_symmetry(self) -> float:
        num_along_plane = 0
        num_symmetrical = 0
        for x, y, z in product(
            range(self.bounding_box_depth),
            range(1, (self.bounding_box_width - 1) // 2),
            range(self.bounding_box_height),
        ):
            if self.symmetry_grid[x, self.core_grid_position[1], z] is not None:
                num_along_plane += 1
            if self.symmetry_grid[
                x, self.core_grid_position[1] + y, z
            ] is not None and type(
                self.symmetry_grid[x, self.core_grid_position[1] + y, z]
            ) is type(
                self.symmetry_grid[x, self.core_grid_position[1] - y, z]
            ):
                num_symmetrical += 2
        difference = self.num_modules - num_along_plane
        return num_symmetrical / difference if difference > 0.0 else difference

    def __calculate_yz_symmetry(self) -> float:
        num_along_plane = 0
        num_symmetrical = 0
        for x, y, z in product(
            range(1, (self.bounding_box_depth - 1) // 2),
            range(self.bounding_box_width),
            range(self.bounding_box_height),
        ):
            if self.symmetry_grid[self.core_grid_position[0], y, z] is not None:
                num_along_plane += 1
            if self.symmetry_grid[
                self.core_grid_position[0] + x, y, z
            ] is not None and type(
                self.symmetry_grid[self.core_grid_position[0] + x, y, z]
            ) is type(
                self.symmetry_grid[self.core_grid_position[0] - x, y, z]
            ):
                num_symmetrical += 2
        difference = self.num_modules - num_along_plane
        return num_symmetrical / difference if difference > 0.0 else difference





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

    # @property
    # def num_filled_bricks(self) -> int:
    #     """
    #     Get the number of bricks which have all slots filled with other modules.

    #     :returns: The number of bricks.
    #     """
    #     return len(self.filled_bricks)

    # @property
    # def num_filled_active_hinges(self) -> int:
    #     """
    #     Get the number of bricks which have all slots filled with other modules.

    #     :returns: The number of bricks.
    #     """
    #     return len(self.filled_active_hinges)

    # @property
    # def num_filled_modules(self) -> int:
    #     """
    #     Get the number of modules which have all slots filled with other modules, including the core.

    #     :returns: The number of modules.
    #     """
    #     return (
    #         self.num_filled_bricks
    #         + self.num_active_hinges
    #         + (1 if self.core_is_filled else 0)
    #     )

    # @property
    # def max_potentionally_filled_core_and_bricks(self) -> int:
    #     """
    #     Get the maximum number of core and bricks that could potentially be filled with this set of modules if rearranged in an optimal way.

    #     This calculates 'b_max' from the paper.

    #     :returns: The calculated number.
    #     """
    #     # Snake-like is an optimal arrangement.
    #     #
    #     #   H H H H
    #     #   | | | |
    #     # H-C-B-B-B-H
    #     #   | | | |
    #     #   H H H H
    #     #
    #     # Every extra brick(B) requires 3 modules:
    #     # The bricks itself and two other modules for its sides(here displayed as H).
    #     # However, the core and final brick require three each to fill, which is cheaper than another brick.
    #     #
    #     # Expected sequence:
    #     # | num modules | 1 2 3 4 5 6 7 8 9 10 11 12 14
    #     # | return val  | 0 0 0 0 1 1 1 2 2 2  3  3  3

    #     pot_max_filled = max(0, (self.num_modules - 2) // 3)

    #     # Enough bricks must be available for this strategy.
    #     # We can count the core as the first brick.
    #     pot_max_filled = min(pot_max_filled, 1 + self.num_bricks)

    #     return pot_max_filled

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
            return (self.nfilled_bricks + (1 if self.core_is_filled else 0)) / max_filled

    # @property
    # def num_single_neighbour_bricks(self) -> int:
    #     """
    #     Get the number of bricks that are only connected to one other module.

    #     Both children and parent are counted.

    #     :returns: The number of bricks.
    #     """
    #     return len(self.single_neighbour_bricks)

    # @property
    # def max_potential_single_neighbour_bricks(self) -> int:
    #     """
    #     Get the maximum number of bricks that could potentially have only one neighbour if this set of modules was rearranged in an optimal way.

    #     This calculates "l_max" from the paper.

    #     :returns: The calculated number.
    #     """
    #     # Snake-like is an optimal arrangement.
    #     #
    #     #   B B B B B
    #     #   | | | | |
    #     # B-C-B-B-B-B-B
    #     #   | | | | |
    #     #   B B B B B
    #     #
    #     # Active hinges are irrelevant because they can always be placed in between two modules without affecting this number.
    #     #
    #     # Expected sequence:
    #     # | num bricks | 0 1 2 3 4 5 6 7 8 9
    #     # | return val | 0 1 2 3 4 4 5 6 6 7

    #     return self.num_bricks - max(0, (self.num_bricks - 2) // 3)


    # @property
    # def num_double_neighbour_bricks(self) -> int:
    #     """
    #     Get the number of bricks that are connected to exactly two other modules.

    #     Both children and parent are counted.

    #     :returns: The number of bricks.
    #     """
    #     return len(self.double_neighbour_bricks)

    # @property
    # def num_double_neighbour_active_hinges(self) -> int:
    #     """
    #     Get the number of active hinges that are connected to exactly two other modules.

    #     Both children and parent are counted.

    #     :returns: The number of active hinges.
    #     """
    #     return len(self.double_neighbour_active_hinges)


    # @property
    # def branching(self) -> float:
    #     """
    #     Get the 'branching' measurement from the paper.

    #     Alias for filled_core_and_bricks_proportion.

    #     :returns: Branching measurement.
    #     """
    #     return self.filled_core_and_bricks_proportion


    



    @property
    def coverage(self) -> float:
        """
        Goal:
            Get the 'coverage' measurement from the paper.
                Alias for bounding_box_volume_coverage.
        ----------------------------------------------------
        Input:
            ....
        ----------------------------------------------------
        Output:
            Coverage measurement.
        """
        
        bounding_box_volume = self.bounding_box_width * self.bounding_box_height * self.bounding_box_depth

        return self.num_modules / bounding_box_volume



    @property
    def symmetry(self) -> float:
        """
        Get the 'symmetry' measurement from the paper, but extended to 3d.

        :returns: Symmetry measurement.
        """
        return max(self.xy_symmetry, self.xz_symmetry, self.yz_symmetry)

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
    

