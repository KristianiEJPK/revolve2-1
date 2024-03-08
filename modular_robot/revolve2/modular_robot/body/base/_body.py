import math
from typing import Generic, Type, TypeVar

from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
from pyrr import Quaternion, Vector3

from .._module import Module
from ._core import Core


TModule = TypeVar("TModule", bound=Module)
TModuleNP = TypeVar("TModuleNP", bound=np.generic)


class Body:
    """Body of a modular robot."""

    _core: Core

    def __init__(self, core: Core) -> None:
        """
        Initialize this object.

        :param core: The core of the body.
        """
        self._core = core

    @classmethod
    def grid_position(cls, module: Module) -> Vector3:
        """
        Goal:
            Calculate the position of this module in a 3d grid with the core as center.
            The distance between all modules is assumed to be one grid cell.
            All module angles must be multiples of 90 degrees.
            Note: raises KeyError in case an attachment point is not found.
        -------------------------------------------------------------------------------------------
        Input:
            module: The module to calculate the position.
        -------------------------------------------------------------------------------------------
        Output:
            The calculated position.
        """
        # Initialize the position.
        position = Vector3()

        # Calculate the position.
        parent = module.parent
        child_index = module.parent_child_index
        while parent is not None and child_index is not None:
            # Get child and check conditions.
            child = parent.children.get(child_index)
            assert child is not None
            assert np.isclose(child.rotation % (math.pi / 2.0), 0.0)

            # Calculate the position.
            position = Quaternion.from_eulers((child.rotation, 0.0, 0.0)) * position
            position += Vector3([1, 0, 0])

            # Get the attachment point.
            attachment_point = parent.attachment_points.get(child_index)
            if attachment_point is None:
                raise KeyError("No attachment point found at the specified location.")
            # Adapt the position.
            position = attachment_point.orientation * position
            position = Vector3.round(position)
            # ????
            child_index = parent.parent_child_index
            parent = parent.parent
        return position

    @classmethod
    def __find_recur(cls, module: Module, module_type: Type[TModule]) -> list[TModule]:
        modules = []
        if isinstance(module, module_type):
            modules.append(module)
        for child in module.children.values():
            modules.extend(cls.__find_recur(child, module_type))
        return modules

    def find_modules_of_type(self, module_type: Type[TModule]) -> list[TModule]:
        """
        Find all Modules of a certain type in the robot.

        :param module_type: The type.
        :return: The list of Modules.
        """
        return self.__find_recur(self._core, module_type)

    def to_grid(self) -> tuple[NDArray[TModuleNP], Vector3[np.int_]]:
        """
        Convert the tree structure to a grid.

        The distance between all modules is assumed to be one grid cell.
        All module angles must be multiples of 90 degrees.

        The grid is indexed depth, width, height, or x, y, z, from the perspective of the core.

        :returns: The created grid with cells set to either a Module or None and a position vector of the core. The position Vector3 is dtype: int.
        """
        return _GridMaker().make_grid(self)

    @property
    def core(self) -> Core:
        """
        Get the core of the Body.

        :return: The core.
        """
        return self._core


class _GridMaker(Generic[TModuleNP]):
    _x: list[int] = []
    _y: list[int] = []
    _z: list[int] = []
    _modules: list[Module] = []

    def make_grid(self, body: Body) -> tuple[NDArray[TModuleNP], Vector3[np.int_]]:

        for child_index, _ in body._core.attachment_points.items():
            print("************************************************")
            # Get child
            child = body._core.children.get(child_index)
            # Get grid
            self._make_grid_recur(child, Vector3([0, 0, 0]), 
                                Vector3([1, 0, 0]), Vector3([0, 0, 1]), )

        minx, maxx = min(self._x), max(self._x)
        miny, maxy = min(self._y), max(self._y)
        minz, maxz = min(self._z), max(self._z)

        depth = maxx - minx + 1
        width = maxy - miny + 1
        height = maxz - minz + 1

        grid = np.empty(shape=(depth, width, height), dtype=Module)
        grid.fill(None)
        for x, y, z, module in zip(self._x, self._y, self._z, self._modules):
            grid[x - minx, y - miny, z - minz] = module

        return grid, Vector3([-minx, -miny, -minz])

    def _make_grid_recur(
        self, module: Module, position: Vector3, forward: Vector3, up: Vector3
    ) -> None:
        
        # Do not add the attachment points of the core.
        self._add(position, module)
        print("Module: ", module)
        print("Position: ", position)
        print("---------------------------")

        if position == Vector3([0, 0, 0]):
            # ----- Get...
            minsmaxs = []
            # Offset of attachment points
            att_arr = [] 
            for att in module.attachment_points.items():
                transf_off = rotatevectors(att[1].offset, up, att[1].orientation)
                att_arr.append(transf_off)
            att_arr = np.array(att_arr)
            # Min and max values of the attachment points
            minsmaxs.append(att_arr.min(axis = 0))
            minsmaxs.append(att_arr.max(axis = 0))

            # # Fill core values
            # for att in module.attachment_points.items():
            #     forward4face = rotatevectors(forward, up, att[1].orientation)
            #     self._add(position + forward4face, module)

        # Rotate forward
        forwardrot = rotatevectors(forward, up, module.attachment_points[0].orientation)

        # Go on
        for child_index, attachment_point in module.attachment_points.items():
            # Get child
            child = module.children.get(child_index)
            if position == Vector3([0, 0, 0]):
                # Get relative location of slot within face
                middle = np.mean(minsmaxs, axis = 0)
                divider = (minsmaxs[1] - middle) # maximum slot location - middle, both are transformed already
                divider[divider == 0] = 1 # To prevent division by zero
                offset_pos = rotatevectors(attachment_point.offset, up, attachment_point.orientation) # Transform offset
                rellocslot_raw = (offset_pos - middle)
                rellocslot = (rellocslot_raw / divider) # to -1, 0, 1
                # Add 1 additional for forward position --> 3 x 3 x 3 core instead of 1 x 1 x 1
                rellocslot = forwardrot + rellocslot
                
                # Set those points
                self._add(position + rellocslot, module)
            else: 
                rellocslot = Vector3([0, 0, 0])
            
            if child is not None:
                assert np.isclose(child.rotation % (math.pi / 2.0), 0.0)
                # rotation = (
                #     orientation
                #     * attachment_point.orientation
                #     * Quaternion.from_eulers([child.rotation, 0, 0])
                # )
                forward_new = rotatevectors(forward, up, attachment_point.orientation)
                position_new = position + forward_new + rellocslot
                up_new = rotatevectors(up, forward_new, Quaternion.from_eulers([child.rotation, 0, 0]))
                
                self._make_grid_recur(child, position_new, forward_new, up_new)

                

    def _add(self, position: Vector3, module: Module) -> None:
        self._modules.append(module)
        x, y, z = position
        self._x.append(int(round(x)))
        self._y.append(int(round(y)))
        self._z.append(int(round(z)))


def rotatevectors(a: Vector3, b: Vector3, rotation: Quaternion) -> Vector3:
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