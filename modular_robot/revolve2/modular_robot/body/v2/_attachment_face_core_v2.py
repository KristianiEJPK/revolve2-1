from itertools import product

import numpy as np
from numpy.typing import NDArray
from pyrr import Quaternion, Vector3

from .. import Module
from .._attachment_point import AttachmentPoint
from ..base._attachment_face import AttachmentFace


class AttachmentFaceCoreV2(AttachmentFace):
    """An AttachmentFace for the V2 Core."""

    _check_matrix: NDArray[np.uint8]
    _child_offset: Vector3

    """
    Check matrix allows us to determine which attachment points can be filled in the face.
    
    check_matrix =  0   0   0
                      C   C  
                    0   0   0
                      C   C  
                    0   0   0
     
    By default the whole matrix is 0. Once we add a module at location x we adjust the C accordingly. 
    When adding a new module we want to have a C of 0 in the corresponding position, otherwise the attachment point cant be populated anymore.
    Applying a simple 2D convolution allows for fast conflict checks.
    """

    def __init__(
        self, face_rotation: float, horizontal_offset: float, vertical_offset: float
    ) -> None:
        """
        Initialize the attachment face for the V2 Core.

        :param face_rotation: The rotation of the face and the attachment points on the module.
        :param horizontal_offset: The horizontal offset for module placement.
        :param vertical_offset:  The vertical offset for module placement.
        """
        self._child_offset = Vector3([0.15 / 2.0, 0.0, 0.0])
        self._check_matrix = np.zeros(shape=(3, 3), dtype=np.uint8)

        """
        Each CoreV2 Face has 9 Module slots as shown below. 
        
        ---------------------------------------------------
        |                 |            |                  |
        | 0 (TOP_LEFT)    | 1 (TOP)    | 2 (TOP_RIGHT)    |
        |                 |            |                  |
        | 3 (MIDDLE_LEFT) | 4 (MIDDLE) | 5 (MIDDLE_RIGHT) |
        |                 |            |                  |
        | 6 (BOTTOM_LEFT) | 7 (BOTTOM) | 8 (BOTTOM_RIGHT) |
        |                 |            |                  |
        ---------------------------------------------------
        """
        # Initialize attachment points
        attachment_points = {}
        rot = Quaternion.from_eulers([0.0, 0.0, face_rotation])
        
        # For all 9 attachment points
        for i in range(9):
            # Horizontal offset
            h_o = (i % 3 - 1) * horizontal_offset
            # Vertical offset
            v_o = -(i // 3 - 1) * vertical_offset
            # Adapt horizontal offset
            h_o = h_o if int(rot.angle / np.pi) % 2 == 0 else -h_o
            # Set offset
            offset = (
                Vector3([0.0, h_o, v_o])
                if np.isclose(rot.angle % np.pi, 0)
                else Vector3([h_o, 0.0, v_o])
            )
            offset = rot * offset
            # Create attachment point
            attachment_points[i] = AttachmentPoint(
                orientation=rot, offset=self._child_offset + offset
            )
        super().__init__(rotation=0.0, attachment_points=attachment_points)

    def can_set_child(
        self,
        module: Module,
        child_index: int, flag: str = "condition"
    ) -> bool:
        """
        Check for conflicts when adding a new attachment point.

        Note that if there is no conflict in the check this function assumes that the slot is being populated and adjusts the check-matrix as such.

        :param module: The module.
        :param child_index: The index of the attachment point.
        :flag: The flag to determine if the function should perform the action ("perform") or just check for the condition ("condition").
        :return: Whether conflicts occurred.
        """
        # Copy the check matrix
        check_matrix = self._check_matrix.copy()
        # Turn 0 into 1
        row = int(np.floor(child_index / 3))
        col = int(child_index - (row * 3))
        check_matrix[row, col] += 1
        
        conv_check = np.zeros(shape=(2, 2), dtype=np.uint8)
        for i, j in product(range(2), repeat=2):
            conv_check[i, j] = np.sum(check_matrix[i : (i + 2), j : (j + 2)])

        if flag == "condition":
            if child_index == 0:
                slots2close = [1, 3, 4]
            elif child_index == 1:
                slots2close = [0, 2, 3, 4, 5]
            elif child_index == 2:
                slots2close = [1, 4, 5]
            elif child_index == 3:
                slots2close = [0, 1, 4, 6, 7]
            elif child_index == 4:
                slots2close = [0, 1, 2, 3, 5, 6, 7, 8]
            elif child_index == 5:
                slots2close = [1, 2, 4, 7, 8]
            elif child_index == 6:
                slots2close = [3, 4, 7]
            elif child_index == 7:
                slots2close = [3, 4, 5, 6, 8]
            elif child_index == 8:
                slots2close = [4, 5, 7]
            
            if np.max(conv_check) > 1:  # Conflict detected.
                return False, slots2close
            else: return True, slots2close
        elif flag == "perform":
            self._check_matrix = check_matrix
            if np.max(conv_check) > 1:  # Conflict detected.
                return False
            else:
                return True
        else: raise ValueError("Incorrect flag")

    @property
    def top_left(self) -> Module | None:
        """
        Get the top_left attachment points module.

        :return: The attachment points module.
        """
        return self.children.get(0)

    @top_left.setter
    def top_left(self, module: Module) -> None:
        """
        Set a module to the top_left attachment point.

        :param module: The module.
        """
        self.set_child(module, 0)

    @property
    def top(self) -> Module | None:
        """
        Get the top attachment points module.

        :return: The attachment points module.
        """
        return self.children.get(1)

    @top.setter
    def top(self, module: Module) -> None:
        """
        Set a module to the top attachment point.

        :param module: The module.
        """
        self.set_child(module, 1)

    @property
    def top_right(self) -> Module | None:
        """
        Get the top_right attachment points module.

        :return: The attachment points module.
        """
        return self.children.get(2)

    @top_right.setter
    def top_right(self, module: Module) -> None:
        """
        Set a module to the top_right attachment point.

        :param module: The module.
        """
        self.set_child(module, 2)

    @property
    def middle_left(self) -> Module | None:
        """
        Get the middle_left attachment points module.

        :return: The attachment points module.
        """
        return self.children.get(3)

    @middle_left.setter
    def middle_left(self, module: Module) -> None:
        """
        Set a module to the middle_left attachment point.

        :param module: The module.
        """
        self.set_child(module, 3)

    @property
    def middle(self) -> Module | None:
        """
        Get the middle attachment points module.

        :return: The attachment points module.
        """
        return self.children.get(4)

    @middle.setter
    def middle(self, module: Module) -> None:
        """
        Set a module to the middle attachment point.

        :param module: The module.
        """
        self.set_child(module, 4)

    @property
    def middle_right(self) -> Module | None:
        """
        Get the middle_right attachment points module.

        :return: The attachment points module.
        """
        return self.children.get(5)

    @middle_right.setter
    def middle_right(self, module: Module) -> None:
        """
        Set a module to the middle_right attachment point.

        :param module: The module.
        """
        self.set_child(module, 5)

    @property
    def bottom_left(self) -> Module | None:
        """
        Get the bottom_left attachment points module.

        :return: The attachment points module.
        """
        return self.children.get(6)

    @bottom_left.setter
    def bottom_left(self, module: Module) -> None:
        """
        Set a module to the bottom_left attachment point.

        :param module: The module.
        """
        self.set_child(module, 6)

    @property
    def bottom(self) -> Module | None:
        """
        Get the bottom attachment points module.

        :return: The attachment points module.
        """
        return self.children.get(7)

    @bottom.setter
    def bottom(self, module: Module) -> None:
        """
        Set a module to the bottom attachment point.

        :param module: The module.
        """
        self.set_child(module, 7)

    @property
    def bottom_right(self) -> Module | None:
        """
        Get the bottom_right attachment points module.

        :return: The attachment points module.
        """
        return self.children.get(8)

    @bottom_right.setter
    def bottom_right(self, module: Module) -> None:
        """
        Set a module to the bottom_right attachment point.

        :param module: The module.
        """
        self.set_child(module, 8)
