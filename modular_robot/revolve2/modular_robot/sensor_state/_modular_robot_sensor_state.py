from abc import ABC, abstractmethod

from ..body.base import ActiveHingeSensor, IMUSensor
from . import CameraSensorState
from ._active_hinge_sensor_state import ActiveHingeSensorState
from ._imu_sensor_state import IMUSensorState


class ModularRobotSensorState(ABC):
    """The state of modular robot's sensors."""

    @abstractmethod
    def get_active_hinge_sensor_state(
        self, sensor: ActiveHingeSensor
    ) -> ActiveHingeSensorState:
        """
        Get the state of the provided active hinge sensor.

        :param sensor: The sensor.
        :returns: The state.
        """

    @abstractmethod
    def get_imu_sensor_state(self, sensor: IMUSensor) -> IMUSensorState:
        """
        Get the state of the provided IMU sensor.

        :param sensor: The sensor.
        :returns: The state.
        """

    @abstractmethod
    def get_camera_sensor_state(self) -> CameraSensorState:
        """
        Get the state of the camera sensor.

        :return: The state.
        """
