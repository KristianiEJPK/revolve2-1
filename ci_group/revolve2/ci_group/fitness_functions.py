"""Standard fitness functions for modular robots."""

import math
from revolve2.modular_robot_simulation import ModularRobotSimulationState


def xy_displacement(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    """
    Goal:
        Calculate the distance traveled on the xy-plane by a single modular robot.
    -------------------------------------------------------------------------------------------
    Input:
        begin_state: Begin state of the robot.
        end_state: End state of the robot.
    -------------------------------------------------------------------------------------------
    Output:
        The calculated fitness.
    """
    # Begin and end Position
    begin_position = begin_state.get_pose().position
    end_position = end_state.get_pose().position

    # Calculate the distance
    return math.sqrt(
        (begin_position.x - end_position.x) ** 2
        + (begin_position.y - end_position.y) ** 2
    )


def x_speed_Miras2021(begin_state: ModularRobotSimulationState, 
                             end_state: ModularRobotSimulationState, simulation_time = float) -> float:
    """Goal:
        Calculate the fitness for speed in x direction for a single modular robot according to 
            Miras (2021).
    -------------------------------------------------------------------------------------------
    Input:
        begin_state: Begin state of the robot.
        end_state: End state of the robot.
        simulation_time: The time of the simulation.
    -------------------------------------------------------------------------------------------
    Output:
        The calculated fitness.
    """
    # Begin and end Position
    begin_position = begin_state.get_pose().position
    end_position = end_state.get_pose().position

    # Calculate the distance
    vx = float(((end_position.x - begin_position.x) / simulation_time) * 100)
    if vx > 0:
        return vx
    elif vx == 0:
        return -0.1
    else:
        return vx / 10