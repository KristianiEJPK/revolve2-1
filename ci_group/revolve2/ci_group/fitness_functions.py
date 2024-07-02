"""Standard fitness functions for modular robots."""

from copy import deepcopy
import math
import numpy as np
from typing import Tuple
from revolve2.modular_robot_simulation import ModularRobotSimulationState


def xy_displacement(x_distance: float, y_distance: float) -> float:
    """
    Goal:
        Calculate the distance traveled on the xy-plane by a single modular robot.
    -------------------------------------------------------------------------------------------
    Input:
        x_distance: The distance traveled in the x direction.
        y_distance: The distance traveled in the y direction.
    -------------------------------------------------------------------------------------------
    Output:
        The calculated fitness.
    """
    # Calculate the distance
    return math.sqrt((x_distance ** 2) + (y_distance ** 2))


def x_speed_Miras2021(x_distance: float, simulation_time = float) -> float:
    """Goal:
        Calculate the fitness for speed in x direction for a single modular robot according to 
            Miras (2021).
    -------------------------------------------------------------------------------------------
    Input:
        x_distance: The distance traveled in the x direction.
        simulation_time: The time of the simulation.
    -------------------------------------------------------------------------------------------
    Output:
        The calculated fitness.
    """
    # Begin and end Position

    # Calculate the speed in x direction
    vx = float((x_distance / simulation_time) * 100)
    if vx > 0:
        return vx
    elif vx == 0:
        return -0.1
    else:
        return vx / 10
    
def x_efficiency(xbest: float, eexp: float, simulation_time: float) -> float:
    """Goal:
        Calculate the efficiency of a robot for locomotion in x direction.
    -------------------------------------------------------------------------------------------
    Input:
        xbest: The furthest distance traveled in the x direction.
        eexp: The energy expended.
        simulation_time: The time of the simulation.
    -------------------------------------------------------------------------------------------
    Output:
        The calculated fitness.
    """
    def food(xbest, bmet):
        # Get food
        if xbest <= 0:
            return 0
        else:
            food = (xbest / 0.05) * (80 * bmet)
            return food
    
    def scale_EEXP(eexp, bmet):
        return eexp / 346 * (80 * bmet)
    
    # Get baseline metabolism
    bmet = 80
    battery = -bmet * simulation_time + food(xbest, bmet) - scale_EEXP(eexp, bmet)
    
    return battery

def directional_displacement(
    start_state: ModularRobotSimulationState, 
    end_state: ModularRobotSimulationState, 
    desired_direction: Tuple[float, float, float],
    max_expected_displacement: float = 10.0,
    direction_weight: float = 0.8,
    distance_weight: float = 0.2
) -> float:
    # Calculate the displacement vector
    displacement = [
        end_state.get_pose().position.x - start_state.get_pose().position.x,
        end_state.get_pose().position.y - start_state.get_pose().position.y,
        end_state.get_pose().position.z - start_state.get_pose().position.z
    ]

    # Calculate the magnitude of the displacement vector
    displacement_magnitude = math.sqrt(sum(d**2 for d in displacement))

    # If there's no displacement, return 0 fitness
    if displacement_magnitude == 0:
        return 0.0

    # Normalize the desired direction
    desired_norm = math.sqrt(sum(d**2 for d in desired_direction))
    if desired_norm == 0:
        raise ValueError("Desired direction cannot be a zero vector.")
    normalized_direction = tuple(d / desired_norm for d in desired_direction)

    # Calculate the dot product of the displacement vector and the normalized desired direction
    dot_product = sum(d * n for d, n in zip(displacement, normalized_direction))

    # Calculate the directional component of fitness (ranges from -1 to 1)
    directional_fitness = dot_product / displacement_magnitude

    # Adjust directional fitness to range from 0 to 1, with 0.5 being perpendicular movement
    adjusted_directional_fitness = (directional_fitness + 1) / 2

    # Calculate the distance component of fitness
    distance_fitness = min(displacement_magnitude / max_expected_displacement, 1.0)

    # Combine directional and distance components
    # We use a weighted sum to balance direction and distance
    combined_fitness = (direction_weight * adjusted_directional_fitness**2 + 
                        distance_weight * distance_fitness)

    return combined_fitness

def directional_speed(
    start_state: ModularRobotSimulationState, 
    end_state: ModularRobotSimulationState, 
    desired_direction: Tuple[float, float, float],
    simulation_time: float
) -> float:
    """
    Goal:
        Calculate the fitness for speed in a specific direction for a single modular robot.
    -------------------------------------------------------------------------------------------
    Input:
        start_state: The initial state of the robot.
        end_state: The final state of the robot.
        desired_direction: The desired direction of movement as a tuple (x, y, z).
        simulation_time: The time of the simulation.
    -------------------------------------------------------------------------------------------
    Output:
        The calculated fitness.
    """
    # Calculate the displacement vector
    displacement = [
        end_state.get_pose().position.x - start_state.get_pose().position.x,
        end_state.get_pose().position.y - start_state.get_pose().position.y,
        end_state.get_pose().position.z - start_state.get_pose().position.z
    ]

    # Normalize the desired direction
    desired_norm = math.sqrt(sum(d**2 for d in desired_direction))
    if desired_norm == 0:
        raise ValueError("Desired direction cannot be a zero vector.")
    normalized_direction = tuple(d / desired_norm for d in desired_direction)

    # Calculate the dot product of the displacement vector and the normalized desired direction
    dot_product = sum(d * n for d, n in zip(displacement, normalized_direction))

    # Calculate the speed in the desired direction
    speed_in_desired_direction = dot_product / simulation_time

    # Return the speed as the fitness value
    return speed_in_desired_direction * 100  # Scaling factor to match the original function
