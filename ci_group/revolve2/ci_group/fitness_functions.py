"""Standard fitness functions for modular robots."""

from copy import deepcopy
import math
import numpy as np
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

    # Calculate the speed in x direction
    vx = float(((end_position.x - begin_position.x) / simulation_time) * 100)
    if vx > 0:
        return vx
    elif vx == 0:
        return -0.1
    else:
        return vx / 10
    
def x_efficiency(states, robot, simulation_time):
    """Goal:
        Calculate the efficiency of a robot for locomotion in x direction.
    -------------------------------------------------------------------------------------------
    Input:
        states: The states of the robot.
        robot: The robot.
        simulation_time: The time of the simulation.
    -------------------------------------------------------------------------------------------
    Output:
        The calculated efficiency.
    """

    # Get state information
    for istate, state in enumerate(states):
        states[istate] = state.get_modular_robot_simulation_state(robot)

    # Begin and end Position
    begin_position = states[0].get_pose().position
    end_position = states[-1].get_pose().position

    # Calculate distance in x direction
    x = (end_position.x - begin_position.x)

    # Calculate the energy expenditure --> 
    # assumed that the EE is the sum of the absolute values of the forces
    def food_increase(x, xbest, bmet):
        # Get food
        food = (x / 0.05) * (80 * bmet)
        new_food = food - xbest[1]
        # Update best
        xbest = [x, food]
        return xbest, new_food
    
    # Get baseline metabolism
    bmet = 1
    battery = bmet * simulation_time * 81
    
    x, best = 0, [0, 0]
    for state in states:
        # Position
        x += (state.get_pose().position.x - x)
        # Energy expenditure due to baseline metabolism
        battery -= bmet
        # Energy expenditure due to work
        forces = state.get_actuator_force()
        eexp = (sum(abs(forces)) / 345.62678648916483) * 80 * bmet
        battery -= eexp
        # Food
        if (x > best[0]) and (x > 0):
            best, food = food_increase(x, best, bmet)
            battery += food
    #print(battery)
    return battery   
    # # Calculate the efficiency: external/internal work
    # try:
    #     efficiency = (abs(x) + 0.001) / eexp
    # except ZeroDivisionError:
    #     efficiency = 0
    # efficiency *= np.sign(x)

    # return efficiency
