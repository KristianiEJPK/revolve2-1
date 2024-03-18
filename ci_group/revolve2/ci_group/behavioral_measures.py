import math
import numpy as np


def to_euler(quat):
    """
    Adapted from 
    https://github.com/the-guild-of-calamitous-intent/squaternion/blob/master/python/squaternion/squaternion.py
    """
    ysqr = quat[2] * quat[2]

    t0 = +2.0 * (quat[0] * quat[1] + quat[2] * quat[3])
    t1 = +1.0 - 2.0 * (quat[1] * quat[1] + ysqr)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (quat[0] * quat[2] - quat[3] * quat[1])
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (quat[0] * quat[3] + quat[1] * quat[2])
    t4 = +1.0 - 2.0 * (ysqr + quat[3] * quat[3])
    Z = math.atan2(t3, t4)

    return (X, Y, Z,)

class BehavioralMeasures():
    
    def __init__(self, states, robot):
        """Goal:
            Initialize this object.
        -------------------------------------------------------------------------------------------
        Input:
            states: A list of states.
            robot: The robot."""
        self.states = states
        self.robot = robot

    def get_measures(self):
        """Goal:
            Get behavioral measures for the robot.
        -------------------------------------------------------------------------------------------"""
        # ---- Initialize
        self.energy_used = 0
        self.roll = 0
        self.pitch = 0
        start_state, end_state = [], []
        history = {"force": [], "energy": [], "efficiency": [], "dx": [], "dy": []}

        # ---- Loop through states
        for istate, state in enumerate(self.states):
            # Get state info
            state_info = state.get_modular_robot_simulation_state(self.robot)
            # Get actuator forces
            forces = state_info.get_actuator_force()
            # Get energy used
            energy_used = sum(abs(forces))
            self.energy_used += energy_used
            # Get positional information
            if istate == 0:
                start_state = [state_info.get_pose().position.x, state_info.get_pose().position.y]
            elif istate == len(self.states) - 1:
                end_state = [state_info.get_pose().position.x, state_info.get_pose().position.y]
            
            # Save delta distances
            history["dx"].append(state_info.get_pose().position.x - start_state[0])
            history["dy"].append(state_info.get_pose().position.y - start_state[1])

            # Save forces, energy and efficiency for step
            if forces.size == 0: pass
            else:
                history["force"].append(forces)
            history["energy"].append(energy_used)
            history["efficiency"].append(history["dx"][-1] / history["energy"][-1] if history["energy"][-1] else 0)

            # Save rotation
            euler = to_euler(state_info._multi_body_system.pose.orientation)
            eulers = [euler[0], euler[1], euler[2]]
            self.roll += abs(eulers[0]) * 180 / math.pi
            self.pitch += abs(eulers[1]) * 180 / math.pi

        # Overall Measures for Distance    
        self.x_distance = end_state[0] - start_state[0]
        self.tot_xdistance = sum([abs(histx) for histx in history["dx"]]) # Accumulated distance
        self.xmax = np.max(np.cumsum(history["dx"])) # Maximum distance reached
        self.y_distance = end_state[1] - start_state[1]
        self.tot_ydistance = sum([abs(histy) for histy in history["dy"]])

        # Overall Measures for Speed x-direction
        self.min_dx = np.min(history["dx"]) # Minimum displacement
        self.dx25 = np.percentile(history["dx"], 25) # 25th percentile of displacement
        self.mean_dx = np.mean(history["dx"])
        self.median_dx = np.median(history["dx"]) # Median displacement
        self.dx75 = np.percentile(history["dx"], 75) # 75th percentile of displacement
        self.max_dx = np.max(history["dx"]) # Maximum displacement
        self.std_dx = np.std(history["dx"]) # Standard deviation of displacement

        # Overall Measures for Speed y-direction
        self.min_dy = np.min(history["dy"])
        self.dy25 = np.percentile(history["dy"], 25)
        self.mean_dy = np.mean(history["dy"])
        self.median_dy = np.median(history["dy"])
        self.dy75 = np.percentile(history["dy"], 75)
        self.max_dy = np.max(history["dy"])
        self.std_dy = np.std(history["dy"])

        # Energy Distribution
        if history["energy"] != []:
            self.energy_used_min = np.min(history["energy"])
            self.energy_used_25 = np.percentile(history["energy"], 25)
            self.energy_used_mean = np.mean(history["energy"])
            self.energy_used_median = np.median(history["energy"])
            self.energy_used_75 = np.percentile(history["energy"], 75)
            self.energy_used_max = np.max(history["energy"])
            self.energy_used_std = np.std(history["energy"])
        else:
            self.energy_used_min = 0
            self.energy_used_25 = 0
            self.energy_used_mean = 0
            self.energy_used_median = 0
            self.energy_used_75 = 0
            self.energy_used_max = 0
            self.energy_used_std = 0

        # Force Distribution
        if history["force"] != []:
            # Per Motor over Time
            force_std_motor = np.std(history["force"], axis = 0) # Across timepoints, i.e. per motor
            self.force_std_motor_min = np.min(force_std_motor)
            self.force_std_motor_25 = np.percentile(force_std_motor, 25)
            self.force_std_motor_mean = np.mean(force_std_motor)
            self.force_std_motor_median = np.median(force_std_motor)
            self.force_std_motor_75 = np.percentile(force_std_motor, 75)
            self.force_std_motor_max = np.max(force_std_motor)
            self.force_std_motor_std = np.std(force_std_motor)

            # Across All Motors per Time
            force_std_all = np.std(history["force"], axis = 1)
            self.force_std_all_min = np.min(force_std_all)
            self.force_std_all_25 = np.percentile(force_std_all, 25)
            self.force_std_all_mean = np.mean(force_std_all)
            self.force_std_all_median = np.median(force_std_all)
            self.force_std_all_75 = np.percentile(force_std_all, 75)
            self.force_std_all_max = np.max(force_std_all)
            self.force_std_all_std = np.std(force_std_all)
        else:
            self.force_std_motor_min = 0
            self.force_std_motor_25 = 0
            self.force_std_motor_mean = 0
            self.force_std_motor_median = 0
            self.force_std_motor_75 = 0
            self.force_std_motor_max = 0
            self.force_std_motor_std = 0
            self.force_std_all_min = 0
            self.force_std_all_25 = 0
            self.force_std_all_mean = 0
            self.force_std_all_median = 0
            self.force_std_all_75 = 0
            self.force_std_all_max = 0
            self.force_std_all_std = 0

        # Efficiency
        if history["energy"] != []:
            self.efficiency = self.x_distance / self.energy_used if self.energy_used > 0 else 0
            self.efficiency_min = np.min(history["efficiency"])
            self.efficiency_25 = np.percentile(history["efficiency"], 25)
            self.efficiency_mean = np.mean(history["efficiency"])
            self.efficiency_median = np.median(history["efficiency"])
            self.efficiency_75 = np.percentile(history["efficiency"], 75)
            self.efficiency_max = np.max(history["efficiency"])
            self.efficiency_std = np.std(history["efficiency"])
        else:
            self.efficiency = 0
            self.efficiency_min = 0
            self.efficiency_25 = 0
            self.efficiency_mean = 0
            self.efficiency_median = 0
            self.efficiency_75 = 0
            self.efficiency_max = 0
            self.efficiency_std = 0
        
        # Balance
        self.balance = (self.roll + self.pitch) / (len(self.states) * 180 * 2)
        self.balance = 1 - self.balance
    
        return self

