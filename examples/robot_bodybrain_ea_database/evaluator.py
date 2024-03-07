"""Evaluator class."""
import config
from revolve2.ci_group import fitness_functions, terrains
from revolve2.ci_group.morphological_measures import MorphologicalMeasures
from revolve2.simulation.simulator import BatchParameters
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulators.mujoco_simulator import LocalSimulator



class Evaluator:
    """Goal:
        Provides evaluation of robots."""

    _simulator: LocalSimulator
    _terrain: Terrain

    def __init__(
        self,
        headless: bool,
        num_simulators: int,
        terrain: str,
        fitness_function: str,
        simulation_time: int,
        sampling_frequency: float,
        simulation_timestep: float,
        control_frequency: float,
    ) -> None:
        """
        Goal:
            Initialize this object.
        -------------------------------------------------------------------------------------------
        Input:
            headless: `headless` parameter for the physics simulator.
            num_simulators: `num_simulators` parameter for the physics simulator.
        """
        # ---- Set the simulator.
        self._simulator = LocalSimulator(
            headless = headless, num_simulators=num_simulators
        )

        # ---- Set the simulation parameters.
        self.simulation_time = simulation_time
        self.sampling_frequency = sampling_frequency
        self.simulation_timestep = simulation_timestep
        self.control_frequency = control_frequency

        # ---- Set the terrain.
        if terrain == "flat":
            self._terrain = terrains.flat()
        else:
            raise ValueError(f"Unknown terrain: {terrain}")
        
        # ---- Set the fitness function.
        if fitness_function in ["xy_displacement", "x_speed_Miras2021", "x_efficiency"]:
            self.fitness_function = fitness_function
        else:
            raise ValueError(f"Unknown fitness function: {fitness_function}")
        

    def evaluate(
        self,
        robots: list[ModularRobot],
    ) -> list[float]:
        """
        Goal:
            Evaluate multiple robots. Fitness is the distance traveled on the xy plane.
        -------------------------------------------------------------------------------------------
        Input:
            robots: The robots to simulate.
        -------------------------------------------------------------------------------------------
        Output:
            The fitnesses of the robots.
        """
        # ---- Create batch parameters.
        batch_params = BatchParameters(
            simulation_time = self.simulation_time,
            sampling_frequency = self.sampling_frequency,
            simulation_timestep = self.simulation_timestep,
            control_frequency = self.control_frequency,)
        
        # ---- Create the scenes.
        scenes = []
        for robot in robots:
            scene = ModularRobotScene(terrain = self._terrain)
            scene.add_robot(robot)
            scenes.append(scene)

        # ---- Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters = batch_params,
            scenes=scenes,
        )


        # ---- Get Morphological Measures
        for robot in robots:
            morphological_measures = MorphologicalMeasures(robot.body, config.MAX_PARTS)
            print("Size: ", morphological_measures.size)
            print("Proportion: ", morphological_measures.proportion_2d)
            print("Limbs: ", morphological_measures.limbs)

            maxrel_length, meanrel_length, stdrel_length = morphological_measures.length_of_limbs
            print("Length of Limbs (maxrel): ", maxrel_length)
            print("Length of Limbs (meanrel): ", meanrel_length)
            print("Length of Limbs (stdrel): ", stdrel_length)

            print("Joints: ", morphological_measures.joints) # Deze klopt nog niet ---> attached to brick or core!!!!!!!!!
            print("Joint-Brick Ratio: ", morphological_measures.joint_brick_ratio)
            
            
            # Misschien ook nog een joint/brick ratio?

            print("Symmetry: ", morphological_measures.symmetry)
            print("Coverage: ", morphological_measures.coverage)
            print("Branching: ", morphological_measures.branching)

            print("Number of Modules: ", morphological_measures.num_modules)
            print("Number of Active Hinges: ", morphological_measures.num_active_hinges)
            print("Number of Bricks: ", morphological_measures.num_bricks)
            
            
            
            
            
            
            
            
            


        # ---- Calculate the fitnesses.
        if self.fitness_function == "xy_displacement":
            fitnesses = [
                fitness_functions.xy_displacement(
                    states[0].get_modular_robot_simulation_state(robot),
                    states[-1].get_modular_robot_simulation_state(robot),
                )
                for robot, states in zip(robots, scene_states)
            ]
        elif self.fitness_function == "x_speed_Miras2021":
            fitnesses = [
                fitness_functions.x_speed_Miras2021(
                    states[0].get_modular_robot_simulation_state(robot),
                    states[-1].get_modular_robot_simulation_state(robot), simulation_time = self.simulation_time
                )
                for robot, states in zip(robots, scene_states)
            ]
        elif self.fitness_function == "x_efficiency":
            fitnesses = [
                fitness_functions.x_efficiency(
                    states = states, robot = robot, simulation_time = self.simulation_time
                )
                for robot, states in zip(robots, scene_states)
            ]


        return fitnesses
