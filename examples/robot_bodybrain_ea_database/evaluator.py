"""Evaluator class."""
from revolve2.ci_group import fitness_functions, terrains
from revolve2.ci_group.morphological_measures import MorphologicalMeasures
from revolve2.ci_group.behavioral_measures import BehavioralMeasures
from revolve2.simulation.simulator import BatchParameters
from revolve2.simulation.simulator import RecordSettings
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
        control_frequency: float, record: bool = False,
        writefiles: bool = "False",
        video_path: str = None
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
        # Convert the headless parameter to a boolean.
        if type(headless) is not bool:
            if headless == "True":
                headless = True
            elif headless == "False":
                headless = False
            else:
                raise ValueError("headless must be either True or False")
        # Initialize the simulator.
        self._simulator = LocalSimulator(
            headless = headless, num_simulators=num_simulators
        )

        # ---- Set the simulation parameters.
        self.simulation_time = simulation_time
        self.sampling_frequency = sampling_frequency
        self.simulation_timestep = simulation_timestep
        self.control_frequency = control_frequency
        self.record = record
        self.writefiles = writefiles
        self.video_path = video_path

        # ---- Set the terrain.
        if terrain == "flat":
            self._terrain = terrains.flat()
        elif terrain == "tilted":
            self._terrain = terrains.tilted_flat(z = 0.1)
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
            control_frequency = self.control_frequency)
        
        # ---- Create record settings.
        if (self.record == "True") or (self.writefiles == "True"):
            record_settings = RecordSettings(video_directory = self.video_path)
        else: pass
        # ---- Create the scenes.
        scenes = []
        for robot in robots:
            scene = ModularRobotScene(terrain = self._terrain)
            scene.add_robot(robot)
            scenes.append(scene)
        # ---- Simulate all scenes.
        if self.record != "True":
            scene_states = simulate_scenes(
                simulator=self._simulator,
                batch_parameters = batch_params,
                scenes=scenes
            )
        else:
            scene_states = simulate_scenes(
                simulator=self._simulator,
                batch_parameters = batch_params,
                scenes=scenes, record_settings=record_settings
            )


        # ---- Get Morphological Measures
        behavioral_measures, ids = [], []
        for irobot, robot in enumerate(robots):
            # ---- Morphological measures
            #morphological_measures = MorphologicalMeasures(robot.body, robot.brain, int(os.environ["MAXPARTS"]))
            ids.append(robot.brain.id_string)
            # ---- Behavioral Measures
            # States
            behave = BehavioralMeasures(scene_states[irobot], robot).get_measures()
            behavioral_measures.append({})
            for variable, valuevar in vars(behave).items():
                if variable not in ["states", "robot"]:
                    behavioral_measures[-1][variable] = valuevar
              
        # ---- Calculate the fitnesses.
        if self.fitness_function == "xy_displacement":
            fitnesses = [
                fitness_functions.xy_displacement(
                    behavioral_measure["x_distance"], behavioral_measure["y_distance"])
                for behavioral_measure in behavioral_measures
            ]
        elif self.fitness_function == "x_speed_Miras2021":
            fitnesses = [
                fitness_functions.x_speed_Miras2021(
                    behavioral_measure["x_distance"], simulation_time = self.simulation_time
                )
                for behavioral_measure in behavioral_measures]
        elif self.fitness_function == "x_efficiency":
            fitnesses = [
                fitness_functions.x_efficiency(
                    xmax = behavioral_measure["xmax"], eexp = behavioral_measure["energy_used"], simulation_time = self.simulation_time
                )
                for behavioral_measure in behavioral_measures
            ]
        
        return fitnesses, behavioral_measures, ids
