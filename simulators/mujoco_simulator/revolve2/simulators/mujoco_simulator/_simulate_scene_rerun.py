import logging
import math

import cv2
import mujoco
import numpy as np
import numpy.typing as npt

from revolve2.simulation.scene import Scene, SimulationState
from revolve2.simulation.simulator import RecordSettings

from ._control_interface_impl import ControlInterfaceImpl
from ._custom_mujoco_viewer import CustomMujocoViewer
from ._scene_to_model import scene_to_model
from ._simulation_state_impl import SimulationStateImpl

import pickle


def write_files(model, data, time):
    filename = f"RERUN\XMLs\mujoco_{time}.xml"
    with open(filename, "w") as mjcf_file:
        mujoco.mj_saveLastXML(mjcf_file.name, model)
    filename = f"RERUN\PKLs\mujoco_{time}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def simulate_scene(
    scene_id: int,
    scene: Scene,
    headless: bool,
    record_settings: RecordSettings | None,
    start_paused: bool,
    control_step: float,
    sample_step: float | None,
    simulation_time: int | None,
    simulation_timestep: float,
    cast_shadows: bool,
    fast_sim: bool,
) -> list[SimulationState]:
    """
    Simulate a scene.

    :param scene_id: An id for this scene, unique between all scenes ran in parallel.
    :param scene: The scene to simulate.
    :param headless: If False, a viewer will be opened that allows a user to manually view and manually interact with the simulation.
    :param record_settings: If not None, recording will be done according to these settings.
    :param start_paused: If true, the simulation will start in a paused state. Only makessense when headless is False.
    :param control_step: The time between each call to the handle function of the scene handler. In seconds.
    :param sample_step: The time between each state sample of the simulation. In seconds.
    :param simulation_time: How long to simulate for. In seconds.
    :param simulation_timestep: The duration to integrate over during each step of the simulation. In seconds.
    :param cast_shadows: If shadows are cast.
    :param fast_sim: If fancy rendering is disabled.
    :returns: The results of simulation. The number of returned states depends on `sample_step`.
    """
    logging.info(f"Simulating scene {scene_id}")

    # Set model and data
    model, mapping = scene_to_model(
        scene, simulation_timestep, cast_shadows=cast_shadows, fast_sim=fast_sim
    )
    data = mujoco.MjData(model)

    # Set fps and video step
    videostep = 1 / record_settings.fps

    # Initialize variables
    last_control_time = 0.0
    last_sample_time = 0.0
    last_video_time = 0.0  # time at which last video frame was saved

    # The measured states of the simulation
    simulation_states: list[SimulationState] = []

    # Compute forward dynamics without actually stepping forward in time.
    # This updates the data so we can read out the initial state.
    mujoco.mj_forward(model, data)

    # Sample initial state.
    if sample_step is not None:
        simulation_states.append(
            SimulationStateImpl(data=data, abstraction_to_mujoco_mapping=mapping)
        )

    control_interface = ControlInterfaceImpl(
        data=data, abstraction_to_mujoco_mapping=mapping
    )
    
    # Sample initial state
    write_files(model, data, 0)


    # Run simulation
    while (time := data.time) < (
        float("inf") if simulation_time is None else simulation_time
    ):
        # do control if it is time
        if time >= last_control_time + control_step:
            last_control_time = math.floor(time / control_step) * control_step

            simulation_state = SimulationStateImpl(
                data=data, abstraction_to_mujoco_mapping=mapping
            )
            scene.handler.handle(simulation_state, control_interface, control_step)

        # sample state if it is time
        if sample_step is not None:
            if time >= last_sample_time + sample_step:
                last_sample_time = int(time / sample_step) * sample_step
                simulation_states.append(
                    SimulationStateImpl(
                        data=data, abstraction_to_mujoco_mapping=mapping
                    )
                )

        # step simulation
        mujoco.mj_step(model, data)


        # Write files
        if time >= last_video_time + videostep:
            write_files(model, data, time)
            last_video_time = int(time / videostep) * videostep


    # Sample one final time.
    if sample_step is not None:
        simulation_states.append(
            SimulationStateImpl(data=data, abstraction_to_mujoco_mapping=mapping)
        )

    logging.info(f"Scene {scene_id} done.")

    return simulation_states
