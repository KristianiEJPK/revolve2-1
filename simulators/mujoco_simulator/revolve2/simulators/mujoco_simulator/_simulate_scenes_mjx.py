import logging
import math

from functools import partial
from jax import jit
import mujoco
import numpy as np
from dataclasses import dataclass, field
import jax.numpy as jnp
from jax.lax import while_loop
from jax.tree_util import tree_flatten
from mujoco import mjx
from revolve2.simulation.scene import Scene, SimulationState

from ._control_interface_impl import ControlInterfaceImpl
import jax
from ._scene_to_model import scene_to_model
from ._abstraction_to_mujoco_mapping import AbstractionToMujocoMapping
from ._simulation_state_impl_mjx import SimulationStateImplMJX

def _simulation_loop(
        data: mjx.Data,
        scene: Scene,
        model: mjx.Model,
        mapping: AbstractionToMujocoMapping,
        control_interface: ControlInterfaceImpl,
        simulation_states: list[SimulationStateImplMJX],
        simulation_time: int,
        control_step: float | None,
        sample_step: float | None,

) -> None:
    last_control_time = 0.0
    last_sample_time = 0.0
    while (time := data.time) < simulation_time:
        # do control if it is time
        if time >= last_control_time + control_step:
            last_control_time = math.floor(time / control_step) * control_step

            simulation_state = SimulationStateImplMJX(
                data=data, abstraction_to_mujoco_mapping=mapping
            )
            scene.handler.handle(simulation_state, control_interface, control_step)

        # sample state if it is time
        if sample_step is not None:
            if time >= last_sample_time + sample_step:
                last_sample_time = int(time / sample_step) * sample_step
                simulation_states.append(
                    SimulationStateImplMJX(
                        data=data, abstraction_to_mujoco_mapping=mapping
                    )
                )

        # step simulation
        mjx.step(model, data)


def simulate_scenes_mjx(
        scenes: list[Scene],
        control_step: float,
        sample_step: float | None,
        simulation_time: int,
        simulation_timestep: float,
        cast_shadows: bool,
        fast_sim: bool,
) -> list[list[SimulationState]]:
    """
    Simulate scenes in parallel.

    :param scenes: The scenes to simulate.
    :param control_step: The time between each call to the handle function of the scene handler. In seconds.
    :param sample_step: The time between each state sample of the simulation. In seconds.
    :param simulation_time: How long to simulate for. In seconds.
    :param simulation_timestep: The duration to integrate over during each step of the simulation. In seconds.
    :param cast_shadows: If shadows are cast.
    :param fast_sim: If fancy rendering is disabled.
    :returns: The results of simulations. The number of returned states depends on `sample_step`.
    """
    logging.info(f"Simulating scenes")

    models, mappings = [], []

    for scene in scenes:
        model, mapping = scene_to_model(
            scene, simulation_timestep, cast_shadows=cast_shadows, fast_sim=fast_sim
        )
        model = mjx.put_model(model)
        models.append(model)
        mappings.append(mapping)

    models = jax.tree_map(lambda *x: jnp.stack(x), *models)
    data = jax.vmap(lambda model: mjx.make_data(model))(models)

    # The measured states of the simulation
    simulation_states: list[list[SimulationState]] = []

    # Compute forward dynamics without actually stepping forward in time.
    # This updates the data so we can read out the initial state.
    jax.pmap(mjx.forward)(models, data)

    # Sample initial state.
    if sample_step is not None:
        simulation_states.append(
            [
                SimulationStateImplMJX(data=dta, abstraction_to_mujoco_mapping=mapping) for dta, mapping in
                zip(data, mappings)
            ]
        )

    control_interfaces = [ControlInterfaceImpl(data=dta, abstraction_to_mujoco_mapping=mapping) for dta, mapping in
                          zip(data, mappings)]

    jax.vmap(_simulation_loop, in_axes=(0, 0, 0, 0, 0, 0, None, None, None))(
        data,
        scenes,
        models,
        mappings,
        control_interfaces,
        simulation_states,
        simulation_time,
        control_step,
        sample_step,
    )

    # Sample one final time.
    if sample_step is not None:
        simulation_states.append(
            [
                SimulationStateImplMJX(data=dta, abstraction_to_mujoco_mapping=mapping) for dta, mapping in
                zip(data, mappings)
            ]
        )

    logging.info(f"Scenes done.")

    return simulation_states
