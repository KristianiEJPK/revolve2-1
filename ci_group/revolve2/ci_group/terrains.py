"""Standard terrains."""

import math
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from noise import pnoise2
from pyrr import Quaternion, Vector3

from revolve2.modular_robot_simulation import Terrain
from revolve2.simulation.scene import Pose
from revolve2.simulation.scene import Pose
from revolve2.simulation.scene.geometry import GeometryHeightmap, GeometryPlane
from revolve2.simulation.scene.vector2 import Vector2

@dataclass
class Size:
    x: float
    y: float
    z: float

def flat(size: Vector2 = Vector2([20.0, 20.0])) -> Terrain:
    """
    Create a flat plane terrain.

    :param size: Size of the plane.
    :returns: The created terrain.
    """
    return Terrain(
        static_geometry=[
            GeometryPlane(
                pose=Pose(),
                mass=0.0,
                size=size,
            )
        ]
    )

def tilted_flat(size: Vector2 = Vector2([20.0, 20.0]), z: float = 0) -> Terrain:
    """
    Create a flat plane terrain.

    :param size: Size of the plane.
    :returns: The created terrain.
    """
    return Terrain(
        static_geometry=[
            GeometryPlane(
                pose=Pose(orientation=Quaternion.from_eulers([0, 0, math.pi / 90 * 10])),
                mass=0.0,
                size=size,
            )
        ]
    )


class WaterTerrain:
    def __init__(self, grid_size=(50, 50), wave_speed=1.0, damping=0.99, time_step=0.01, fluid_density=1000):
        self.grid_size = grid_size
        self.wave_speed = wave_speed
        self.damping = damping
        self.time_step = time_step
        self.fluid_density = fluid_density

        self.water_depth = np.zeros(grid_size)
        self.velocity = np.zeros(grid_size)
        self.wave_center = (grid_size[0] // 2, grid_size[1] // 2)
        self.water_depth[self.wave_center] = 1

    def update_water(self):
        new_depth = self.water_depth.copy()
        new_velocity = self.velocity.copy()

        for i in range(1, self.water_depth.shape[0] - 1):
            for j in range(1, self.water_depth.shape[1] - 1):
                laplacian = (
                    self.water_depth[i-1, j] + self.water_depth[i+1, j] +
                    self.water_depth[i, j-1] + self.water_depth[i, j+1] - 4 * self.water_depth[i, j]
                )
                new_velocity[i, j] += self.wave_speed * laplacian * self.time_step
                new_velocity[i, j] *= self.damping

        new_depth += new_velocity * self.time_step
        self.water_depth, self.velocity = new_depth, new_velocity

    def compute_buoyancy(self, body_pos, body_volume, water_level):
        if body_pos[2] < water_level:
            submerged_volume = body_volume * (water_level - body_pos[2]) / body_volume
            buoyant_force = self.fluid_density * submerged_volume * 9.81
            return buoyant_force
        return 0

    def compute_drag(self, body_velocity, drag_coefficient, body_area):
        drag_force = 0.5 * self.fluid_density * drag_coefficient * body_area * body_velocity**2
        return drag_force

    def create_terrain(self):
        for _ in range(200):
            self.update_water()

        heightmap = self.water_depth

        terrain = Terrain(
            static_geometry=[
                GeometryHeightmap(
                    pose=Pose(),
                    mass=0.0,
                    size=Vector3([20.0, 20.0, 1.0]),
                    base_thickness=0.1,
                    heights=heightmap
                )
            ]
        )
        return terrain
    
def water(size: Vector2 = Vector2([20.0, 20.0])) -> Terrain:
    """
    Create a water terrain.

    :param size: Size of the water terrain.
    :returns: The created terrain.
    """
    water_terrain = WaterTerrain(grid_size=(int(size[0]), int(size[1])))
    return water_terrain.create_terrain()


def crater(
    size: tuple[float, float],
    ruggedness: float,
    curviness: float,
    granularity_multiplier: float = 1.0,
) -> Terrain:
    r"""
    Create a crater-like terrain with rugged floor using a heightmap.

    It will look like::

        |            |
         \_        .'
           '.,^_..'

    A combination of the rugged and bowl heightmaps.

    :param size: Size of the crater.
    :param ruggedness: How coarse the ground is.
    :param curviness: Height of the edges of the crater.
    :param granularity_multiplier: Multiplier for how many edges are used in the heightmap.
    :returns: The created terrain.
    """
    NUM_EDGES = 100  # arbitrary constant to get a nice number of edges

    num_edges = (
        int(NUM_EDGES * size[0] * granularity_multiplier),
        int(NUM_EDGES * size[1] * granularity_multiplier),
    )

    rugged = rugged_heightmap(
        size=size,
        num_edges=num_edges,
        density=1.5,
    )
    bowl = bowl_heightmap(num_edges=num_edges)

    max_height = ruggedness + curviness
    if max_height == 0.0:
        heightmap = np.zeros(num_edges)
        max_height = 1.0
    else:
        heightmap = (ruggedness * rugged + curviness * bowl) / (ruggedness + curviness)

    return Terrain(
        static_geometry=[
            GeometryHeightmap(
                pose=Pose(),
                mass=0.0,
                size=Vector3([size[0], size[1], max_height]),
                base_thickness=0.1 + ruggedness,
                heights=heightmap,
            )
        ]
    )


def rugged_heightmap(
    size: tuple[float, float],
    num_edges: tuple[int, int],
    density: float = 1.0,
) -> npt.NDArray[np.float_]:
    """
    Create a rugged terrain heightmap.

    It will look like::

        ..^.__,^._.-.

    Be aware: the maximum height of the heightmap is not actually 1.
    It is around [-1,1] but not exactly.

    :param size: Size of the heightmap.
    :param num_edges: How many edges to use for the heightmap.
    :param density: How coarse the ruggedness is.
    :returns: The created heightmap as a 2 dimensional array.
    """
    OCTAVE = 10
    C1 = 4.0  # arbitrary constant to get nice noise

    return np.fromfunction(
        np.vectorize(
            lambda y, x: pnoise2(
                x / num_edges[0] * C1 * size[0] * density,
                y / num_edges[1] * C1 * size[1] * density,
                OCTAVE,
            ),
            otypes=[float],
        ),
        num_edges,
        dtype=float,
    )


def bowl_heightmap(
    num_edges: tuple[int, int],
) -> npt.NDArray[np.float_]:
    r"""
    Create a terrain heightmap in the shape of a bowl.

    It will look like::

        |         |
         \       /
          '.___.'

    The height of the edges of the bowl is 1.0 and the center is 0.0.

    :param num_edges: How many edges to use for the heightmap.
    :returns: The created heightmap as a 2 dimensional array.
    """
    return np.fromfunction(
        np.vectorize(
            lambda y, x: (x / num_edges[0] * 2.0 - 1.0) ** 2
            + (y / num_edges[1] * 2.0 - 1.0) ** 2
            if math.sqrt(
                (x / num_edges[0] * 2.0 - 1.0) ** 2
                + (y / num_edges[1] * 2.0 - 1.0) ** 2
            )
            <= 1.0
            else 0.0,
            otypes=[float],
        ),
        num_edges,
        dtype=float,
    )
