"""Configuration parameters for this example."""

DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS = 1#5
NUM_SIMULATORS = 8
POPULATION_SIZE = 100#100
OFFSPRING_SIZE = 100#50
NUM_GENERATIONS = 0#100
NPARENTS = 2
PARENT_TOURNAMENT_SIZE = 4
SURVIVOR_TOURNAMENT_SIZE = 4
TERRAIN = "flat"
FITNESS_FUNCTION = "x_speed_Miras2021" # "xy_displacement"
ZDIRECTION = False # Whether to evolve in the z-direction.
CPPNBIAS = False # Whether BIAS is an Input for the CPPN.
CPPNCHAINLENGTH = False # Whether CHAINLENGTH is an Input for the CPPN.
CPPNEMPTY = True # Whether EMPTY Module is an Output for the CPPN.

SIMULATION_TIME = 30
SAMPLING_FREQUENCY = 5
SIMULATION_TIMESTEP = 0.001
CONTROL_FREQUENCY = 20

