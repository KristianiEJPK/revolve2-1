"""Configuration parameters for this example."""

DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS = 1#5
NUM_SIMULATORS = 8
POPULATION_SIZE = 100#100
OFFSPRING_SIZE = 100#50
NUM_GENERATIONS = 100#100
NPARENTS = 2
PARENT_TOURNAMENT_SIZE = 4
SURVIVOR_TOURNAMENT_SIZE = 4
CROSSOVER_PROBABILITY = 0
MUTATION_PROBABILITY = 0.9
TERRAIN = "flat"
FITNESS_FUNCTION = "x_speed_Miras2021" # "xy_displacement"
ZDIRECTION = False # Whether to evolve in the z-direction.
CPPNBIAS = False # Whether BIAS is an Input for the CPPN.
CPPNCHAINLENGTH = False # Whether CHAINLENGTH is an Input for the CPPN.
CPPNEMPTY = False # Whether EMPTY Module is an Output for the CPPN.

MAX_PARTS = 20 # Maximum number of parts in the body --> better pass as parameter???? 
MODE_COLLISION = True # Whether to stop if collision occurs
MODE_CORE_MULT = False # Whether to allow multiple core slots
MODE_SLOTS4FACE = False # Whether multiple slots can be used for a single face for the core module
MODE_SLOTS4FACE_ALL = False # Whether slots can be set for all 9 attachments, or only 3, 4, 5
MODE_NOT_VERTICAL = True # Whether to disable vertical expansion of the body



SIMULATION_TIME = 30
SAMPLING_FREQUENCY = 5
SIMULATION_TIMESTEP = 0.001
CONTROL_FREQUENCY = 20

