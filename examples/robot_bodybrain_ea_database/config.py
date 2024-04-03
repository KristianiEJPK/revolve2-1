"""Configuration parameters for this example."""

DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS = 1#5
NUM_SIMULATORS = 8
POPULATION_SIZE = 100#100#100
OFFSPRING_SIZE = 50 # 50#50
NUM_RANDOM_SEARCH = 50
NUM_GENERATIONS = 50 #100
NPARENTS = 2
PARENT_TOURNAMENT_SIZE = 4
SURVIVOR_TOURNAMENT_SIZE = 4
CROSSOVER_PROBABILITY = 1
MUTATION_PROBABILITY = 0.9
TERRAIN = "flat" # "tilted
FITNESS_FUNCTION = "x_speed_Miras2021" # "x_efficiency"# "xy_displacement"
ALGORITHM = "GRN" # "CPPN"


ZDIRECTION = False # Whether to evolve in the z-direction.
CPPNBIAS = False # Whether BIAS is an Input for the CPPN.
CPPNCHAINLENGTH = False # Whether CHAINLENGTH is an Input for the CPPN.
CPPNEMPTY = False # Whether EMPTY Module is an Output for the CPPN.

MAX_PARTS = 20 # Maximum number of parts in the body --> better pass as parameter???? 
MODE_COLLISION = False # Whether to stop if collision occurs
MODE_CORE_MULT = True # Whether to allow multiple core slots
MODE_SLOTS4FACE = False # Whether multiple slots can be used for a single face for the core module
MODE_SLOTS4FACE_ALL = False # Whether slots can be set for all 9 attachments, or only 3, 4, 5
MODE_NOT_VERTICAL = True # Whether to disable vertical expansion of the body

SIMULATION_TIME = 30
SAMPLING_FREQUENCY = 5
SIMULATION_TIMESTEP = 0.001
CONTROL_FREQUENCY = 20


# Assertions
assert SIMULATION_TIME == 30, "SIMULATION_TIME must be 30"
assert (NUM_GENERATIONS == 150) or (NUM_RANDOM_SEARCH == 50), "NUM_GENERATIONS must be 150 or NUM_RANDOM_SEARCH must be 50"
assert POPULATION_SIZE == 100, "POPULATION_SIZE must be 100"
assert OFFSPRING_SIZE == 50, "OFFSPRING_SIZE must be 50"
assert NPARENTS == 2, "NPARENTS must be 2"
assert PARENT_TOURNAMENT_SIZE == 4, "PARENT_TOURNAMENT_SIZE must be 4"
assert SURVIVOR_TOURNAMENT_SIZE == 4, "SURVIVOR_TOURNAMENT_SIZE must be 4"
assert not ((NUM_RANDOM_SEARCH != 0) and (NUM_GENERATIONS != NUM_RANDOM_SEARCH)), "NUM_RANDOM_SEARCH must be equal to NUM_GENERATIONS if NUM_RANDOM_SEARCH != 0"

if ALGORITHM == "GRN":
    assert MUTATION_PROBABILITY == 0.9, "MUTATION_PROBABILITY must be 0.9 if ALGORITHM is GRN"
    assert CROSSOVER_PROBABILITY == 1, "CROSSOVER_PROBABILITY must be 1 if ALGORITHM is GRN"
elif ALGORITHM == "CPPN":
    assert MUTATION_PROBABILITY == 0.9, "MUTATION_PROBABILITY must be 0.9 if ALGORITHM is CPPN"
    assert CROSSOVER_PROBABILITY == 0, "CROSSOVER_PROBABILITY must be 0 if ALGORITHM is CPPN"
else:
    raise ValueError("ALGORITHM must be either GRN or CPPN")


if MODE_CORE_MULT and (ALGORITHM == "GRN"):
    print("For GRN MODE_CORE_MULT only will provide a 3 x 3 grid during querying. No additional attachments!")
else:
    assert not MODE_SLOTS4FACE, "MODE_SLOTS4FACE must be False if MODE_CORE_MULT is True"
    assert not MODE_SLOTS4FACE_ALL, "MODE_SLOTS4FACE_ALL must be False if MODE_CORE_MULT is True"

if MODE_SLOTS4FACE_ALL:
    assert MODE_SLOTS4FACE, "MODE_SLOTS4FACE must be True if MODE_SLOTS4FACE is True"




