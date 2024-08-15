**This code is for the Bachelor Thesis:** <br />

  "EMERGENT ROBOT TRAITS IN CHANGING ENVIRONMENTS DEPEND ON ENCODING PROPERTIES"<br />

The code is forked from Nielssss99/revolve2 for Niels' Master Thesis: <br />
  "Towards Biologically Plausible Encodings in the Evolution of Artificial Creatures"<br />

The dataset consists of the behavioural and morphological measures. The experiments consist of three changing terrains and repeated for two encodings: <br />

* Compositional Pattern Producing Networks (CPPN) <br />
* Gene Regulatory Networks (GRN) <br />

The related data can be found here: (https://www.kaggle.com/datasets/julianap/cppn-and-grn-dataset-for-bachelor-thesis/data) <br />

**Explanation of Code** <br />
The encodings are in revolve2\examples\revolve2\ci_group\genotypes\cppnwin\modular_robot\v2\: <br />

* CPPN: _body_develop.py <br />
* GRN: _body_develop_grn.py <br />

The terrain and fitness function codes are in ci_group/revolve2/ci_group/:<br />

* Terrains: terrains.py <br />
* Fitness functions: fitness_functions.py <br />

The main files of the thesis are in examples/eunike_robot_bodybrain_ea_database. <br />

Random Body Generator: <br />
* create_random_bodies <br />

Gradient Experiments: <br />
* tryout4gradients: Jupyter Notebook to play with gradients to see what is happening with different configurations <br />

Here, we can find files to analyze data: <br />
* analyze_behavioral: Jupyter Notebook to directly plot boxplots from the .sqlite databases. Used for the boxplots in the thesis. <br />
* analyze_concentrations: Jupyter Notebook to create the GIFs of the concentrations.  <br />
* analyze_evolved_bodies: Jupyter Notebook to analyze morphological and behavioral traits. <br />

And the main file and rerun files: <br />
* main: Main file. run with arguments "algo" "mode" "file_name" "bool indicating continue on old database or not". algo is either CPPN or GRN, mode is either random search or evolution and file_name is as name.sqlite <br />
* rerun: File to rerun experiments. run with arguments "algo" "mode" "file_name" "bool indicating headless or not" "bool indicating write files or not" "bool indicating write videos or not" <br />
* rerun_indiv: File to rerun experiments and choose which one to rerun based on genotype_id from the sqlite file. run with arguments "algo" "mode" "file_name" "bool indicating headless or not" "bool indicating write files or not" "bool indicating write videos or not" "genotype_id" <br />

And files to develop the morphology from strings: <br />
* get_morphology: Get morphology from strings in .sqlite database. run with arguments "algo" "mode" "file_name" "Experiment id to start with" "Population number to start with" "Number of experiments" "Number of Populations" <br />
* get_morphology_based_on_string: Same as previous, but now loads strings from a json file. Those files were used to redevelop the random bodies. <br />


