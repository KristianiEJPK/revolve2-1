**Code belonging to a Master Thesis:** <br />

  "Towards Biologically Plausible Representations in the Evolution of Artificial Creatures"<br />

The dataset covers the morphological measures and fitness of Random Search and Evolution Experiments. The experiments are repeated for three encodings: <br />

* Compositional Pattern Producing Networks (CPPN) <br />
* Gene Regulatory Networks (GRN) <br />
* More Realistic GRN (mrGRN) <br />

The related data can be found here: https://www.kaggle.com/datasets/nielsss999/master-thesis-data/data <br />

**Explanation of Code** <br />
The encodings are in revolve2\ci_group\revolve2\ci_group\genotypes\cppnwin\modular_robot\v2\: <br />

* CPPN: _body_develop.py <br />
* GRN: _body_develop_grn.py <br />
* mrGRN: _body_develop_grn_system.py <br />

The main files of the thesis are in revolve2//examples//robot_bodybrain_ea_database. <br />

Random Body Generator: <br />
* create_random_bodies <br />

Here, we can find files to analyze data: <br />
* analyze_behavioral: Jupyter Notebook to directly plot boxplots from the .sqlite databases. Used for the boxplots in the thesis. <br />
* analyze_concentrations: Jupyter Notebook to create the GIFs of the concentrations.  <br />
* analyze_evolved_bodies: Jupyter Notebook to analyze morphological and behavioral traits. <br />

And the main file and rerun files: <br />
* main: Main file. run with arguments "algo" "mode" "file_name" "bool indicating continue on old database or not". algo is either CPPN, GRN or GRN_system (mrGRN), mode is either random search or evolution and file_name is as name.sqlite <br />
* rerun <br />

And files to develop the morphology from strings: <br />
* get_morphology: Get morphology from strings in .sqlite database. run with arguments "algo" "mode" "file_name" "Experiment id to start with" "Population number to start with" "Number of experiments" "Number of Populations" <br />
* get_morphology_based_on_string: Same as previous, but now loads strings from a json file. <br />


