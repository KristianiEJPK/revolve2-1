**Code belonging to a Master Thesis:** <br />

  "Towards Biologically Plausible Representations in the Evolution of Artificial Creatures"<br />

The dataset covers the morphological measures and fitness of Random Search and Evolution Experiments. The experiments are repeated for three encodings: <br />

* Compositional Pattern Producing Networks (CPPN) <br />
* Gene Regulatory Networks (GRN) <br />
* More Realistic GRN (mrGRN) <br />

The related data can be found here: https://www.kaggle.com/datasets/nielsss999/master-thesis-data/data <br />

**Explanation of Code** <br />
The main files of the thesis are in revolve2//examples//robot_bodybrain_ea_database. <br />

Here, we can find files to analyze data: <br />
* analyze_behavioral: Jupyter Notebook to directly plot boxplots from the .sqlite databases. Used for the boxplots in the thesis. <br />
* analyze_concentrations: Jupyter Notebook to create the GIFs of the concentrations.  <br />
* analyze_evolved_bodies: Jupyter Notebook to analyze morphological and behavioral traits. <br />

And the main file and rerun files:<br />
* main: Main file. run with arguments "algo" "mode" "file_name" "bool indicating continue on old database or not". algo is either CPPN, GRN or GRN_system (mrGRN), mode is either random search or evolution and file_name is as name.sqlite <br />
* rerun <br />


.........................

