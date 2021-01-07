# MachineLearningLandscapes
This repository contains the code for **Determining Usefulness of Machine Learning in Materials Discovery Using Simulated Research Landscapes** by _M del Cueto_ and _A Troisi_.

A description on how to use Simulated Research Landscapes to assess ML is given in the manuscript. Calculation parameters can be changed in **input_MLL.inp** file:

- **General Landscape Parameters**: it allows to select if we will create the Structure Property Functions (SPFs) or they will be read from previously generated file. Also allows to select smoothness, dimensionality and other aspects of the SPFs, as well as other general options like verbosity and parallelism.

- **Exploration general parameters**: general options regarding the exploration of the SPFs to generate the research landscapes: how many adventurousness values will be used per SPF, and what is the threshold distance in the a-weighted exploration.

- **N1 exploration parameters**: general options to control if N1 exploration is performed, what method and number of steps will be used.

- **N1 error metric analysis**: options to select N1 error metric and whether to plot it.

- **N2 exploration parameters**: general options to control if N2 exploration is performed, what method and number of steps will be used.

- **Machine Learning parameters**: select type of cross-validation and model-specific options.

- **Differential evolution parameters**: general options for the differential evolution algorithms used to optimize hyperparameters of GPR and KRR models

---

## Prerequisites

The necessary packages (with the tested versions with Python 3.7.6) are specified in the file _requirements.txt_. These packages can be installed with pip:

```
pip3 install -r requirements.txt
```
---

## Contents
- **input_MLL.inp**: input file containing options to control type of exploration and its parameters

- **MLL.py**: main python code to create, explore and analyze the research landscapes

- **scripts**: directory containing several scripts used to handle and analyze intermediate data (brief description in each file)

- **example_exploration_landscapes**: directory containing minimalistic example run to generate three 2D Structure Property Functions, and explore them with N<sub>0</sub>=15, N<sub>1</sub>=100 and N<sub>2</sub>=5, with adventurousness a = {10,100}. Output files are provided, and can be reproduced simply by running:

```
python MLL.py
```

---

## License
**Authors**: Marcos del Cueto and Alessandro Troisi

Licensed under the [MIT License](LICENSE.md) 
