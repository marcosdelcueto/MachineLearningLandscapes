# MachineLearningLandscapes
Main program to generate research landscapes, explore and analyze them.
Calculation parameters can be changed in **input_MLL.inp** file

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
- **raw_data**: directory containing main raw data of the manuscript (see below)
- **scripts**: directory conraining several scripts used to handle and analyze intermediate data

---

## Raw Data

For reproducibility, we include several large .tar.gz files containing the following results for the main results discussed in the manuscript for n=3 and S=0.10:

- The 100 structure-property functions: 100SPFs_S0_10_n3.tar.gz
- 10-fold cross-validation, N1 exploration with N<sub>1</sub>=50: logs_N1_0050_kfold_10_CV.tar.gz
- 10-fold cross-validation, N1 exploration with N<sub>1</sub>=200: logs_N1_0200_kfold_10_CV.tar.gz
- 10-fold cross-validation, N1 exploration with N<sub>1</sub>=1000:logs_N1_1000_kfold_10_CV.tar.gz
- Last-10% validation, N1 exploration with N<sub>1</sub>=50: logs_N1_0050_last_10percent_validation.tar.gz
- Last-10% validation, N1 exploration with N<sub>1</sub>=200: logs_N1_0200_last_10percent_validation.tar.gz
- Last-10% validation, N1 exploration with N<sub>1</sub>=1000: logs_N1_1000_last_10percent_validation.tar.gz

---

## Scripts

- **MLL_calculate_a.py**:
- **MLL_calculate_S.py**:
- **MLL_consolidate_logs.sh**:
- **MLL_consolidate_logs_smart.sh**:
- **MLL_delete_verbose_info_from_log.sh**:
- **MLL_extract_x_G_data.sh**:
- **MLL_get_fig_rmse_from_stdout.py**:
- **MLL_get_stdout_from_logs.py**:
- **MLL_logs_to_savelogs.sh**:

---

## License
