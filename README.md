# Project LUNA-PASSA
This is a clone of the Final Project of AM207 - Stochastic Methods of Data Aanalysis, Inference and Optimization. The original repository can be found [here](https://github.com/weyesians/luna-passa).


## Overview
Final project for AM207 at Harvard. LUNA stands for "Learned Uncertainty-Aware"; and PASSA for "Paulina, Avriel, Sangyoon, Shih-Yi, and Arthur". 

## Team Members

* Avriel Epps-Darling
* Paulina Toro Isaza
* Sangyoon Park
* Shih-Yi Tseng
* Arthur Young

## Description
The goal of this project is to reproduce a SOTA method in published ICML work, "[Learned Uncertainty-Aware (LUNA) Bases for Bayesian Regression using Multi-Headed Auxiliary Networks](https://arxiv.org/abs/2006.11695)". In this project, we implemented LUNA and reproduced key results in this paper, and extended the scope to invistigate how different factors impact the behavior of LUNA. A complete report can be found in this [notebook](https://github.com/sytseng/AM207_Final_Project_Fall_2020/blob/main/am207_final_project_luna.ipynb).

## Directory Structure
```
├── README.md          <- The top-level README for developers using this project.
│
├── pyproject.toml     <- The file that defines build system, project metadata, and other
│                         install requirements.
│
├── poetry.lock        <- The file that resolves and downloads dependencies in `pyproject.toml`.
│
├── data
│
├── src                <- Source code for use in this project.
│
├── notebooks          <- Jupyter notebooks.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
```

## Dependency Management
The current project repo uses [`poetry`](https://python-poetry.org/docs/) to manage
dependencies among different Python packages, which is essential to reproducibility.
Hence, all users of and contributors to the current project repo are expected to use
`poetry` to install and/or update packages. Following are some basic commands that can
get you started:

Once you clone the current repo into your local machine, you can run:
```
$ poetry install
```
to install the right versions of packages for running scripts in the project repo.

To use the new Python configuration that has been installed, you need to run:
```
$ poetry shell
```
which will activate the virtual environment for the project repo.

You can simply type:
```
$ exit
```
to exit from the virtual environment and return to the global (or system) Python installation.

If a new script that you are planning to add and commit to the repo requires a new package,
you can run:
```
$ poetry add [package-name]
```
This will add the new package to the repo's virtual environment and update the corresponding
information in the `pyproject.toml` and `poetry.lock`, which you need to add and commit as well.

Conversely, if changes to the project repo make a certain package not needed anymore,
you can remove it by running:
```
$ poetry remove [package-name]
```

Once you set up the virtual environment using `poetry`, you can create the corresponding `jupyter` kernel as follows:
```
poetry run ipython kernel install --user --name=luna-passa
```
Running a notebook on this new kernel (`luna-passa`) will enable you to use the project-specific packages installed in the virtual environment.

You can check out the following resources to learn more about how to set up and use `poetry`:

- [Official Documentation](https://python-poetry.org/docs/)
- [This tutorial](https://blog.jayway.com/2019/12/28/pyenv-poetry-saviours-in-the-python-chaos/)
walks you through how to install and use `poetry` along with `pyenv` (`pyenv` helps to better manage
multiple versions of Python installation in the local machine).
