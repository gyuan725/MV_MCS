# Code and Computational Results

This repository contains the source code, datasets, and detailed computational results associated with the paper "Sorting under multiple eyes: A multi-view preference learning framework for~multi-criteria decision aiding".

## Repository structure

```text
.
├── data/          # Datasets used in the experiments
├── code/          # Source code
├── detailed_results/       # Detailed computational results
└── README.md
```

The main source files include:

- `main.py`: main experimental script
- `train.py`: model training and testing
- `dataload.py`: data loading and preprocessing
- `config.py`: dataset-specific configurations

## Software requirements

The experiments were implemented in Python 3.9.13. The main dependencies are:

- NumPy 1.26.4
- pandas 1.5.2
- PyTorch 1.13.0
- scikit-learn 1.1.2
- imbalanced-learn 0.12.3
- SciPy 1.13.1

The code was tested on a 64-bit Windows environment.

## Data

The `data/` directory contains the processed datasets used directly in the experiments. Each dataset is stored in a separate subdirectory, for example:

```text
data/
├── BCW/
├── CCA/
├── AD/
└── HCDR/
```

```python
# Preprocess and load data
# gama specifies the number of sub-intervals used for the piecewise representation.
X, V, y = MV_data('../data/', args.dataset, args.gama, config.cols[args.dataset])
```
The processed {dataset}{gama}.pkl files can also be loaded directly for use in the experiments.

## Running the code

Run the experiments from the `models` directory:

The experiments reported in the paper use five repetitions of five-fold stratified cross-validation with random seeds 0, 1, 2, 3, and 4.

To run the corresponding experiment, enable the relevant function call at the end of `main.py`:

```python
# Main repeated cross-validation
repeat_metrics, repeat_df, repeat_records = repeat_CV(args, X, V, y)
```

## Detailed Results

The `detailed_results/` directory contains the individual computational results underlying the aggregated results reported in the manuscript.

The subdirectories correspond to the main experiments and supplementary analyses:

- `performance_comparison/`: detailed results for the main performance comparison. 
- `performance_comparison_noisy/`: detailed results for the experiments under noisy data. 
- `view_combinations/`: results for view weights and different combinations of views.
- `hyperparam/`: results for the hyperparameter analysis.
- `benchmark/`: results for analysis of heterogeneous preference-model configurations using benchmark datasets.
- `case/`: results for the case study.
- `appendix/`: additional results reported in the appendix.
Where applicable, instance-level computational records are provided in `.pkl` format.
