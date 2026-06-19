## Nonlinear Granger Causality with Kernels

We propose two novel methods for nonlinear Granger causality for time series causal discovery. The first method is the Gaussian Process with Smooth Information Criterion ($GP_{SIC}$). This is a score-based method in which the SIC penalised marginal-likelihood of a GP model is optimised, such that the thresholded lengthscales of an Automatic Relevance Determination (ARD) kernel may be used as indicators of a Granger Causal relationship.

The second model is a Kernel Principal Component Regression (KPCR) model. This constraint-based method uses KPCR to perform the regression problems of the standard GC framework, coupled with an f-test for determining significant causal relationships. The Nystrom approximation may be used with this method for datasets with a large number of observations.

This repository includes:
- ``data`` contains scripts for generating the simulated data used in the experiments
- ``src``
    - ``gpsic_model`` contains the source code for the $GP_{SIC}$ model and the $GP_{SIC}$-based algorithm for identifying contemporaneous causal interactions
    - ``kpcr_model`` contains the source code for the KPCR model 
    - ``baseline_gp_models`` contains implementations of the baseline GP models that were compared to $GP_{SIC}$ in the manuscript
- ``tutorials`` contains tutorials for running the KPCR model, $GP_{SIC}$ model, and $GP_{SIC}$-based method for contemporaneous interactions

The non-GP baseline methods were implemented using code from public repositories, as referenced. The real-world datasets tested in the manuscript were also sourced from publicly available repositories, as referenced. 

### Usage

The enviroment with necessary dependencies may be replicated using

``conda env create -f environment.yml``

The tutorials provide a guide on how to run the models on a given time-series dataset. 

### Citation

Please cite this work as follows: 

```
@article{murphy_constraint-_2026,
	title = {Constraint- and {Score}-{Based} {Nonlinear} {Granger} {Causality} {Discovery} with {Kernels}},
	volume = {115},
	issn = {1573-0565},
	url = {https://doi.org/10.1007/s10994-026-07093-z},
	number = {7},
	journal = {Machine Learning},
	author = {Murphy, Fiona and Benavoli, Alessio},
	month = jun,
	year = {2026},
	pages = {150},
}
```