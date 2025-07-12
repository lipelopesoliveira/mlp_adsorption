# MLP Adsorption

This repository contains the MLP Adsorption package, which provides tools for molecular simulations and adsorption studies using machine learning potentials.

## Feature Overview

The MLP Adsorption package includes modules for performing Grand Canonical Monte Carlo (GCMC) simulations and Widom insertion tests. It is designed to work with the ASE (Atomic Simulation Environment) framework and supports various machine learning potentials.

## Usage

To use the MLP Adsorption package, you can import the necessary classes from the `mlp_adsorption` module. For example, to perform GCMC simulations, you can use the `GCMC` class:

```python
from mlp_adsorption.gcmc import GCMC

# Initialize the GCMC simulation with the required parameters
gcmc_simulation = GCMC(model, framework_atoms, adsorbate_atoms, temperature, pressure, n_steps)

# Run the simulation for a specified number of steps
gcmc_simulation.run(10000)
```
