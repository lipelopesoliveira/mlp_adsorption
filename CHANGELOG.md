# Changelog

## v[X.Y.Z] - YYYY-MM-DD (Unreleased)

### New Features 🎉

### Fixed 🐛

### Enhanced ✨

### Documentation 📖

### Removed 🗑️

## v[0.4.0] - Unreleased

### New Features 🎉

- Added the Peng-Robinson equation of state (EOS) to the `mlp_adsorption.eos` module.
  - This allows for the calculation of fugacity coefficients and bulk phase density using the Peng-Robinson EOS.
- Add functions to calculate perpendicular lengths and unit cell repetitions in `utilities`.
- Add a new parameter `cutoff` on the `mlp.gcmc` class for controlling the supercell check based on a cutoff radius for the potential.
- Add the `get_density` function to the `mlp_adsorption.utilities` module.
- Add a `mlp_adsorption.utilities.make_cubic` function that can create a cubic (or close to) supercell from a given structure.
- Now all movements are based on a `numpy.random.Generator` to ensure reproducibility. A `random_seed` can be provided to the `GCMC` and `Widom` classes for this purpose.
- New module for checking the overlap between atoms (`mlp_adsorption.operations.check_overlap`)
- A new `BaseSimulator` method created to abstract the system state management and general simulation logic.
- Now all the output of the simulations are managed by a `mlp_adsorption.logger` instance. It introduces:
  - `BaseLogger`: A base class for logging simulation information, including restart and iteration details.
  - `GCMCLogger`: A logger specifically for GCMC simulations, extending `BaseLogger` with additional functionality.
  - `WidomLogger`: A logger specifically for Widom simulations, extending `BaseLogger` with additional functionality.

### Fixed 🐛

- Fix the framework density calculation on `GCMC` and `Widom` class.

### Enhanced ✨

- Move the random operations (insertion, rotation, translation) to a separate `mlp_adsorption.operations.py` module for better organization and reusability.

### Documentation 📖

- Add the critical parameters to the examples.

### Removed 🗑️

- Old `mlp_adsorption.operations.vdw_overlap` and `mlp_adsorption.operations.vdw_overlap2` functions.

## v[0.3.2] - 2025-08-17

### New Features 🎉

- New `restart` method in the `GCMC` class to allow restarting a GCMC simulation from a saved state.
  - It reads the state from an existing `Trajectory` object, enabling the continuation of simulations without losing progress.
  - It reads the total uptake, total energy, and adsorption energy `npy` files for seamless simulation restoration.
- Now the `vdw_factor` can be set when initializing the `GCMC` class, allowing for more flexible control over the Van der Waals radii scaling. By default, it is set to 0.6.
- Add `flush=True` to print statements for immediate output in optimization and simulation logs

### Fixed 🐛

- Fixed the handling of NaN values in the Van der Waals radii in both `gcmc.py` and `widom.py` files to ensure that any NaN value is replaced by 1.5, preventing potential issues during simulations.
- Fixed unit for Henry constant on header and footer of Widom output. Now it correctly reports the unit as [mol kg-1 Pa-1].
- Change the energy return value in the `try_insertion` method of the `Widom` class to 1000 eV when there is a Van der Waals overlap and limit the number of insertion attempts to 100. This change ensures that the method prevents an infinite loop in the simulation.

### Enhanced ✨

- Restart of a GCMC simulation:
  - Now the `load_state` method in the `GCMC` class can load the state from a `Trajectory` object, allowing for restarting simulations from saved states.

- Add new examples for running GCMC simulations with different configurations, including:

  - Basic
    - Widom Insertion
    - Rigid GCMC
    - Rigid GCMC Isotherm
  - Intermediate
    - Geometry Optimization + GCMC
    - Flexible GCMC: Molecular Dynamics + GCMC
  
### Documentation 📖

- New examples are documented in the README file, providing clear instructions on how to run GCMC simulations and Widom insertion tests.

### Removed 🗑️

- Removed commented-out code for saving the system state in xyz format in the `gcmc.py` file to clean up the codebase.

## v[0.3.1] - 2025-08-05

### New Features 🎉

### Fixed 🐛

- Fixed the calculation of Henry's law constant in the Widom insertion method to ensure it is correctly computed in units of [mol kg-1 Pa-1].
- Fixed the calculation of the heat of adsorption (Qst) in the Widom insertion method to ensure it is correctly computed in units of [kJ/mol].

### Enhanced ✨

### Documentation 📖

### Removed 🗑️

## v[0.3.0] - 2025-07-31

### New Features 🎉

- Now GCMC has three different methods for NPT Molecular Dynamics (MD):
  - Isotropic volume change and fixed shape with Berendsen thermostat/barostat (`mode=iso_shape`)
  - Anisotropic volume change and fixed shape with Berendsen thermostat/barostat (`mode=aniso_shape`)
  - Anisotropic volume change and flexible shape with Nose-Hoover/Parrinello-Rahman thermostat/barostat (`mode=aniso_flex`).
- Added a new method `nvt` to the GCMC class for running NVT MD simulations.

### Fixed 🐛

[#10](https://github.com/lipelopesoliveira/mlp_adsorption/issues/10) - The cell angles are not changing during the MD simulation.

### Enhanced ✨

- Now the NPT MD simulation log also prints the cell volume.
- Now it is possible to control the type of NPT MD simulation through the use of a keyword argument on the `npt` method of GCMC simulation.

### Documentation 📖

### Removed 🗑️

## v[0.2.0] - 2025-07-12

### New Features 🎉

- Support for saving the snapshots of the simulation on the same trajectory file in the `gcmc.py` and `widom.py` codes.
- New examples for using the GCMC and Widom insertion methods in the `examples` directory.

### Fixed 🐛

- Incorrect statement on the insertion method in the `gcmc.py` file.

### Enhanced ✨

- Updated the `gcmc.py` file to use the `Trajectory.write` method instead of `write_proteindatabank`.
- Updated the `ase_utils.py` file to accept a `trajectory` parameter in the `npt_md` function.
- Replaces PDB output with ASE Trajectory for snapshot saving in `widom.py`.
- Removed the unused import of `write_proteindatabank` in `widom.py`.

### Documentation 📖

- Added docstrings to all functions and classes in the `gcmc.py` and `widom.py` files.
- Updated the README file with examples of how to use the GCMC and Widom insertion methods.

### Removed 🗑️

- Unused imports and commented-out code in `widom.py`.

## v[0.1.0] - 2025-07-12

- Initial release of the MLP Adsorption code, including the GCMC and Widom methods with basic functionality on the simulation of adsorption processes.
