# Changelog

## Unreleased

### New Features 🎉

### Fixed 🐛

### Enhanced ✨

### Documentation 📖

### Removed 🗑️

## v[0.3.2] - 2025-08-05

### New Features 🎉

- New `restart` method in the `GCMC` class to allow restarting a GCMC simulation from a saved state.
  - It reads the state from an existing `Trajectory` object, enabling the continuation of simulations without losing progress.
  - It reads the total uptake, total energy, and adsorption energy `npy` files for seamless simulation restoration.

### Fixed 🐛

### Enhanced ✨

- Restart of a GCMC simulation:
  - Now the `load_state` method in the `GCMC` class can load the state from a `Trajectory` object, allowing for restarting simulations from saved states.

### Documentation 📖

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
