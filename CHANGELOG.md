# Changelog

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
