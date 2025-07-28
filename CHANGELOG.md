# Changelog

## Unreleased

## [0.2.0] - 2025-07-12

### Added

- Support for saving the snapshots of the simulation on the same trajectory file in the `gcmc.py` and `widom.py` codes.

### Fixed

- Incorrect statement on the insertion method in the `gcmc.py` file.

### Changed

- Updated the `gcmc.py` file to use the `Trajectory.write` method instead of `write_proteindatabank`.
- Updated the `ase_utils.py` file to accept a `trajectory` parameter in the `npt_md` function.
- Replaces PDB output with ASE Trajectory for snapshot saving in `widom.py`.
- Removed the unused import of `write_proteindatabank` in `widom.py`.

### Removed

- Unused imports and commented-out code in `widom.py`.

## [0.1.0] - 2025-07-12

- Initial release of the MLP Adsorption code.

### Added

- Initial implementation of GCMC and Widom insertion methods.
