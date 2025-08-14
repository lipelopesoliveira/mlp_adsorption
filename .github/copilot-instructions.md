# GitHub Copilot Instructions for MLP Adsorption

This repository is a specialized Python toolkit for computational chemistry researchers working with adsorption phenomena, machine learning potentials, and molecular simulations.

## Repository Context

### Scientific Domain
This package focuses on:
- **Adsorption phenomena modeling** - Studying how molecules interact with surfaces and porous materials
- **Machine learning potential integration** - Using ML models (MACE, torch-dftd) for atomic interactions
- **Grand Canonical Monte Carlo (GCMC)** - Simulations in the μVT ensemble for adsorption studies
- **Widom insertion method** - Computing Henry coefficients and adsorption enthalpies
- **Molecular dynamics (MD)** - Time-evolution simulations of molecular systems

### Key Scientific Libraries
- **ASE (Atomic Simulation Environment)** - Core framework for atomic structure manipulation and calculations
- **MACE-torch** - State-of-the-art machine learning potential for molecular systems
- **PyMatGen** - Materials analysis and crystal structure handling
- **NumPy/SciPy** - Numerical computations and scientific algorithms

## Core Package Structure (mlp_adsorption/)

### Primary Modules
- `gcmc.py` - **GCMC class**: Grand Canonical Monte Carlo implementation with insertion/deletion/movement moves
- `widom.py` - **Widom class**: Widom insertion test for computing adsorption properties
- `ase_utils.py` - ASE integration utilities including:
  - `nPT_Berendsen`, `nPT_NoseHoover` - Pressure-temperature coupling
  - `nVT_Berendsen` - Temperature coupling
  - `crystalOptmization` - Structural relaxation
- `lj.py` - Lennard-Jones potential implementations for classical interactions
- `utilities.py` - Core utility functions:
  - `random_position`, `random_rotation` - Monte Carlo move generators
  - `vdw_overlap` - Overlap detection for molecular insertions
  - `enthalpy_of_adsorption` - Thermodynamic property calculations

### Data Directory
- `data/` - Contains molecular structures and force field parameters for common adsorbates (CO₂, H₂O, etc.)

## Code Patterns and Conventions

### ASE Integration
All molecular structures are handled as `ase.Atoms` objects:
```python
framework_atoms: ase.Atoms  # Host material structure
adsorbate_atoms: ase.Atoms  # Guest molecule structure
```

### Calculator Interface
Energy calculations use ASE's calculator interface:
```python
model: ase.calculators.calculator.Calculator
# Common calculators: MACE, LAMMPS, or custom ML potentials
```

### Monte Carlo Simulations
- Use `np.random` for stochastic moves
- Energy differences for acceptance criteria: `ΔE = E_new - E_old`
- Boltzmann factor: `exp(-ΔE / (kB * T))`
- Chemical potential coupling in GCMC: `μ`, `fugacity_coeff`

### Temperature and Pressure Units
- Temperature: Kelvin (K)
- Pressure: Pascal (Pa) or bar, convert using `ase.units`
- Energy: eV (ASE default)
- Distance: Angstrom (ASE default)

### Van der Waals Handling
- `vdw_radii: np.ndarray` - Atomic radii for overlap detection
- Hard-sphere approximation for molecular insertions
- Lennard-Jones parameters for classical interactions

## Development Standards

### Type Hints
- All functions should use comprehensive type hints
- Import types from `typing` module when needed
- ASE types: `ase.Atoms`, `ase.calculators.calculator.Calculator`

### Error Handling
- Validate input parameters (temperature > 0, pressure > 0)
- Handle calculator failures gracefully
- Check for atomic overlaps before energy calculations

### Logging and Output
- Use `tqdm` for progress bars in long simulations
- Support both file output and console logging
- Include timestamps and system information in output files

### Testing Patterns
- Unit tests in `tests/` directory using pytest
- Test both successful cases and error conditions
- Mock expensive calculator calls when possible
- Validate thermodynamic properties against known results

## Common Development Tasks

### Adding New Monte Carlo Moves
1. Implement move in utilities.py
2. Add acceptance criteria based on energy difference
3. Update statistics tracking in GCMC class
4. Add unit tests for the new move

### Integrating New Calculators
1. Ensure calculator follows ASE interface
2. Test with simple systems first
3. Validate energy conservation in MD runs
4. Check gradient accuracy if using forces

### Performance Optimization
- Use NumPy vectorization for array operations
- Cache expensive calculations (e.g., distance matrices)
- Consider numba compilation for hot loops
- Profile memory usage for large systems

## Scientific Best Practices

### Simulation Validation
- Always equilibrate systems before data collection
- Monitor convergence of thermodynamic properties
- Compare with experimental data or other simulation packages
- Report statistical uncertainties

### Reproducibility
- Set random seeds for stochastic simulations
- Save input parameters and system configurations
- Use version control for simulation scripts
- Document calculation settings clearly

### Physical Realism
- Ensure charge neutrality in ionic systems
- Respect periodic boundary conditions
- Use appropriate ensemble for the physical situation
- Validate force field parameters for your system

## Build and Test Commands

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all checks (linting, type checking, tests)
make run-checks

# Build documentation
make docs

# Run specific tests
pytest tests/utilities_test.py -v

# Format code
black mlp_adsorption/
isort mlp_adsorption/
```

When suggesting code, prioritize:
1. Scientific accuracy and physical realism
2. Integration with existing ASE workflow
3. Proper error handling and validation
4. Clear documentation of scientific assumptions
5. Efficient NumPy/SciPy usage for numerical operations