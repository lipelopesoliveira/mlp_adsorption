import numpy as np
import warnings
from ase import Atoms, units
from ase.cell import Cell
from pymatgen.core import Structure
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)

from flames.exceptions import InsertionDeletionError, MoveKeyError


def enthalpy_of_adsorption(energy, number_of_molecules, temperature):
    """
    Calculates the enthalpy of adsorption as

    H = <EN> - <E><N> / <N^2> - <N>^2 - RT

    adapted from J. Phys. Chem. 1993, 97, 51, 13742-13752.

    Please note that Heat of adsorption (Q_iso) = -Enthalpy of adsorption (H).

    The isosteric enthalpy of adsorption, H, is defined as the heat which is released
    when an adsorptive binds to a surface. The enthalpy of adsorption (H) is a negative
    number and the isosteric heat (Q_iso) of adsorption is a positive number.
    For a deeper discussion see: Dalton Trans., 2020, 49, 10295.

    Parameters
    ----------
    energy : 1D array
        List with the potential energy of the adsorbed phase for each MC cycle in units of Kelvin.

    number_of_molecules : 1D array
        List with the number of molecules in the simulation system for each MC cycle.

    temperature : float
        Temperature of the simulation in Kelvin

    Returns
    ----------

    H : float
        Enthalpy of adsorption in units of kJ⋅mol-1
    """
    # Define basic constants
    R = units.kB / (units.kJ / units.mol)  # kJ⋅K−1⋅mol−1

    # Convert energy from Kelvin to kJ/mol
    E = np.array(energy) * R
    N = np.array(number_of_molecules)

    EN = E * N

    # Calculate the enthalpy of adsorption. Here <N^2> - <N>^2 = VAR(N)
    H = (EN.mean() - E.mean() * N.mean()) / np.var(N) - R * temperature

    return H


def get_density(structure: Atoms) -> float:
    """
    Get the density of the framework in g/cm^3
    """

    mass = np.sum(structure.get_masses()) / units.kg * 1e3  # Convert from amu to g
    volume = structure.get_volume() * (1e-8**3)  # Convert from Angs^3 to cm^3

    return mass / volume


def get_perpendicular_lengths(cell: Cell) -> tuple[float, float, float]:
    """
    Calculate the perpendicular lengths of a unit cell.

    Parameters
    ----------
    cell : ase.Cell
        The unit cell for which to calculate the perpendicular lengths.

    Returns
    -------
    tuple[float, float, float]
        The perpendicular lengths in the x, y, and z directions.
    """

    a, b, c = cell.array

    axb = np.cross(a, b)
    bxc = np.cross(b, c)
    cxa = np.cross(c, a)

    # Calculate perpendicular widths
    cx = float(cell.volume / np.linalg.norm(bxc))
    cy = float(cell.volume / np.linalg.norm(cxa))
    cz = float(cell.volume / np.linalg.norm(axb))

    return cx, cy, cz


def calculate_unit_cells(cell: Cell, cutoff: float = 12.6) -> list[int]:
    """
    Calculate the number of unit cell repetitions so that all supercell lengths are larger than
    twice the interaction potential cut-off radius.

    RASPA considers the perpendicular directions the directions perpendicular to the `ab`, `bc`,
    and `ca` planes. Thus, the directions depend on who the crystallographic vectors `a`, `b`,
    and `c` are and the length in the perpendicular directions would be the projections
    of the crystallographic vectors on the vectors `a x b`, `b x c`, and `c x a`.
    (here `x` means cross product)

    Parameters
    ----------
    cell : ase.Cell
        The unit cell for which to calculate the perpendicular lengths.
    cutoff : float
        The interaction potential cut-off radius.

    Returns
    -------
    supercell : list[int]
        (3,1) list containg the number of repeating units in `x`, `y`, `z` directions.
    """

    cx, cy, cz = get_perpendicular_lengths(cell)

    # Calculate UnitCells array
    supercell = [int(i) for i in np.ceil(2.0 * cutoff / np.array([cx, cy, cz]))]

    return supercell


def make_cubic(
    structure: Atoms,
    min_length: int = 10,
    force_diagonal: bool = False,
    force_90_degrees: bool = False,
    min_atoms: int = 0,
    max_atoms: int = 10000,
    angle_tolerance: float = 1e-3,
) -> Atoms:
    """
    Transform the primitive structure into a supercell with alpha, beta, and
    gamma equal, or close, to 90 degrees. The algorithm will iteratively increase the size
    of the supercell until the largest inscribed cube's side length is at least 'min_length'
    and the number of atoms in the supercell falls in the range ``min_atoms < n < max_atoms``.

    Parameters
    ----------
    min_length : float, optional
        Minimum length of the cubic cell (default is 10)
    force_diagonal : bool, optional
        If True, generate a transformation with a diagonal transformation matrix (default is False)
    force_90_degrees : bool, optional
        If True, force the angles to be 90 degrees (default is False)
    min_atoms : int, optional
        Minimum number of atoms in the supercell (default is 0)
    max_atoms : int, optional
        Maximum number of atoms in the supercell (default is 10000)
    angle_tolerance : float, optional
        The angle tolerance for the transformation (default is 1e-3)

    Returns
    """

    pmg_structure = Structure.Structure.from_ase_atoms(structure)

    cubic_dict = CubicSupercellTransformation(
        min_length=min_length,
        force_90_degrees=force_90_degrees,
        force_diagonal=force_diagonal,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        angle_tolerance=angle_tolerance,
    ).apply_transformation(pmg_structure)

    ase_structure = cubic_dict.to_ase_atoms()

    return ase_structure


def check_weights(move_weights: dict) -> dict:
    """
    Check if the move weights are valid and normalize them to sum to 1.

    Parameters:
    - move_weights (dict): A dictionary containing the move weights for 'insertion', 'deletion', 'translation', and 'rotation'.
    Returns:
    - dict: Normalized move weights.
    """

    valid_keys = {"insertion", "deletion", "translation", "rotation"}

    # Check if move_weights is a dictionary
    if type(move_weights) is not dict:
        raise TypeError("move_weights must be a dictionary, not " + str(type(move_weights)))
    
    # Check if the keys in move weights are insertion, deletion, translation, rotation
    if not all([i in valid_keys for i in set(move_weights.keys())]):
        raise MoveKeyError(list(move_weights.keys()))

    # Raise a warning if any of the four moves are missing
    for key in valid_keys:
        if key not in move_weights:
            warnings.warn(f"Warning: move_weights is missing the key '{key}'. Assuming weight 0 for this move.")
            move_weights[key] = 0

    # Check if all weights are numbers and non-negative
    for k, v in move_weights.items():
        if type(v) not in [int, float]:
            raise TypeError(f"move_weights['{k}'] must be a number, not " + str(type(v)))
        if v < 0:
            raise ValueError(f"move_weights['{k}'] must be non-negative, not " + str(v))

    # Check if insertion and deletion weights are equal
    if move_weights["insertion"] != move_weights["deletion"]:
        raise InsertionDeletionError(move_weights["insertion"], move_weights["deletion"])
    
    # Normalize weights to sum to 1
    total_weight = sum(move_weights.values())
    move_weights = {k: v / total_weight for k, v in move_weights.items()}

    return move_weights
