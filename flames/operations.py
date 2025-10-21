import ase
import numpy as np
from flames.ase_utils import unwrap_positions
from scipy.spatial.transform import Rotation


def random_rotation(
    original_position: np.ndarray, cell: np.ndarray, rnd_generator: np.random.Generator
) -> np.ndarray:
    """
    Generates a random rotation of the original position vector around its geometrical center
    using a provided generator.

    Parameters
    ----------
    original_position (np.ndarray):
        The original position of the atom or molecule to be rotated as a 3D vector.
        Can be a single point (shape `(3,)`) or multiple points (shape `(N, 3)`).
    cell (np.ndarray):
        Unit cell used to unwrap the atomic positions and perform the rotation correctly
    rnd_generator (np.random.Generator):
        A random number generator instance for reproducibility.

    Returns
    ----------
        np.ndarray:
            A 3D vector or array of vectors representing the rotated position(s).
    """
    # 1. Unwap the molecule positions to perform the rotation correctly
    unrwap_pos = unwrap_positions(positions=original_position, cell=cell)

    # 2. Calculate the geometric center (centroid) of the points.
    center = np.mean(unrwap_pos, axis=0)

    # 3. Translate the points so their center is at the origin (0, 0, 0).
    # Rotation is always performed around the origin.
    centered_points = np.array(unrwap_pos) - center

    # 4. Generate a uniform random rotation in 3D space.
    #    Pass the provided generator to the 'random_state' parameter.
    random_rot = Rotation.random(rng=rnd_generator)

    # 5. Apply the random rotation to the centered points.
    rotated_centered_points = random_rot.apply(centered_points)

    # 6. Translate the rotated points back to their original center.
    rotated_points = rotated_centered_points + center

    return rotated_points


def random_rotation_limited(
    original_position: np.ndarray,
    cell: np.ndarray,
    rnd_generator: np.random.Generator,
    theta_max: float,
) -> np.ndarray:
    """
    Generates a random rotation of the molecule around a random axis,
    with rotation angle limited between -theta_max and theta_max (in radians).

    Parameters
    ----------
    original_position : np.ndarray
        Coordinates of shape (N, 3) or (3,) representing atoms in space.
    cell (np.ndarray):
        Unit cell used to unwrap the atomic positions and perform the rotation correctly
    rnd_generator : np.random.Generator
        Random number generator for reproducibility.
    theta_max : float
        Maximum rotation angle (in radians).

    Returns
    -------
    np.ndarray
        Rotated coordinates with the same shape as input.
    """
    # 1. Unwap the molecule positions to perform the rotation correctly
    unrwap_pos = unwrap_positions(positions=original_position, cell=cell)

    # Compute geometric center
    center = np.mean(unrwap_pos, axis=0)

    # Center coordinates at origin
    centered_points = np.array(unrwap_pos) - center

    # --- Generate random axis uniformly on the unit sphere ---
    axis = rnd_generator.normal(size=3)
    axis /= np.linalg.norm(axis)

    # --- Generate random angle in [-theta_max, theta_max] ---
    angle = rnd_generator.uniform(-theta_max, theta_max)

    # --- Create rotation object from axis-angle representation ---
    rot = Rotation.from_rotvec(axis * angle)

    # Apply rotation
    rotated_points = rot.apply(centered_points) + center

    return rotated_points


def random_translation(
    original_position: np.ndarray,
    cell: np.ndarray,
    max_translation: float,
    rnd_generator: np.random.Generator,
) -> np.ndarray:
    """
    Generates a random translation vector for the original positions on the interval
    [-max_translation/2, max_translation/2] using a provided generator.

    Parameters
    ----------
    original_position (np.ndarray):
        The original positions of the atoms or molecules to be translated as a 3D vector.
        Can be a single point (shape `(3,)`) or multiple points (shape `(N, 3)`).
    cell (np.ndarray):
        Unit cell used to unwrap the atomic positions and perform the rotation correctly
    max_shift (float):
        The maximum shift for the translation.
    rnd_generator (np.random.Generator):
        A random number generator instance for reproducibility.

    Returns
    ----------
        np.ndarray:
            A 3D vector or array of vectors representing the translated position(s).
    """
    # 1. Generate random translation vectors on the interval [-0.5, 0.5].
    translation_vectors = rnd_generator.uniform(
        -max_translation / 2, max_translation / 2, size=(1, 3)
    )

    # 2. Unrap the atomic positions
    unwrap_pos = unwrap_positions(positions=original_position, cell=cell)

    # 2. Apply the translation to the original positions.
    translated_positions = unwrap_pos + translation_vectors

    return translated_positions


def random_position_cell(
    original_position: np.ndarray, lattice_vectors: np.ndarray, rnd_generator: np.random.Generator
) -> np.ndarray:
    """
    Generates a random translation vector within the parallelepiped
    defined by the lattice vectors, using a specific seed for reproducibility.

    Parameters:
    ----------
    original_position (np.ndarray):
        The original position of the atom or molecule to be translated as a 3D vector.

    lattice_vectors (np.ndarray):
        A 3x3 matrix where each row is a lattice vector defining the unit cell.

    rnd_generator (np.random.Generator):
        A random number generator instance for reproducibility.

    Returns:
    ----------
        np.ndarray:
            A 3D random position inside the unit cell.
    """

    unwrap_pos = unwrap_positions(positions=original_position, cell=lattice_vectors)

    # Ensure original_position is a numpy array at the origin
    # Note: This line might not be necessary depending on your use case.
    # It centers the input `original_position` array before applying the translation.
    original_position = np.array(unwrap_pos) - np.average(unwrap_pos, axis=0)

    # 2. Use the 'rnd_generator' to generate random numbers
    random_fractions = rnd_generator.random(3)

    # Convert fractional coordinates to a Cartesian vector
    translation_vector = random_fractions @ lattice_vectors

    return unwrap_pos + translation_vector


def random_mol_insertion(
    framework: ase.Atoms, molecule: ase.Atoms, rnd_generator: np.random.Generator
) -> ase.Atoms:
    """
    Generates a random position within the unit cell defined by the lattice vectors.

    Parameters:
    ----------
    original_positions (np.ndarray):
        The original positions of the atoms or molecules to be translated as a 3D vector.
    lattice_vectors (np.ndarray):
        A 3x3 matrix where each row is a lattice vector defining the unit cell.
    rnd_generator (np.random.Generator):
        A random number generator instance for reproducibility.

    Returns:
    ----------
        np.ndarray: A 3D random position inside the unit cell
    """

    tmp_molecule = molecule.copy()

    tmp_molecule.set_positions(
        random_rotation(molecule.get_positions(), framework.cell.array, rnd_generator)
    )

    tmp_molecule.set_positions(
        random_position_cell(tmp_molecule.get_positions(), framework.cell.array, rnd_generator)
    )

    new_framework = framework.copy()
    new_framework += tmp_molecule
    new_framework.wrap()

    return new_framework


def check_overlap(
    atoms: ase.Atoms, group1_indices: np.ndarray, group2_indices: np.ndarray, vdw_radii: np.ndarray
) -> bool:
    """
    Checks for van der Waals overlap between two specified groups of atoms.

    This function is more efficient as it calculates a distance matrix between
    the two groups in a single call rather than looping.

    This function takes between 1 to 10 ms to evaluate the structure and has O(N) complexity.

    Parameters:
    ----------
        atoms (ase.Atoms):
            The ASE Atoms object containing the entire system.
        group1_indices (array_like):
            A list or array of indices for atoms in the first group.
        group2_indices (array_like):
            A list or array of indices for atoms in the second group.
        vdw_radii (array_like):
            A n array mapping atomic numbers to van der Waals radii.

    Returns:
        has_overlap (bool):
            True if any atom in group1 overlaps with an atom in group2, False otherwise.
    """
    # Get all necessary atomic numbers and vdW radii at once
    numbers = atoms.get_atomic_numbers()
    radii1 = np.array([vdw_radii[numbers[i]] for i in group1_indices])
    radii2 = np.array([vdw_radii[numbers[j]] for j in group2_indices])

    # Create a matrix of the required vdW sum for each pair
    # Each element (i, j) will be the sum of radii for atom i in group1 and atom j in group2
    vdw_sum_matrix = radii1[:, np.newaxis] + radii2

    # Get the distance matrix between the two groups in a single, efficient call
    distance_matrix = np.array(
        [atoms.get_distances(group1_indices, i, mic=True) for i in group2_indices]
    ).T

    # Check for any overlap using a fast vectorized comparison
    has_overlap: bool = np.any(distance_matrix < vdw_sum_matrix)  # type: ignore

    return has_overlap
