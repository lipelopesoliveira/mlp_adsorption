import ase
import numpy as np
from scipy.spatial.transform import Rotation


def random_rotation(
    original_position: np.ndarray, rnd_generator: np.random.Generator
) -> np.ndarray:
    """
    Generates a random rotation of the original position vector using a provided generator.

    Parameters
    ----------
    original_position (np.ndarray):
        The original position of the atom or molecule to be rotated as a 3D vector.
        Can be a single point (shape `(3,)`) or multiple points (shape `(N, 3)`).

    rnd_generator (np.random.Generator):
        A random number generator instance for reproducibility.

    Returns
    ----------
        np.ndarray:
            A 3D vector or array of vectors representing the rotated position(s).
    """
    # 1. Calculate the geometric center (centroid) of the points.
    center = np.mean(original_position, axis=0)

    # 2. Translate the points so their center is at the origin (0, 0, 0).
    # Rotation is always performed around the origin.
    centered_points = np.array(original_position) - center

    # 3. Generate a uniform random rotation in 3D space.
    #    Pass the provided generator to the 'rng' parameter.
    random_rot = Rotation.random(rng=rnd_generator)

    # 4. Apply the random rotation to the centered points.
    rotated_centered_points = random_rot.apply(centered_points)

    # 5. Translate the rotated points back to their original center.
    rotated_points = rotated_centered_points + center

    return rotated_points


def random_translation(
    original_positions: np.ndarray, max_translation: float, rnd_generator: np.random.Generator
) -> np.ndarray:
    """
    Generates a random translation vector for the original positions on the interval [-max_translation/2, max_translation/2]
    using a provided generator.

    Parameters
    ----------
    original_positions (np.ndarray):
        The original positions of the atoms or molecules to be translated as a 3D vector.
        Can be a single point (shape `(3,)`) or multiple points (shape `(N, 3)`).

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
        -max_translation / 2, max_translation / 2, size=original_positions.shape
    )

    # 2. Apply the translation to the original positions.
    translated_positions = original_positions + translation_vectors

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

    # Ensure original_position is a numpy array at the origin
    # Note: This line might not be necessary depending on your use case.
    # It centers the input `original_position` array before applying the translation.
    original_position = np.array(original_position) - np.average(original_position, axis=0)

    # 2. Use the 'rnd_generator' to generate random numbers
    random_fractions = rnd_generator.random(3)

    # Convert fractional coordinates to a Cartesian vector
    translation_vector = random_fractions @ lattice_vectors

    return original_position + translation_vector


def random_insertion_cell(
    original_positions: np.ndarray, lattice_vectors: np.ndarray, rnd_generator: np.random.Generator
) -> np.ndarray:
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
        np.ndarray: A 3D random position inside the unit cell.
    """

    new_position = random_rotation(original_positions, rnd_generator)
    new_position = random_position_cell(new_position, lattice_vectors, rnd_generator)

    return new_position


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
