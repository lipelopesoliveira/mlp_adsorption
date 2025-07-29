# fmt: off

import numpy as np
import json

from ase.calculators.calculator import Calculator, all_changes
from ase import units


class CustomLennardJones(Calculator):
    """
    Custom Lennard Jones potential calculator based on the ASE calculator interface.
    This method is intended to be as close as possible to RASPA2 implementation.

    The fundamental definition of this potential is a pairwise energy:

    ``u_ij = 4 epsilon ( sigma^12/r_ij^12 - sigma^6/r_ij^6 )``

    For convenience, we'll use d_ij to refer to "distance vector" and
    ``r_ij`` to refer to "scalar distance". So, with position vectors `r_i`:

    ``r_ij = | r_j - r_i | = | d_ij |``

    Therefore:

    ``d r_ij / d d_ij = + d_ij / r_ij``
    ``d r_ij / d d_i  = - d_ij / r_ij``

    The derivative of u_ij is:

    ::

        d u_ij / d r_ij
        = (-24 epsilon / r_ij) ( 2 sigma^12/r_ij^12 - sigma^6/r_ij^6 )

    We can define a "pairwise force"

    ``f_ij = d u_ij / d d_ij = d u_ij / d r_ij * d_ij / r_ij``

    The terms in front of d_ij are combined into a "general derivative".

    ``du_ij = (d u_ij / d d_ij) / r_ij``

    We do this for convenience: `du_ij` is purely scalar The pairwise force is:

    ``f_ij = du_ij * d_ij``

    The total force on an atom is:

    ``f_i = sum_(j != i) f_ij``

    There is some freedom of choice in assigning atomic energies, i.e.
    choosing a way to partition the total energy into atomic contributions.

    We choose a symmetric approach (`bothways=True` in the neighbor list):

    ``u_i = 1/2 sum_(j != i) u_ij``

    The total energy of a system of atoms is then:

    ``u = sum_i u_i = 1/2 sum_(i, j != i) u_ij``

    Differentiating `u` with respect to `r_i` yields the force,
    independent of the choice of partitioning.

    ::

        f_i = - d u / d r_i = - sum_ij d u_ij / d r_i
            = - sum_ij d u_ij / d r_ij * d r_ij / d r_i
            = sum_ij du_ij d_ij = sum_ij f_ij

    This justifies calling `f_ij` pairwise forces.

    The stress can be written as ( `(x)` denoting outer product):

    ``sigma = 1/2 sum_(i, j != i) f_ij (x) d_ij = sum_i sigma_i ,``
    with atomic contributions

    ``sigma_i  = 1/2 sum_(j != i) f_ij (x) d_ij``

    Another consideration is the cutoff. We have to ensure that the potential
    goes to zero smoothly as an atom moves across the cutoff threshold,
    otherwise the potential is not continuous. In cases where the cutoff is
    so large that u_ij is very small at the cutoff this is automatically
    ensured, but in general, `u_ij(rc) != 0`.

    This implementation offers two ways to deal with this:

    Either, we shift the pairwise energy

    ``u'_ij = u_ij - u_ij(rc)``

    which ensures that it is precisely zero at the cutoff. However, this means
    that the energy effectively depends on the cutoff, which might lead to
    unexpected results! If this option is chosen, the forces discontinuously
    jump to zero at the cutoff.

    An alternative is to modify the pairwise potential by multiplying
    it with a cutoff function that goes from 1 to 0 between an onset radius
    ro and the cutoff rc. If the function is chosen suitably, it can also
    smoothly push the forces down to zero, ensuring continuous forces as well.
    In order for this to work well, the onset radius has to be set suitably,
    typically around 2*sigma.

    In this case, we introduce a modified pairwise potential:

    ``u'_ij = fc * u_ij``

    The pairwise forces have to be modified accordingly:

    ``f'_ij = fc * f_ij + fc' * u_ij``

    Where `fc' = d fc / d d_ij`.

    This approach is taken from Jax-MD (https://github.com/google/jax-md),
    which in turn is inspired by HOOMD Blue
    (https://glotzerlab.engin.umich.edu/hoomd-blue/).

    """

    implemented_properties = ['energy', 'energies']
    default_parameters = {
        'epsilon': 1.0,
        'sigma': 1.0,
        'rc': None,
        'ro': None,
        'smooth': False,
    }
    nolabel = True

    # with open('lj_params.json', 'r') as f:
    #    lj_params = json.load(f)

    def __init__(self, lj_parameters: dict, **kwargs):
        """
        Parameters
        ----------
        lj_parameters : dict
            Dictionary containing the Lennard-Jones parameters.
            The parameters should be in the form:
            "O": {
                "sigma": 3.03315,  # In Angstroms
                "epsilon": 48.1581 # In Kelvin
                }
        vdw_cutoff : float, optional
            Cutoff distance for the van der Waals interactions.
            Default is 12.0 Angstroms.
        """

        Calculator.__init__(self, **kwargs)

        self.lj_params: dict = lj_parameters
        self.vdw_cutoff = kwargs.get('vdw_cutoff', 12.0)

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        np.seterr(invalid='ignore')

        nAtoms = len(self.atoms)  # type: ignore

        # Preallocate arrays
        sigmas = np.empty((nAtoms, nAtoms))
        epsilons = np.empty((nAtoms, nAtoms))

        sigma_vec = np.array(
            [self.lj_params[s]['sigma'] for s in self.atoms.get_chemical_symbols()]  # type: ignore
            )
        epsilon_vec = np.array(
            [self.lj_params[s]['epsilon'] for s in self.atoms.get_chemical_symbols()]  # type: ignore
            )

        # Use broadcasting instead of loops
        sigmas = (sigma_vec[:, None] + sigma_vec[None, :]) / 2
        epsilons = np.sqrt(epsilon_vec[:, None] * epsilon_vec[None, :])

        rij = self.atoms.get_all_distances(mic=True)  # type: ignore

        # Replace all distances greater than the cutoff with 0
        rij[rij > self.vdw_cutoff] = 0

        energy = 4 * epsilons * ((sigmas / rij)**12 - (sigmas / rij)**6)

        # Replace any NaN values with 0
        energy[np.isnan(energy)] = 0.0

        # Sum the energy matrix and divide by 2 to avoid double counting since the energy matrix is symmetric
        energy /= 2

        # Convert from K to eV
        energy *= units.kB

        self.results['energy'] = energy.sum()
        self.results['energies'] = energy.sum(axis=1)
        self.results['free_energy'] = energy.sum()
