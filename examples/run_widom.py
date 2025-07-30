import os
import sys

import ase
import torch
from ase.data import vdw_radii
from ase.io import read
from ase.optimize import LBFGS
from mace.calculators import mace_mp

sys.path.append("C:\\Users\\flopes\\Documents\\PRs\\mlp_adsorption")

# Hide UserWarning and RuntimeWarning messages
import warnings

from mlp_adsorption.ase_utils import crystalOptmization
from mlp_adsorption.widom import Widom

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "cuda" if torch.cuda.is_available() else "cpu"

FrameworkPath = "cau-10.cif"
AdsorbatePath = "H2O.xyz"

model = mace_mp(
    model="CAU-10_pbe_10samples.model",
    dispersion=True,
    damping="zero",
    dispersion_xc="pbe",
    default_dtype="float32",
    device=device,
)


# Load the framework structure
framework: ase.Atoms = read(FrameworkPath)  # type: ignore

resultsDict, frameworkOpt = crystalOptmization(
    atoms_in=framework,
    calculator=model,
    optimizer=LBFGS,  # type: ignore
    out_file=sys.stdout,
    fmax=0.001,
    opt_cell=True,
    fix_symmetry=True,
    hydrostatic_strain=True,
    constant_volume=False,
    scalar_pressure=0.0,
    max_steps=1000,
    trajectory="FrameworkOpt.traj",
    verbose=False,
)

# Remove constrains from frameworkOpt
frameworkOpt.set_constraint(None)

# Load the adsorbate structure
adsorbate: ase.Atoms = read(AdsorbatePath)  # type: ignore

resultsDict, adsorbatekOpt = crystalOptmization(
    atoms_in=adsorbate,
    calculator=model,
    optimizer=LBFGS,  # type: ignore
    out_file=sys.stdout,
    fmax=0.001,
    opt_cell=False,
    fix_symmetry=False,
    hydrostatic_strain=False,
    constant_volume=False,
    scalar_pressure=0.0,
    max_steps=1000,
    trajectory="AdsorbateOpt.traj",
    verbose=False,
)

# Remove constrains from adsorbateOpt
adsorbatekOpt.set_constraint(None)

Temperature = 298.0

NSteps = 30000

widom = Widom(
    model=model,
    framework_atoms=frameworkOpt,
    adsorbate_atoms=adsorbatekOpt,
    temperature=Temperature,
    device=device,
    vdw_radii=vdw_radii,
    debug=False,
    output_to_file=True,
)

widom.print_introduction()

widom.run(NSteps)
widom.print_finish()
