import os
import sys

# Hide UserWarning and RuntimeWarning messages
import warnings

import ase
import numpy as np
import torch
from ase.data import vdw_radii
from ase.io import Trajectory, read
from ase.optimize import LBFGS
from mace.calculators import mace_mp

sys.path.append("/home/felipe/PRs/mlp_adsorption/")

from mlp_adsorption.ase_utils import crystalOptmization
from mlp_adsorption.gcmc import GCMC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "cuda" if torch.cuda.is_available() else "cpu"

FrameworkPath = "mg-mof-74.cif"
AdsorbatePath = "co2.xyz"

model = mace_mp(
    model="medium-0b2",
    dispersion=True,
    damping="zero",  # choices: ["zero", "bj", "zerom", "bjm"]
    dispersion_xc="pbe",
    default_dtype="float32",
    device=device,
)

# Load the framework structure
framework: ase.Atoms = read(FrameworkPath)  # type: ignore

# Load the adsorbate structure
adsorbate: ase.Atoms = read(AdsorbatePath)  # type: ignore

Temperature = 298.0  # in Kelvin
pressure = 100_000  # in Pa = 1 bar
MCSteps = 3_000
MDSteps = 3_000

print(
    f"Running GCMC simulation for pressure: {pressure:.2f} Pa at temperature: {Temperature:.2f} K"
)

gcmc = GCMC(
    model=model,
    framework_atoms=framework,
    adsorbate_atoms=adsorbate,
    temperature=Temperature,
    pressure=pressure,
    fugacity_coeff=1,
    device=device,
    vdw_radii=vdw_radii,
    vdw_factor=0.6,
    save_frequency=1,
    debug=True,
    output_to_file=True,
)


gcmc.print_introduction()

for j in range(5):
    gcmc.run(MCSteps)
    gcmc.npt(nsteps=MDSteps, time_step=0.5, mode="aniso_flex")

gcmc.run(MCSteps)
gcmc.print_finish()
