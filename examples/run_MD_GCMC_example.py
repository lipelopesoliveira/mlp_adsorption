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

from mlp_adsorption.ase_utils import crystalOptmization
from mlp_adsorption.gcmc import GCMC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append("C:\\Users\\flopes\\Documents\\PRs\\mlp_adsorption")

device = "cuda" if torch.cuda.is_available() else "cpu"

FrameworkPath = ""
AdsorbatePath = ""

model = mace_mp(
    model="mace-dac-1.model",
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

# Load the adsorbate structure
adsorbate: ase.Atoms = read(AdsorbatePath)  # type: ignore

resultsDict, frameworkOpt = crystalOptmization(
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

Temperature = 298.0
pressure_list = np.arange(10, 5000, 200).astype(float)  # Example pressure list in Pa
MCSteps = 30000
MDSteps = 30000

for i, pressure in enumerate(pressure_list):

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
        debug=True,
        output_to_file=True,
    )

    gcmc.print_introduction()

    if pressure > pressure_list[0]:
        print("Loading previous state for continuation...")
        output_dir = f"results_{Temperature:.2f}_{pressure_list[i-1]:.2f}"
        if os.path.exists(os.path.join(output_dir, "GCMC_Trajectory.traj")):
            traj = Trajectory(os.path.join(output_dir, "GCMC_Trajectory.traj"))
            if len(traj) > 0:
                gcmc.load_state(traj[-1])  # type: ignore
                print(f"Loaded last snapshot from {output_dir}/GCMC_Trajectory.traj")

    for j in range(5):
        gcmc.run(MCSteps)
        gcmc.npt(nsteps=MDSteps, time_step=0.5)

    gcmc.run(MCSteps)
    gcmc.print_finish()
