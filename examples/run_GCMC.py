import argparse
import os
import sys

# Hide UserWarning and RuntimeWarning messages
import warnings

import ase
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

parser = argparse.ArgumentParser(description="Run GCMC-MD simulation with MACE.")
# Required arguments
parser.add_argument("output_path", type=str, help="Path to save the GCMC output files.")
parser.add_argument(
    "--FrameworkPath",
    type=str,
    required=True,
    help="Path to the framework structure file in any formate readable to ASE.",
)
parser.add_argument(
    "--AdsorbatePath",
    type=str,
    required=True,
    help="Path to the adsorbate structure file in any formate readable to ASE.",
)
parser.add_argument(
    "--Temperature",
    type=float,
    default=293.15,
    help="Temperature of the ideal reservoir in Kelvin (default: 293.15 K).",
)
parser.add_argument(
    "--PressureList",
    type=str,
    default="100000",
    help="Pressure list as csv for the ideal reservoir in Pa (Ex: 100000,200000,300000).",
)
# Optional arguments
parser.add_argument(
    "--MCSteps",
    type=int,
    default=30000,
    help="Number of Monte Carlo steps to perform (default: 30000).",
)
parser.add_argument(
    "--MDSteps",
    type=int,
    default=30000,
    help="Number of Molecular Dynamics steps to perform (default: 30000).",
)
parser.add_argument(
    "--debug",
    type=bool,
    default=False,
    action="store_true",
    help="Enable debug mode on the simulations.",
)
parser.add_argument(
    "--OptFramework",
    type=bool,
    default=False,
    action="store_true",
    help="Enable optimization of the framework structure.",
)
parser.add_argument(
    "--OptAdsorbate",
    type=bool,
    default=False,
    action="store_true",
    help="Enable optimization of the adsorbate structure.",
)
parser.add_argument(
    "--nProcs",
    type=int,
    default=9,
    help="Number of processes to use for the simulation for CPU (default: 9).",
)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    torch.set_num_threads(args.nProcs)

model = mace_mp(
    model="mace-dac-1.model",
    dispersion=True,
    damping="zero",
    dispersion_xc="pbe",
    default_dtype="float32",
    device=device,
)

# Load the framework structure
framework: ase.Atoms = read(args.FrameworkPath)  # type: ignore

if args.OptFramework:
    print("Optimizing framework structure...")
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
    framework = frameworkOpt
    framework.set_constraints(None)  # type: ignore

# Load the adsorbate structure
adsorbate: ase.Atoms = read(args.AdsorbatePath)  # type: ignore

if args.OptAdsorbate:
    print("Optimizing adsorbate structure...")
    resultsDict, adsorbateOpt = crystalOptmization(
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

    adsorbate = adsorbateOpt
    adsorbate.set_constraints(None)  # type: ignore

pressure_list = list(map(float, args.PressureList.split(",")))

for i, pressure in enumerate(pressure_list):

    print(
        f"Running GCMC simulation for pressure: {pressure:.2f} Pa at temperature: {args.Temperature:.2f} K"
    )

    gcmc = GCMC(
        model=model,
        framework_atoms=framework,
        adsorbate_atoms=adsorbate,
        temperature=args.Temperature,
        pressure=pressure,
        fugacity_coeff=1,
        device=device,
        vdw_radii=vdw_radii,
        debug=True,
        output_to_file=True,
    )

    gcmc.print_introduction()

    if pressure > 10:
        print("Trying to load previous state for continuation...")
        # Check if the output directory exists and load the last snapshot
        output_dir = f"results_{args.Temperature:.2f}_{pressure_list[i-1]:.2f}"

        if os.path.exists(os.path.join(output_dir, "GCMC_Trajectory.traj")):
            traj = Trajectory(os.path.join(output_dir, "GCMC_Trajectory.traj"))
            if len(traj) > 0:
                gcmc.load_state(traj[-1])  # type: ignore
                print(f"Loaded last snapshot from {output_dir}/GCMC_Trajectory.traj")

    for j in range(5):
        gcmc.run(args.MCSteps)
        gcmc.npt(nsteps=args.MDSteps, time_step=0.5)

    gcmc.run(args.MCSteps)
    gcmc.print_finish()
