import os
import argparse
import torch
import sys
import ase
from ase.io import read
from ase.data import vdw_radii

from mace.calculators import mace_mp

# Hide UserWarning and RuntimeWarning messages
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


parser = argparse.ArgumentParser(description="Run GCMC-MD simulation with MACE.")
# Required arguments
parser.add_argument('output_path',
                    type=str,
                    help='Path to save the GCMC output files.')
parser.add_argument('--FrameworkPath',
                    type=str,
                    required=True,
                    help='Path to the framework structure file in any formate readable to ASE.')
parser.add_argument('--AdsorbatePath',
                    type=str,
                    required=True,
                    help='Path to the adsorbate structure file in any formate readable to ASE.')
parser.add_argument('--Temperature',
                    type=float,
                    default=293.15,
                    help='Temperature of the ideal reservoir in Kelvin (default: 293.15 K).')
parser.add_argument('--Pressure',
                    type=float,
                    default=1e5,
                    help='Pressure of the ideal reservoir in Pa (default: 100000 Pa = 1 bar).')
# Optional arguments
parser.add_argument('--MCSteps',
                    type=int,
                    default=30000,
                    help='Number of Monte Carlo steps to perform (default: 30000).')
parser.add_argument('--nProcs',
                    type=int,
                    default=9,
                    help='Number of processes to use for the simulation for CPU (default: 9).')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    torch.set_num_threads(args.nProcs)

model = mace_mp(model="mace-dac-1.model",
                dispersion=True,
                damping='zero',
                dispersion_xc='pbe',
                default_dtype="float32",
                device=device)

# Load the framework structure
framework: ase.Atoms = read(args.FrameworkPath)  # type: ignore

# Load the adsorbate structure
adsorbate: ase.Atoms = read(args.AdsorbatePath)  # type: ignore

# vdw_radii = np.array([1.0 for _ in vdw_radii])

# eos = PREOS.from_name('carbondioxide')
# fugacity = eos.calculate_fugacity(T, P)
pressure_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 500000, 750000, 1000000]

for i, pressure in enumerate(pressure_list):

    i += 6

    args.Pressure = pressure_list[i]

    print(f"Running GCMC simulation for pressure: {args.Pressure:.2f} Pa at temperature: {args.Temperature:.2f} K")

    gcmc = GCMC_ase(model=model,
                    framework_atoms=framework,
                    adsorbate_atoms=adsorbate,
                    temperature=args.Temperature,
                    pressure=args.Pressure,
                    fugacity_coeff=1,
                    device=device,
                    vdw_radii=vdw_radii,
                    debug=True,
                    output_to_file=True)

    gcmc.print_introduction()

    if args.Pressure > 10:
        print("Loading previous state for continuation...")
        snapshots = os.listdir(f'results_{args.Temperature:.2f}_{pressure_list[i-1]:.2f}/Movies/')
        last_snapshot = max(int(float(s.split('_')[1])) for s in snapshots if s.startswith('snapshot_') and s.endswith('.xyz'))
        print(f"Last snapshot found: {last_snapshot} for pressure {pressure_list[i-1]:.2f} Pa")
        # Load the previous state if not the first iteration
        gcmc.load_state(f'results_{args.Temperature:.2f}_{pressure_list[i-1]:.2f}/Movies/snapshot_{last_snapshot}_{pressure_list[i-1]:.2f}_{args.Temperature:.2f}.xyz')

    gcmc.run(args.MCSteps)
    gcmc.print_finish()
