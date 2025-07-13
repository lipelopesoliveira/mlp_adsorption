import os
import sys
import datetime
import numpy as np

from typing import TextIO

import ase
from ase import units
from ase import Atoms
from ase.optimize.optimize import Optimizer
from ase.calculators.calculator import Calculator
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io.trajectory import Trajectory
from ase.spacegroup.symmetrize import check_symmetry

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md import MDLogger


def crystalOptmization(
        atoms_in: Atoms,
        calculator: Calculator,
        optimizer: Optimizer,
        out_file: TextIO,
        fmax: float = 0.005,
        opt_cell: bool = True,
        fix_symmetry: bool = False,
        hydrostatic_strain: bool = False,
        constant_volume: bool = False,
        scalar_pressure: float = 0.0,
        max_steps: int = 1000,
        trajectory: bool | str = "opt.traj",
        verbose: bool = True,
        symm_tol=1e-3
        ) -> tuple[dict, Atoms]:
    """
    Optimize the cell and positions of the atoms with the given calculator.
    If fix_symmetry is True, the symmetry of the cell is fixed during the optimization.
    If hydrostatic_strain is True, the cell is allowed to change only isotropically.

    Helpful conversion table for fmax:

        - 0.05 eV/A^3 = 8 GPA = 8000 bar
        - 0.003 eV/A^3 = 0.48 GPa = 480 bar
        - 0.0006 eV/A^3 = 0.096 GPa = 96 bar
        - 0.0003 eV/A^3 = 0.048 GPa = 48 bar
        - 0.0001 eV/A^3 = 0.02 GPa = 20 bar


    Parameters
    ----------

    atoms_in: Atoms
        The atoms object to optimize.
    calculator: Calculator
        The calculator to use for the optimization.
    optimizer: Optimizer
        The optimizer to use for the optimization. Recommended: lBFGS.
    fmax: float
        The maximum force allowed during the optimization. Default: 0.005 eV/Ang.
    opt_cell: bool
        If True, the cell is optimized during the optimization.
    fix_symmetry: bool
        If True, the symmetry of the cell is fixed during the optimization.
    hydrostatic_strain: bool
        If True, the cell is allowed to change only isotropically.
    constant_volume: bool
        If True, the volume of the cell is fixed during the optimization.
    scalar_pressure: float
        The pressure to use during the optimization. Default is 0.0 GPa.
    max_steps: int
        The maximum number of steps to run the optimization for. Default is 1000.
    trajectory: str
        The name of the trajectory file to write the optimization steps to.
    verbose: bool
        If True, print the optimization steps to the console during the optimization.
    symm_tol: float
        The tolerance to use when checking the symmetry of the cell. Default is 1e-5.

    Returns
    -------

    resultsDict: dict
        A dictionary containing the results of the optimization, including:
    atoms: Aoms
        The optimized atoms object.
    """

    atoms = atoms_in.copy()

    atoms.calc = calculator

    if fix_symmetry:
        atoms.set_constraint([FixSymmetry(atoms)])

    if opt_cell:
        ecf = FrechetCellFilter(
            atoms,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
            scalar_pressure=scalar_pressure)

        opt = optimizer(ecf, logfile=None)  # type: ignore

    else:
        opt = optimizer(atoms, logfile=None)  # type: ignore

    opt_history = []

    start_time = datetime.datetime.now()

    def custom_ase_log():

        e_tot = atoms.get_potential_energy()
        if opt_cell:
            stress = atoms.get_stress(voigt=False)
            pressure = -1/3 * np.trace(stress)
        else:
            pressure = 0.0
            stress = np.zeros(6)

        forces = atoms.get_forces()
        max_force = np.linalg.norm(forces, axis=1).max()
        sum_force = np.sum(np.linalg.norm(forces, axis=1))
        rmsd_force = np.sqrt(np.mean(forces**2))

        opt_history.append({
            'cellParameters': atoms.cell.cellpar().tolist(),
            'cellMatrix': np.array(atoms.cell).tolist(),
            'atomTypes': atoms.get_chemical_symbols(),
            'cartCoordinates': atoms.get_positions().tolist(),
            'energy': e_tot,
            'forces': forces.tolist(),
            'stress': stress.tolist(),
            }
        )

        line_txt = "{:5} {:>18.8f}    {:>15.8f}    {:>15.8f}     {:>15.8f}    {:>18.8f}   {:>18.8f} {:>12.2f}"

        print(line_txt.format(
            len(opt_history),
            e_tot,
            max_force,
            sum_force,
            rmsd_force,
            pressure*1e5,
            atoms.get_volume() if opt_cell else 0.0,
            (datetime.datetime.now() - start_time).total_seconds()/60),
            file=out_file
            )

    opt.attach(custom_ase_log, interval=1)

    traj = Trajectory(trajectory, 'w', atoms)
    if trajectory:
        opt.attach(traj)

    headers = ["Step", "Energy (eV)", "Max Force (eV/A)", "Sum Force (eV/A)",
               "RMSD Force (eV/A)", "Pressure (bar)", "Volume (A3)", "Time (min)"]
    print("{:^5} {:^18}    {:^15}    {:^15}     {:^15}    {:^15}     {:^15}    {:^14}".format(*headers),
          file=out_file)

    opt.run(fmax=fmax, steps=max_steps)

    if trajectory:
        traj.close()

    print("Optimization finished. Total time: {:.2f} minutes".format(
        (datetime.datetime.now() - start_time).total_seconds()/60),
        file=out_file
        )

    print(f"Optimization {'' if opt.converged() else 'did not '}converged.")

    resultsDict = {
        'status': 'Finished',
        'optConverged': 'Yes' if opt.converged() else 'No',
        'warningList': [],
        'executionTime': {
            'unit': 's',
            'value': (datetime.datetime.now() - start_time).total_seconds()
        },
        'startTime': start_time.isoformat(),
        'endTime': datetime.datetime.now().isoformat(),
        'calculationResults': opt_history}

    # Print final information about the symmetry of the cell
    if opt_cell:
        symm = check_symmetry(atoms, symprec=symm_tol, verbose=False)

        if symm is not None:
            print("Symmetry information:", file=out_file)
            print("no      : ", symm.number, file=out_file)
            print("symbol  : ", symm.international, file=out_file)
            print("lattice : ", atoms.cell.get_bravais_lattice().longname, file=out_file)

        resultsDict['symmetryInformation'] = {
                'number': symm.number if symm else None,
                'international': symm.international if symm else None,
                'bravaisLattice': atoms.cell.get_bravais_lattice().longname
            }

    return resultsDict, atoms


def nVT_Berendsen(
        atoms: ase.Atoms,
        model: Calculator,
        temperature: float,
        pressure: float = 0.0,
        time_step: float = 0.5,
        num_md_steps: int = 1000000,
        output_interval: int = 100,
        movie_interval: int = 1,
        taut: float = 1.0,
        out_file: TextIO = sys.stdout
        ) -> ase.Atoms:
    """
    Run NVT molecular dynamics simulation using the Berendsen thermostat.

    The Berendsen thermostat is a deterministic type of velocity scaling method that
    adjusts the velocities of the atoms to maintain a target temperature of the entire system.
    It is not a true canonical ensemble method, but it is often used for its simplicity and speed.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure to simulate.
    temperature : float
        The target temperature in Kelvin.
    pressure : float, optional
        The target pressure in Pa (default is 0.0 Pa). Not used in NVT simulations!
    time_step : float, optional
        The time step for the simulation in femtoseconds (default is 0.5 fs).
    num_md_steps : int, optional
        The total number of MD steps to run (default is 1,000,000).
    output_interval : int, optional
        The interval for logging output (default is 100 steps).
    movie_interval : int, optional
        The interval for saving trajectory frames (default is 1 step).
    taut : float, optional
        The time constant for the Berendsen thermostat in femtoseconds (default is 1.0 fs).

    Returns
    -------
    ase.Atoms
        The final atomic structure after the MD simulation.
    """

    atoms.calc = model

    print("""
===========================================================================
  Starting NVT Molecular Dynamics Simulation using Berendsen Thermostat

    Temperature: {:.2f} K
    Pressure: {:.2f} Pa (Not used in NVT!)
    Time Step: {:.2f} fs
    Number of MD Steps: {}
    Output Interval: {} steps
    Movie Interval: {} steps
    Time Constant (taut): {:.2f} fs

===========================================================================
""".format(
        temperature, pressure, time_step, num_md_steps, output_interval, movie_interval, taut), file=out_file)

    existing_md_traj = [i for i in os.listdir('.') if i.startswith("NVT-Berendsen") and i.endswith(".traj")]
    traj_filename = f"NVT-Berendsen_{temperature:.2f}K_{len(existing_md_traj)}.traj"
    log_filename = f"NVT-Berendsen_{temperature:.2f}K_{len(existing_md_traj)}.log"

    # Set the momenta corresponding to the given "temperature"
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)
    # Set zero total momentum to avoid drifting
    Stationary(atoms)

    # run Berendsen MD
    dyn = NVTBerendsen(
        atoms=atoms,
        timestep=time_step*units.fs,
        temperature_K=temperature,
        taut=taut*units.fs,
        loginterval=output_interval,
        trajectory=traj_filename
        )

    # Print statements
    def print_md_log() -> None:
        step = dyn.get_number_of_steps()
        etot = atoms.get_total_energy()
        temp_K = atoms.get_temperature()
        stress = atoms.get_stress(include_ideal_gas=True) / units.GPa
        stress_ave = (stress[0] + stress[1] + stress[2]) / 3.0
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        print("  {:>7}  | {:13.6f}  |  {:11.3f}  |  {:7.2f} | {:9.1f}".format(
            step, etot, temp_K, stress_ave, elapsed_time
            ), file=out_file)

    dyn.attach(print_md_log, interval=output_interval)
    dyn.attach(
        MDLogger(dyn, atoms, log_filename, header=True, stress=True, peratom=True, mode="a"),
        interval=movie_interval
        )

    # Now run the dynamics
    start_time = datetime.datetime.now()
    print("    Step   |  Total Energy  |  Temperature  |  Stress  | Elapsed Time ", file=out_file)
    print("    [-]    |      [eV]      |      [K]      |   [GPa]  |     [s]      ", file=out_file)
    print(" --------- | -------------- | ------------- | -------- | -------------", file=out_file)

    dyn.run(num_md_steps)

    print("=========================================================================", file=out_file)
    print("NVT MD simulation completed.", file=out_file)
    print("Final structure saved to:", traj_filename, file=out_file)
    print("Log file saved to:", log_filename, file=out_file)
    print("Total simulation time: {:.2f} seconds".format(
        (datetime.datetime.now() - start_time).total_seconds()), file=out_file)
    print("=========================================================================", file=out_file)

    return atoms


def nPT_Berendsen(
        atoms: ase.Atoms,
        model: Calculator,
        temperature: float,
        pressure: float = 0.0,
        compressibility: float = 1e-3,
        time_step: float = 0.5,
        num_md_steps: int = 1000000,
        output_interval: int = 100,
        movie_interval: int = 1,
        taut: float = 10.0,
        taup: float = 500.0,
        out_file: TextIO = sys.stdout
        ) -> ase.Atoms:
    """
    Run NPT molecular dynamics simulation using the Berendsen thermostat and barostat.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure to simulate.
    temperature : float
        The target temperature in Kelvin.
    pressure : float, optional
        The desired pressure, in bar (1 bar = 1e5 Pa).
    compressibility : float, optional
        The compressibility of the material, in bar-1.
    time_step : float, optional
        The time step for the simulation in femtoseconds (default is 0.5 fs).
    num_md_steps : int, optional
        The total number of MD steps to run (default is 1,000,000).
    output_interval : int, optional
        The interval for logging output (default is 100 steps).
    movie_interval : int, optional
        The interval for saving trajectory frames (default is 1 step).
    taut : float, optional
        The time constant for the Berendsen thermostat in femtoseconds (default is 10.0 fs).
    taup : float, optional
        The time constant for the Berendsen barostat in femtoseconds (default is 500.0 fs).

    Returns
    -------
    ase.Atoms
        The final atomic structure after the MD simulation.
    """

    atoms.calc = model

    print("""===========================================================================
    Starting NPT Molecular Dynamics Simulation using Berendsen Thermostat/Barostat
    Temperature: {:.2f} K
    Pressure: {:.2f} Pa (Not used in NVT!)
    Time Step: {:.2f} fs
    Number of MD Steps: {}
    Output Interval: {} steps
    Movie Interval: {} steps
    Time Constant (taut): {:.2f} fs
===========================================================================""".format(
        temperature, pressure, time_step, num_md_steps, output_interval, movie_interval, taut), file=out_file)

    existing_md_traj = [i for i in os.listdir('.') if i.startswith("NPT-Berendsen") and i.endswith(".traj")]
    traj_filename = f"NPT-Berendsen_{temperature:.2f}K_{len(existing_md_traj)}.traj"
    log_filename = f"NPT-Berendsen_{temperature:.2f}K_{len(existing_md_traj)}.log"

    # Set the momenta corresponding to the given "temperature"
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)
    # Set zero total momentum to avoid drifting
    Stationary(atoms)

    # run Berendsen MD
    dyn = NPTBerendsen(
        atoms=atoms,
        timestep=time_step*units.fs,
        temperature_K=temperature,
        pressure=pressure,
        compressibility_au=compressibility / (1e5 * units.Pascal),
        taut=taut * units.fs,
        taup=taup * units.fs,
        loginterval=movie_interval,
        trajectory=traj_filename
        )

    # Print statements
    def print_md_log():
        step = dyn.get_number_of_steps()
        etot = atoms.get_total_energy()
        temp_K = atoms.get_temperature()
        stress = atoms.get_stress(include_ideal_gas=True) / units.GPa
        stress_ave = (stress[0] + stress[1] + stress[2]) / 3.0
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        print("  {:>7}  | {:13.6f}  |  {:11.3f}  |  {:7.2f} | {:9.1f}".format(
            step, etot, temp_K, stress_ave, elapsed_time),
            file=out_file)

    dyn.attach(print_md_log, interval=output_interval)
    dyn.attach(
        MDLogger(dyn, atoms, log_filename, header=True, stress=True, peratom=True, mode="a"),
        interval=movie_interval
        )

    # Now run the dynamics
    start_time = datetime.datetime.now()
    print("    Step   |  Total Energy  |  Temperature  |  Stress  | Elapsed Time", file=out_file)
    print("    [-]    |      [eV]      |      [K]      |   [GPa]  |     [s]", file=out_file)
    print(" --------- | -------------- | ------------- | -------- | -------------", file=out_file)

    dyn.run(num_md_steps)

    print("=========================================================================", file=out_file)
    print("NPT MD simulation completed.", file=out_file)
    print("Final structure saved to:", traj_filename, file=out_file)
    print("Log file saved to:", log_filename, file=out_file)
    print("=========================================================================", file=out_file)

    return atoms
