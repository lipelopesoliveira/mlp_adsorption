import datetime
import itertools
import os
import platform
import sys
from typing import TextIO, Union

import ase
import numpy as np
from ase import units
from ase.calculators import calculator
from ase.io import Trajectory, write
from tqdm import tqdm

from mlp_adsorption import VERSION
from mlp_adsorption.utilities import random_position, vdw_overlap2


class Widom:
    def __init__(
        self,
        framework_atoms: ase.Atoms,
        adsorbate_atoms: ase.Atoms,
        temperature: float,
        model: calculator.Calculator,
        vdw_radii: np.ndarray,
        vdw_factor: float = 0.6,
        device: str = "cpu",
        save_frequency: int = 100,
        output_to_file: bool = True,
        debug: bool = False,
    ) -> None:
        """
        Base class for Widom insertion method using ASE.

        This class performs the Widom insertion method to calculate the enthalpy of adsorption
        and Henry coefficient of an adsorbate in a framework.

        Currently, it supports only one adsorbate molecule and any ASE-compatible calculator
        for energy calculations.

        Parameters
        ----------
        framework_atoms : ase.Atoms
            The empty framework structure where the adsorbate will be inserted.
        adsorbate_atoms : ase.Atoms
            The adsorbate molecule to be inserted into the framework.
        temperature : float
            Temperature of the ideal reservoir in Kelvin.
        model : ase.calculators.calculator.Calculator
            ASE-compatible calculator for energy calculations.
        vdw_radii : np.ndarray
            Van der Waals radii of the atoms in the framework and adsorbate.
        vdw_factor : float, optional
            Factor to scale the Van der Waals radii (default is 0.6).
        device : str, optional
            Device to run the calculations on, either 'cpu' or 'cuda'. Default is 'cpu'.
        """

        self.start_time = datetime.datetime.now()

        os.makedirs(f"results_{temperature:.2f}_0.0", exist_ok=True)
        os.makedirs(f"results_{temperature:.2f}_0.0/Movies", exist_ok=True)

        self.out_folder = f"results_{temperature:.2f}_0.0"

        if output_to_file:
            self.out_file: Union[TextIO, None] = open(
                os.path.join(self.out_folder, "Widom_Output.out"), "a"
            )
        else:
            self.out_file = None

        self.trajectory = Trajectory(os.path.join(self.out_folder, "Widom_Trajectory.traj"), "a")  # type: ignore

        self.debug: bool = debug
        self.save_every: int = save_frequency

        # Framework setup
        self.framework = framework_atoms
        self.framework.calc = model
        self.framework_energy = self.framework.get_potential_energy()
        self.n_atoms_framework = len(self.framework)
        self.cell: np.ndarray = np.array(self.framework.get_cell())
        self.V: float = np.linalg.det(self.cell) / units.m**3  # Convert to m^3
        self.framework_mass: float = np.sum(self.framework.get_masses()) / units.kg

        self.density: float = self.framework_mass / self.V  # kg/m^3

        # Adsorbate setup
        self.adsorbate = adsorbate_atoms
        # self.adsorbate.set_cell(self.cell, scale_atoms=False)
        self.adsorbate.calc = model
        self.adsorbate_energy = self.adsorbate.get_potential_energy()

        self.n_ads: int = len(self.adsorbate)
        self.adsorbate_mass: float = np.sum(self.adsorbate.get_masses()) / units.kg

        self.T: float = temperature
        self.model = model
        self.beta: float = 1 / (units.kB * temperature)
        self.device = device
        self.vdw: np.ndarray = vdw_radii * vdw_factor  # Adjust van der Waals radii to avoid overlap

        # Replace any NaN value by 1.5 on self.vdw to avoid potential problems
        self.vdw[np.isnan(self.vdw)] = 1.5

        self.energy_list = np.zeros(100, dtype=float)

        self.minimum_configuration: ase.Atoms = self.framework.copy()
        self.minimum_energy: float = 0

    def print_header(self):
        """
        Print the header for the simulation output.
        This method is called at the beginning of the simulation to display the initial parameters.
        """

        header = f"""
===========================================================================
                      Grand Canonical Monte Carlo Simulation
                            powered by Python + ase
                        Author: Felipe Lopes de Oliveira
===========================================================================

Code version: {VERSION}
Simulation started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
Hostname: {platform.node()}
OS type: {platform.system()}
OS release: {platform.release()}
OS version: {platform.version()}

Python version: {sys.version.split()[0]}
Numpy version: {np.__version__}
ASE version: {ase.__version__}

Current directory: {os.getcwd()}

Model: {self.model.name}
Running on device: {self.device}

===========================================================================

Constants used:
Boltzmann constant:     {units.kB} eV/K
Beta (1/kT):            {self.beta:.3f} eV^-1

===========================================================================

Simulation Parameters:
Temperature: {self.T} K

===========================================================================

System Information:
Framework: {self.framework.get_chemical_formula()}
Framework: {self.n_atoms_framework} atoms,
Framework mass: {np.sum(self.framework.get_masses())} g/mol, {self.framework_mass} kg
Framework energy: {self.framework_energy} eV
Framework volume: {self.V} m^3
Framework density: {self.density} kg/m^3
Framework cell:
    {self.cell[0, 0]:12.7f} {self.cell[0, 1]:12.7f} {self.cell[0, 2]:12.7f}
    {self.cell[1, 0]:12.7f} {self.cell[1, 1]:12.7f} {self.cell[1, 2]:12.7f}
    {self.cell[2, 0]:12.7f} {self.cell[2, 1]:12.7f} {self.cell[2, 2]:12.7f}

Atomic positions:
"""
        for atom in self.framework:
            header += "  {:2} {:12.7f} {:12.7f} {:12.7f}\n".format(atom.symbol, *atom.position)  # type: ignore
        header += f"""
===========================================================================
Adsorbate: {self.adsorbate.get_chemical_formula()}
Adsorbate: {self.n_ads} atoms, {self.adsorbate_mass} kg
Adsorbate energy: {self.adsorbate_energy} eV

Atomic positions:
"""
        for atom in self.adsorbate:
            header += "  {:2} {:12.7f} {:12.7f} {:12.7f}\n".format(atom.symbol, *atom.position)  # type: ignore

        header += """
===========================================================================
Shortest distances:
"""
        atomic_numbers = set(
            list(self.framework.get_atomic_numbers()) + list(self.adsorbate.get_atomic_numbers())
        )

        for i, j in list(itertools.combinations(atomic_numbers, 2)):
            header += f"  {ase.Atom(i).symbol:2} - {ase.Atom(j).symbol:2}: {self.vdw[i] + self.vdw[j]:.3f} A\n"

        header += """
===========================================================================
"""
        print(header, file=self.out_file, flush=True)

    def print_footer(self):
        """
        Print the footer for the simulation output.
        This method is called at the end of the simulation to display the final results and elapsed time.
        """

        boltz_fac = np.exp(-self.beta * self.energy_list)

        # kH = β <exp(-β ΔE)> [mol kg-1 Pa-1]
        kH = self.beta * boltz_fac.mean() * (units.J / units.mol) / self.density

        # Qst = - < ΔE * exp(-β ΔE) > / <exp(-β ΔE)>  + kB.T # [kJ/mol]
        Qst = (self.energy_list * boltz_fac).mean() / boltz_fac.mean() - (units.kB * self.T)
        Qst /= units.kJ / units.mol

        footer = """
===========================================================================

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Finishing simulation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    Average properties of the system:
    ------------------------------------------------------------------------------
    Henry coefficient: [mol/kg/Pa]                       {:12.5e} +/- {:12.5e} [-]
    Enthalpy of adsorption: [kJ/mol]                     {:12.5f} +/- {:12.5f} [-]

===========================================================================
Simulation finished suscessfully!
===========================================================================

Simulation finished at {}
Simulation duration: {}

===========================================================================

""".format(
            kH,
            0.0,
            Qst,
            0.0,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            datetime.datetime.now() - self.start_time,
        )
        print(footer, file=self.out_file, flush=True)

    def debug_movement(self, movement, deltaE, prefactor, acc, rnd_number) -> None:
        """
        Print debug information about the current state of the simulation.
        This method is called to provide detailed information about the current state of the system.
        """
        print(
            f"""
=======================================================================================================
Movement type: {movement}
Interaction energy: {deltaE} eV, {(deltaE / (units.kJ / units.mol))} kJ/mol
Exponential factor:     {-self.beta * deltaE:.3E}
Exponential:            {np.exp(-self.beta * deltaE):.3E}
Prefactor:              {prefactor:.3E}
Acceptance probability: {acc:.3f}
Random number:          {rnd_number:.3f}
Accepted: {rnd_number < acc}
=======================================================================================================
""",
            file=self.out_file,
            flush=True,
        )

    def try_insertion(self):
        """
        Try to insert a new adsorbate molecule into the framework.
        This method randomly places the adsorbate in the framework and checks for van der Waals overlap.
        If there is no overlap, it calculates the new potential energy and decides whether to accept the insertion
        based on the acceptance criteria.

        Returns
        -------
        bool
            True if the insertion was accepted, False otherwise.
        """

        atoms_trial = self.framework.copy() + self.adsorbate.copy()
        atoms_trial.calc = self.model

        pos = atoms_trial.get_positions()
        pos[-self.n_ads :] = random_position(pos[-self.n_ads :], atoms_trial.get_cell())
        atoms_trial.set_positions(pos)
        atoms_trial.wrap()

        if vdw_overlap2(atoms_trial, self.vdw, self.n_ads):
            return 1000, atoms_trial  # Return 1000 energy to indicate overlap

        atoms_trial.calc = self.model
        e_new = atoms_trial.get_potential_energy()

        deltaE = e_new - self.framework_energy - self.adsorbate_energy

        return deltaE, atoms_trial

    def run(self, N) -> None:

        self.energy_list = np.zeros(N, dtype=float)

        header = """
Iteration  |  dE (eV)  |  dE (kJ/mol)  | kH [mol kg-1 Pa-1]  |  dH (kJ/mol) | Time (s)
---------------------------------------------------------------------------------------"""
        print(header, file=self.out_file, flush=True)

        for i in tqdm(range(1, N + 1), disable=(self.out_file is None), desc="Widom Step"):

            step_time_start = datetime.datetime.now()

            accepted = False

            deltaE = 0.0

            atoms_trial = self.framework.copy()

            insert_iter = 0

            while not accepted:
                insert_iter += 1
                deltaE, atoms_trial = self.try_insertion()
                if deltaE < 1000 or insert_iter > 100:
                    accepted = True

            if deltaE < self.minimum_energy:
                self.minimum_configuration = atoms_trial.copy()
                self.minimum_energy = deltaE

                write(
                    "results_{:.2f}_0.0/Movies/minimum_configuration_{:.2f}.cif".format(
                        self.T, deltaE / (units.kJ / units.mol)
                    ),
                    atoms_trial,
                    format="cif",
                )

            self.trajectory.write(atoms_trial)  # type: ignore

            self.energy_list[i - 1] = deltaE

            boltz_fac = np.exp(-self.beta * self.energy_list)

            # kH = β <exp(-β ΔE)> [mol kg-1 Pa-1]
            kH = self.beta * boltz_fac[:i].mean() * (units.J / units.mol) / self.density

            # Qst = - < ΔE * exp(-β ΔE) > / <exp(-β ΔE)>  + kB.T # [kJ/mol]
            Qst = (self.energy_list[:i] * boltz_fac[:i]).mean() / boltz_fac[:i].mean() - (
                units.kB * self.T
            )
            Qst /= units.kJ / units.mol

            print(
                "{:^10} | {:^9.6f} | {:>13.2f} | {:>19.3e} | {:12.2f} | {:8.2f}".format(
                    i,
                    deltaE,
                    deltaE / (units.kJ / units.mol),
                    kH,
                    Qst,
                    (datetime.datetime.now() - step_time_start).total_seconds(),
                ),
                file=self.out_file,
                flush=True,
            )
