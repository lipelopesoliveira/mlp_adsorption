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
from ase.io import Trajectory, read
from ase.optimize import LBFGS
from tqdm import tqdm

from mlp_adsorption import VERSION
from mlp_adsorption.ase_utils import (
    crystalOptmization,
    nPT_Berendsen,
    nPT_NoseHoover,
    nVT_Berendsen,
)
from mlp_adsorption.utilities import (
    enthalpy_of_adsorption,
    random_position,
    random_rotation,
    vdw_overlap,
)


class GCMC:
    def __init__(
        self,
        model: calculator.Calculator,
        framework_atoms: ase.Atoms,
        adsorbate_atoms: ase.Atoms,
        temperature: float,
        pressure: float,
        fugacity_coeff: float,
        device: str,
        vdw_radii: np.ndarray,
        save_frequency: int = 100,
        output_to_file: bool = True,
        debug: bool = False,
    ):
        """
        Base class for Grand Canonical Monte Carlo (GCMC) simulations using ASE.

        This clas employs Monte Carlo simulations under the grand canonical ensemble (μVT) ensemble
        to study the adsorption of molecules in a framework material. It allows for movements such as
        insertion, deletion, translation, and rotation of adsorbate molecules within the framework.

        Currently, it supports any ASE-compatible calculator for energy calculations.

        Parameters
        ----------
        model : ase.calculators.Calculator
            The calculator to use for energy calculations. Can be any ASE-compatible calculator.
            The outpyt of the calculator should be in eV.
        framework_atoms : ase.Atoms
            The framework structure as an ASE Atoms object.
        adsorbate_atoms : ase.Atoms
            The adsorbate structure as an ASE Atoms object.
        temperature : float
            Temperature of the ideal reservoir in Kelvin.
        pressure : float
            Pressure of the ideal reservoir in Pascal.
        fugacity_coeff : float
            Fugacity coefficient to correct the pressure.
        device : str
            Device to run the simulation on, e.g., 'cpu' or 'cuda'.
        vdw_radii : np.ndarray
            Van der Waals radii for the atoms in the framework and adsorbate.
            Should be an array of the same length as the number of atomic numbers in ASE.
        save_frequency : int, optional
            Frequency at which to save the simulation state and results (default is 100).
        output_to_file : bool, optional
            If True, writes the output to a file named 'GCMC_Output.out' in the 'results' directory
        debug : bool, optional
            If True, prints detailed debug information during the simulation (default is False).
        """

        self.start_time = datetime.datetime.now()

        self.out_folder = f"results_{temperature:.2f}_{pressure:.2f}"
        os.makedirs(self.out_folder, exist_ok=True)
        os.makedirs(os.path.join(self.out_folder, "Movies"), exist_ok=True)

        if output_to_file:
            self.out_file: Union[TextIO, None] = open(
                os.path.join(self.out_folder, "GCMC_Output.out"), "a"
            )
        else:
            self.out_file = None

        self.model = model

        self.trajectory = Trajectory(
            os.path.join(self.out_folder, "GCMC_Trajectory.traj"),
            "a",
        )

        # Framework setup
        self.set_framework(framework_atoms)

        # Adsorbate setup
        self.set_adsorbate(adsorbate_atoms)

        # Simulation parameters
        self.T: float = temperature
        self.P: float = pressure
        self.fugacity_coeff: float = fugacity_coeff
        self.fugacity: float = (
            pressure * fugacity_coeff * units.J
        )  # Convert fugacity from Pa (J/m^3) to eV / m^3

        self.model = model
        self.device = device
        self.beta: float = 1 / (units.kB * temperature)  # Boltzmann weight, 1 / [eV/K * K]

        kB = 1.380649e-23  # m2 kg s-2 K-1 = J K-1
        h = 6.62607015e-34  # m2 kg / s = J s

        beta = 1 / (kB * self.T)  # J^-1
        self.lmbd = np.sqrt(h**2 * beta / (2 * np.pi * self.adsorbate_mass))  # unit: m

        self.save_every = save_frequency
        self.debug = debug

        # Constants
        R = 8.31446261815324  # Gas constant (J⋅mol^-1⋅K^-1) or (m^3⋅Pa⋅K^-1⋅mol^-1)

        self.mu_i = self.get_ideal_chemical_potential()  # Ideal chemical potential in eV

        atm2pa = 101325
        mol2cm3 = R * 273.15 / atm2pa

        self.conv_factors = {
            "mol/kg": (1 / units.mol) / self.framework_mass,
            "mg/g": (self.adsorbate_mass * 1e3) / self.framework_mass,
            "cm^3 STP/gr": mol2cm3 / units.mol / self.framework_mass * 1e3,
            "cm^3 STP/cm^3": 1e6 * mol2cm3 / units.mol / (self.framework.get_volume() * (1e-8**3)),
        }

        self.vdw: np.ndarray = vdw_radii * 0.6  # Adjust van der Waals radii to avoid overlap

        # Define the current state of the system that will be updated during the simulation
        self.current_system: ase.Atoms = self.framework.copy()
        self.current_system.calc = self.model
        self.current_total_energy: float = self.current_system.get_potential_energy()
        self.N_ads: int = 0

        # Store the main results during the simulation
        self.uptake_list: list[int] = []
        self.total_energy_list: list[float] = []
        self.total_ads_list: list[float] = []

        self.mov_dict: dict = {"insertion": [], "deletion": [], "translation": [], "rotation": []}

    def set_framework(self, framework_atoms: ase.Atoms) -> None:
        """
        Set the framework structure for the simulation.

        Parameters
        ----------
        framework_atoms : ase.Atoms
            The new framework structure as an ASE Atoms object.
        """
        self.framework = framework_atoms
        self.framework.calc = self.model
        self.framework_energy = self.framework.get_potential_energy()
        self.n_atoms_framework = len(self.framework)
        self.cell = np.array(self.framework.get_cell())
        self.V = np.linalg.det(self.cell) / units.m**3
        self.framework_mass = np.sum(self.framework.get_masses()) / units.kg

    def set_adsorbate(self, adsorbate_atoms: ase.Atoms) -> None:
        """
        Set the adsorbate structure for the simulation.

        Parameters
        ----------
        adsorbate_atoms : ase.Atoms
            The new adsorbate structure as an ASE Atoms object.
        """
        self.adsorbate = adsorbate_atoms
        self.adsorbate.calc = self.model
        self.adsorbate.set_cell(self.framework.get_cell())

        self.adsorbate_energy = self.adsorbate.get_potential_energy()
        self.n_ads = len(self.adsorbate)
        self.adsorbate_mass = np.sum(self.adsorbate.get_masses()) / units.kg

    def set_state(self, state: ase.Atoms) -> None:
        """
        Set the current state of the simulation.

        Parameters
        ----------
        state : ase.Atoms
            The current state of the simulation as an ASE Atoms object.
        """
        self.current_system = state.copy()
        self.current_system.calc = self.model
        self.current_total_energy = self.current_system.get_potential_energy()

    def load_state(self, state_file: str):
        """
        Load the state of the simulation from a file.

        Parameters
        ----------
        state_file : str
            Path to the file containing the saved state of the simulation.
        """
        print(f"Loading state from {state_file}...")
        state: ase.Atoms = read(state_file)  # type: ignore

        self.set_state(state)

        self.N_ads = int((len(state) - self.n_atoms_framework) / len(self.adsorbate))
        average_binding_energy = (
            (self.current_total_energy - self.framework_energy - self.N_ads * self.adsorbate_energy)
            / (units.kJ / units.mol)
            / self.N_ads
            if self.N_ads > 0
            else 0
        )

        print(
            """
===========================================================================

Restart file requested.

Loaded state with {} total atoms.

Curent total energy: {:.3f} eV
Current number of adsorbates: {}
Current average binding energy: {:.3f} kJ/mol

===========================================================================
""".format(
                len(state),
                self.current_total_energy,
                self.N_ads,
                average_binding_energy,
            ),
            file=self.out_file,
        )

    def print_introduction(self):
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
Fugacity coefficient:   {self.fugacity_coeff} (dimensionless)
Critical Lenght scale:  {self.lmbd:.3e} m

===========================================================================

Simulation Parameters:
Temperature: {self.T} K
Pressure: {self.P / 1e5} bar
Fugacity: {self.fugacity / units.J:.3} Pa
Fugacity: {self.fugacity} eV/m^3
Chemical potential: {self.mu_i} eV | {self.mu_i / (units.kJ / units.mol):.3f} kJ/mol
β * V * f = {self.V * self.beta * self.fugacity} [-]
V * exp(β μ_i) / Λ^3 = {self.V * np.exp(self.beta * self.mu_i) / (self.lmbd ** 3):.3e} [-]

===========================================================================

System Information:
Framework: {self.framework.get_chemical_formula()}
Framework: {self.n_atoms_framework} atoms,
Framework mass: {np.sum(self.framework.get_masses())} g/mol, {self.framework_mass} kg
Framework energy: {self.framework_energy} eV
Framework volume: {self.V} m^3
Framework density: {self.framework_mass / self.V} kg/m^3, {self.framework_mass / self.V * 1e3} g/cm^3
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
            header += f"  {ase.Atom(i).symbol:2} - {ase.Atom(j).symbol:2}: {self.vdw[i] + self.vdw[j]:.3f} Å\n"

        header += f"""
===========================================================================
Conversion factors:
    Conversion factor molecules/unit cell -> mol/kg:         {self.conv_factors['mol/kg']:.9f}
    Conversion factor molecules/unit cell -> mg/g:           {self.conv_factors['mg/g']:.9f}
    Conversion factor molecules/unit cell -> cm^3 STP/gr:    {self.conv_factors['cm^3 STP/gr']:.9f}
    Conversion factor molecules/unit cell -> cm^3 STP/cm^3:  {self.conv_factors['cm^3 STP/cm^3']:.9f}
    Conversion factor molecules/unit cell -> %wt:            {self.conv_factors['mg/g'] * 1e-3:.9f}

Partial pressure:
           {self.P:>15.5f} Pascal
           {self.P / 1e5:>15.5f} bar
           {self.P / 101325:>15.5f} atm
           {self.P / (101325 * 760):>15.5f} Torr
===========================================================================
"""
        print(header, file=self.out_file)

    def print_finish(self):
        """
        Print the footer for the simulation output.
        This method is called at the end of the simulation to display the final results.
        """

        avg_uptake = np.average(self.uptake_list) if self.uptake_list else 0
        std_uptake = np.std(self.uptake_list) if self.uptake_list else 0

        Qst = enthalpy_of_adsorption(
            energy=np.array(self.total_ads_list) / units.kB,  # Convert to K
            number_of_molecules=self.uptake_list,
            temperature=self.T,
        )

        print(
            """
===========================================================================

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Finishing simulation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    Average properties of the system:
    ------------------------------------------------------------------------------
    Average loading absolute [molecules/unit cell]       {:12.5f} +/- {:12.5f} [-]
    Average loading absolute [mol/kg framework]          {:12.5f} +/- {:12.5f} [-]
    Average loading absolute [mg/g framework]            {:12.5f} +/- {:12.5f} [-]
    Average loading absolute [cm^3 (STP)/gr framework]   {:12.5f} +/- {:12.5f} [-]
    Average loading absolute [cm^3 (STP)/cm^3 framework] {:12.5f} +/- {:12.5f} [-]
    Average loading absolute [%wt framework]             {:12.5f} +/- {:12.5f} [-]


    Enthalpy of adsorption: [kJ/mol]                     {:12.5f} +/- {:12.5f} [-]

===========================================================================
Simulation finished suscessfully!
===========================================================================

Simulation finished at {}
Simulation duration: {}

===========================================================================

""".format(
                avg_uptake,
                std_uptake,
                avg_uptake * self.conv_factors["mol/kg"],
                std_uptake * self.conv_factors["mol/kg"],
                avg_uptake * self.conv_factors["mg/g"],
                std_uptake * self.conv_factors["mg/g"],
                avg_uptake * self.conv_factors["cm^3 STP/gr"],
                std_uptake * self.conv_factors["cm^3 STP/gr"],
                avg_uptake * self.conv_factors["cm^3 STP/cm^3"],
                std_uptake * self.conv_factors["cm^3 STP/cm^3"],
                avg_uptake * self.conv_factors["mg/g"] * 1e-3,
                std_uptake * self.conv_factors["mg/g"] * 1e-3,
                Qst,
                0.0,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                datetime.datetime.now() - self.start_time,
            ),
            file=self.out_file,
        )

    def debug_movement(self, movement, deltaE, prefactor, acc, rnd_number) -> None:
        """
        Print debug information about the current state of the simulation.
        This method is called to provide detailed information about the current state of the system.
        """
        print(
            f"""
=======================================================================================================
Movement type: {movement}
Current number of adsorbates: {self.N_ads}
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
        )

    def optimize_framework(
        self,
        max_steps: int = 1000,
        opt_cell: bool = True,
        fix_symmetry: bool = True,
        hydrostatic_strain: bool = True,
        symm_tol: float = 1e-3,
        max_force: float = 0.05,
    ) -> None:
        """
        Optimize the framework structure using the provided calculator.

        Parameters
        ----------
        max_steps : int, optional
            Maximum number of optimization steps (default is 1000).
        tol : float, optional
            Tolerance for convergence (default is 1e-5).

        Returns
        -------
        ase.Atoms
            The optimized framework structure.
        """

        print(
            """
=======================================================================================================
Start optimizing framework structure...
=======================================================================================================
              """,
            file=self.out_file,
        )

        resultsDict, optFramework = crystalOptmization(
            atoms_in=self.framework,
            calculator=self.model,
            optimizer=LBFGS,  # type: ignore
            fmax=max_force,
            opt_cell=opt_cell,
            fix_symmetry=fix_symmetry,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=False,
            scalar_pressure=0,
            max_steps=max_steps,
            trajectory=self.trajectory,  # type: ignore
            verbose=True,
            symm_tol=symm_tol,
            out_file=self.out_file,  # type: ignore
        )

        # Remove any constraints from the optimized framework
        optFramework.set_constraint(None)

        self.set_framework(optFramework.copy())

    def optimize_adsorbate(self, max_steps: int = 1000, max_force: float = 0.05) -> None:
        """
        Optimize the adsorbate structure using the provided calculator.

        Parameters
        ----------
        max_steps : int, optional
            Maximum number of optimization steps (default is 1000).
        symm_tol : float, optional
            Tolerance for symmetry (default is 1e-3).
        max_force : float, optional
            Maximum force tolerance for convergence (default is 0.05 eV/Å).

        Returns
        -------
        ase.Atoms
            The optimized adsorbate structure.
        """

        print(
            """
=======================================================================================================
Start optimizing adsorbate structure...
=======================================================================================================
              """,
            file=self.out_file,
        )

        resultsDict, optAdsorbate = crystalOptmization(
            atoms_in=self.adsorbate,
            calculator=self.model,
            optimizer=LBFGS,  # type: ignore
            fmax=max_force,
            opt_cell=False,
            fix_symmetry=False,
            hydrostatic_strain=True,
            constant_volume=True,
            scalar_pressure=self.P,
            max_steps=max_steps,
            trajectory="Adsorbate_Optimization.traj",
            verbose=True,
            symm_tol=1e3,
            out_file=self.out_file,  # type: ignore
        )

        self.adsorbate = optAdsorbate.copy()
        self.adsorbate.set_constraint(None)
        self.adsorbate.set_cell(self.framework.get_cell())
        self.adsorbate.calc = self.model
        self.adsorbate_energy = self.adsorbate.get_potential_energy()

    def npt(self, nsteps, time_step: float = 0.5, mode: str = "iso_shape"):
        """
        Run a NPT simulation using the Berendsen thermostat and barostat.

        Parameters
        ----------
        nsteps : int
            Number of steps to run the NPT simulation.
        time_step : float, optional
            Time step for the NPT simulation (default is 0.5 fs).
        mode : str, optional
            The mode of the NPT simulation (default is "iso_shape").
            Can be one of "iso_shape", "aniso_shape", or "aniso_flex".
        """

        allowed_modes = ["iso_shape", "aniso_shape", "aniso_flex"]
        assert mode in allowed_modes, f"Mode must be one of {allowed_modes}."

        if mode == "iso_shape" or mode == "aniso_shape":

            new_state = nPT_Berendsen(
                atoms=self.current_system,
                model=self.model,
                temperature=self.T,
                pressure=self.P * 1e-5,
                compressibility=1e-4,
                time_step=time_step,
                num_md_steps=nsteps,
                isotropic=True if mode == "iso_shape" else False,
                out_folder=self.out_folder,
                out_file=self.out_file,  # type: ignore
                trajectory=self.trajectory,
                output_interval=self.save_every,
                movie_interval=self.save_every,
            )
        else:

            new_state = nPT_NoseHoover(
                atoms=self.current_system,
                model=self.model,
                temperature=self.T,
                pressure=self.P * 1e-5,
                time_step=time_step,
                num_md_steps=nsteps,
                ttime=25.0,
                ptime=75.0,
                B_guess=30,
                out_folder=self.out_folder,
                out_file=self.out_file,  # type: ignore
                trajectory=self.trajectory,
                output_interval=self.save_every,
                movie_interval=self.save_every,
            )

        self.set_state(new_state)

        self.set_framework(new_state[: self.n_atoms_framework].copy())  # type: ignore

    def nvt(self, nsteps, time_step: float = 0.5):
        """
        Run a NVT simulation using the Berendsen thermostat.

        Parameters
        ----------
        nsteps : int
            Number of steps to run the NVT simulation.
        time_step : float, optional
            Time step for the NVT simulation (default is 0.5 fs).
        """

        new_state = nVT_Berendsen(
            atoms=self.current_system,
            model=self.model,
            temperature=self.T,
            time_step=time_step,
            num_md_steps=nsteps,
            out_folder=self.out_folder,
            out_file=self.out_file,  # type: ignore
            trajectory=self.trajectory,
            output_interval=self.save_every,
            movie_interval=self.save_every,
        )

        self.set_state(new_state)

    def get_ideal_chemical_potential(self) -> float:
        """
        Calculate the ideal chemical potential for the adsorbate at the given
        temperature and pressure.

        The ideal chemical potential is calculated using the formula:
        μ_i = ln(λ^3 * β * P * f) * kB * T

        where:
        λ = sqrt(h^2 * β / (2 * π * m))
        β = 1 / (kB * T)

        P = Pressure in Pa
        f = Fugacity coefficient (dimensionless)
        kB = Boltzmann constant (J/K)
        h = Planck's constant (J·s)
        m = Mass of the adsorbate molecule in kg
        T = Temperature in Kelvin

        Returns
        -------
        float
            The ideal chemical potential in eV.
        """

        # Constants
        kB = 1.380649e-23  # m2 kg s-2 K-1 = J K-1
        h = 6.62607015e-34  # m2 kg / s = J s

        beta = 1 / (kB * self.T)  # J^-1

        lmbd = np.sqrt(h**2 * beta / (2 * np.pi * self.adsorbate_mass))  # unit: m

        mu_i = np.log(lmbd**3 * beta * self.P * self.fugacity_coeff) * (units.kB * self.T)  # In J
        mu_i *= units.J / units.mol  # Convert to eV

        return mu_i

    def _insertion_acceptance(self, deltaE) -> bool:
        """
        Calculate the acceptance probability for insertion of an adsorbate molecule as

        # Pacc (N -> N + 1) = min(1, β * V * f * exp(-β ΔE) / (N + 1))
        """

        # if deltaE / (units.kJ / units.mol) < 100:
        #    return True  # Always accept if the energy change is too high

        exp_value = np.exp(-self.beta * deltaE)

        pre_factor = self.V * self.beta * self.fugacity / (self.N_ads + 1)

        acc = min(1, pre_factor * exp_value)

        rnd_number = np.random.rand()

        if self.debug:
            self.debug_movement(
                movement="Insertion",
                deltaE=deltaE,
                prefactor=pre_factor,
                acc=acc,
                rnd_number=rnd_number,
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def _deletion_acceptance(self, deltaE) -> bool:
        """
        Calculate the acceptance probability for deletion of an adsorbate molecule as

        Pdel (N -> N - 1 ) = min(1, N / (β * V * f) * exp(-β ΔE) )
        """

        # if deltaE / (units.kJ / units.mol) > 100:
        #    return True  # Always accept if the energy change is too high

        exp_value = np.exp(-self.beta * deltaE)

        pre_factor = self.N_ads / (self.V * self.beta * self.fugacity)

        acc = min(1, pre_factor * exp_value)

        rnd_number = np.random.rand()

        if self.debug:
            self.debug_movement(
                movement="Deletion",
                deltaE=deltaE,
                prefactor=pre_factor,
                acc=acc,
                rnd_number=rnd_number,
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def _move_acceptance(self, deltaE) -> bool:
        """
        Calculate the acceptance probability for translation or rotation of an adsorbate molecule as

        Pmove = min(1, exp(-β ΔE))
        """

        exp_value = np.exp(-self.beta * deltaE)
        acc = min(1, exp_value)

        rnd_number = np.random.rand()

        if self.debug:
            self.debug_movement(
                movement="Movement", deltaE=deltaE, prefactor=1, acc=acc, rnd_number=rnd_number
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def try_insertion(self) -> bool:
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

        atoms_trial = self.current_system.copy() + self.adsorbate.copy()  # type: ignore
        atoms_trial.calc = self.model

        pos = atoms_trial.get_positions()
        pos[-self.n_ads :] = random_position(pos[-self.n_ads :], atoms_trial.get_cell())
        atoms_trial.set_positions(pos)
        atoms_trial.wrap()

        if vdw_overlap(atoms_trial, self.vdw, self.n_atoms_framework, self.n_ads, self.N_ads):
            return False

        atoms_trial.calc = self.model
        e_new = atoms_trial.get_potential_energy()

        deltaE = e_new - self.current_total_energy - self.adsorbate_energy

        # Apply the acceptance criteria for insertion
        if self._insertion_acceptance(deltaE=deltaE):
            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_new
            self.N_ads += 1
            return True
        else:
            return False

    def try_deletion(self) -> bool:
        """
        Try to delete an adsorbate molecule from the framework.
        This method randomly selects an adsorbate molecule and try to apply the deletion.

        If there are no adsorbates, it returns False.

        Returns
        -------
        bool
            True if the deletion was accepted, False otherwise.
        """
        if self.N_ads == 0:
            return False

        # Randomly select an adsorbate molecule to delete
        i_ads = np.random.randint(self.N_ads)
        atoms_trial = self.current_system.copy()
        atoms_trial.calc = self.model  # type: ignore

        # Get the indices of the adsorbate atoms to be deleted
        i_start = self.n_atoms_framework + self.n_ads * i_ads
        i_end = self.n_atoms_framework + self.n_ads * (i_ads + 1)

        # Delete the adsorbate atoms from the trial structure
        del atoms_trial[i_start:i_end]

        # Calculate the new potential energy of the trial structure
        e_new = atoms_trial.get_potential_energy()  # type: ignore

        deltaE = e_new + self.adsorbate_energy - self.current_total_energy

        # Apply the acceptance criteria for deletion
        if self._deletion_acceptance(deltaE=deltaE):

            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_new
            self.N_ads -= 1

            return True
        else:
            return False

    def try_translation(self) -> bool:
        if self.N_ads == 0:
            return False

        i_ads = np.random.randint(self.N_ads)
        atoms_trial = self.current_system.copy()
        atoms_trial.calc = self.model  # type: ignore

        pos = atoms_trial.get_positions()  # type: ignore

        i_start = self.n_atoms_framework + self.n_ads * i_ads
        i_end = self.n_atoms_framework + self.n_ads * (i_ads + 1)

        pos[i_start:i_end] += 0.5 * (np.random.rand(3) - 0.5)

        atoms_trial.set_positions(pos)  # type: ignore
        if vdw_overlap(atoms_trial, self.vdw, self.n_atoms_framework, self.n_ads, i_ads):
            return False

        e_trial = atoms_trial.get_potential_energy()  # type: ignore

        deltaE = e_trial - self.current_total_energy
        if self._move_acceptance(deltaE=deltaE):
            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_trial
            return True
        else:
            return False

    def try_rotation(self) -> bool:
        if self.N_ads == 0:
            return False

        i_ads = np.random.randint(self.N_ads)
        atoms_trial = self.current_system.copy()
        atoms_trial.calc = self.model  # type: ignore

        pos = atoms_trial.get_positions()  # type: ignore
        i_start = self.n_atoms_framework + self.n_ads * i_ads
        i_end = self.n_atoms_framework + self.n_ads * (i_ads + 1)

        pos[i_start:i_end] = random_rotation(pos[i_start:i_end])
        atoms_trial.set_positions(pos)  # type: ignore

        if vdw_overlap(atoms_trial, self.vdw, self.n_atoms_framework, self.n_ads, i_ads):
            return False

        e_trial = atoms_trial.get_potential_energy()  # type: ignore

        deltaE = e_trial - self.current_total_energy

        if self._move_acceptance(deltaE=deltaE):
            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_trial
            return True
        else:
            return False

    def run(self, N) -> None:
        """Run the Grand Canonical Monte Carlo simulation for N iterations."""

        header = """
 Iteration |  Number of  |  Uptake  |    Tot En.   |Av. Ads. En.|  Pacc  |  Pdel  |  Ptra  |  Prot  |  Time
      -    |  Molecules  | [mmol/g] |     [eV]     |  [kJ/mol]  |    %   |    %   |   %    |   %    |   [s]
---------- | ----------- | -------- | ------------ | ---------- | ------ | ------ | ------ | ------ | ------
"""
        print(header, file=self.out_file)

        for iteration in tqdm(range(1, N + 1), disable=(self.out_file is None), desc="GCMC Step"):

            step_time_start = datetime.datetime.now()

            switch = np.random.rand()

            if switch < 0.25:
                accepted = self.try_insertion()
                self.mov_dict["insertion"].append(1 if accepted else 0)

            elif switch < 0.5:
                accepted = self.try_deletion()
                self.mov_dict["deletion"].append(1 if accepted else 0)

            # Translation
            elif switch < 0.75:
                accepted = self.try_translation()
                self.mov_dict["translation"].append(1 if accepted else 0)

            # Rotation
            elif switch >= 0.75:
                accepted = self.try_rotation()
                self.mov_dict["rotation"].append(1 if accepted else 0)

            self.uptake_list.append(self.N_ads)
            self.total_energy_list.append(self.current_total_energy)
            self.total_ads_list.append(
                self.current_total_energy
                - (self.N_ads * self.adsorbate_energy)
                - self.framework_energy
            )

            average_ads_energy = (
                self.current_total_energy
                - (self.N_ads * self.adsorbate_energy)
                - self.framework_energy
            ) / (units.kJ / units.mol)

            average_ads_energy = average_ads_energy / self.N_ads if self.N_ads > 0 else 0

            line_str = "{:^11}|{:^13}|{:>9.2f} |{:>13.4f} |{:>11.4f} |{:7.2f} |{:7.2f} |{:7.2f} |{:7.2f} |{:9.2f}"

            print(
                line_str.format(
                    iteration,
                    self.N_ads,
                    self.N_ads * self.conv_factors["mol/kg"],
                    self.current_total_energy,
                    average_ads_energy,
                    (
                        np.average(self.mov_dict["insertion"]) * 100
                        if len(self.mov_dict["insertion"]) > 0
                        else 0
                    ),
                    (
                        np.average(self.mov_dict["deletion"]) * 100
                        if len(self.mov_dict["deletion"]) > 0
                        else 0
                    ),
                    (
                        np.average(self.mov_dict["translation"]) * 100
                        if len(self.mov_dict["translation"]) > 0
                        else 0
                    ),
                    (
                        np.average(self.mov_dict["rotation"]) * 100
                        if len(self.mov_dict["rotation"]) > 0
                        else 0
                    ),
                    (datetime.datetime.now() - step_time_start).total_seconds(),
                ),
                file=self.out_file,
            )

            if iteration % self.save_every == 0:
                # Save the current state of the system as xyz deactivated for now
                # write(
                # os.path.join(self.out_folder,
                # f'Movies/snapshot_{len(self.uptake_list)}_{self.P:.2f}_{self.T:.2f}.xyz'),
                #       self.current_system,
                #       format='extxyz')

                self.trajectory.write(self.current_system)  # type: ignore

                np.save(
                    os.path.join(self.out_folder, f"uptake_{self.P:.5f}.npy"),
                    np.array(self.uptake_list),
                )

                np.save(
                    os.path.join(self.out_folder, f"total_energy_{self.P:.5f}.npy"),
                    np.array(self.total_energy_list),
                )

                np.save(
                    os.path.join(self.out_folder, f"total_ads_{self.P:.5f}.npy"),
                    np.array(self.total_ads_list),
                )
