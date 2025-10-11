import datetime
import os
from typing import Union

import ase
import numpy as np
import simplejson as json
from ase import units
from ase.calculators import calculator
from ase.io import write
from tqdm import tqdm

from flames.base_simulator import BaseSimulator
from flames.logger import WidomLogger
from flames.operations import check_overlap, random_mol_insertion
from flames.utilities import random_n_splits


class Widom(BaseSimulator):
    def __init__(
        self,
        framework_atoms: ase.Atoms,
        adsorbate_atoms: ase.Atoms,
        temperature: float,
        model: calculator.Calculator,
        vdw_radii: np.ndarray,
        vdw_factor: float = 0.6,
        max_deltaE: float = 1.555,
        device: str = "cpu",
        save_frequency: int = 100,
        output_to_file: bool = True,
        debug: bool = False,
        random_seed: Union[int, None] = None,
        cutoff_radius: float = 6.0,
        automatic_supercell: bool = True,
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
        max_deltaE : float, optional
            Maximum energy difference (in eV) to consider for acceptance criteria.
            This is used to avoid overflow due to problematic calculations (default is 1.555 eV / 150 kJ/mol).
        device : str, optional
            Device to run the calculations on, either 'cpu' or 'cuda'. Default is 'cpu'.
        random_seed : int | None
            Random seed for reproducibility (default is None).
        cutoff_radius : float
            Interaction potential cut-off radius used to estimate the minimum unit cell (default is 6.0).
        automatic_supercell : bool
            Whether to automatically create a supercell based on the cutoff radius (default is True).
        """

        super().__init__(
            model=model,
            framework_atoms=framework_atoms,
            adsorbate_atoms=adsorbate_atoms,
            temperature=temperature,
            pressure=0.0,
            device=device,
            vdw_radii=vdw_radii,
            vdw_factor=vdw_factor,
            max_deltaE=max_deltaE,
            save_frequency=save_frequency,
            output_to_file=output_to_file,
            debug=debug,
            fugacity_coeff=0.0,
            random_seed=random_seed,
            cutoff_radius=cutoff_radius,
            automatic_supercell=automatic_supercell,
        )

        self.logger = WidomLogger(simulation=self, output_file=self.out_file)

        self.start_time = datetime.datetime.now()

        self.minimum_configuration: ase.Atoms = self.framework.copy()
        self.minimum_energy: float = 0
        self.base_iteration = 0
        self.N_ads = 0
        self.int_energy_list = np.zeros(1, dtype=float)

        self.boltz_fac = np.exp(-self.beta * self.int_energy_list)

        # kH = β <exp(-β ΔE)> [mol kg-1 Pa-1]
        self.kH = (
            self.beta
            * self.boltz_fac.mean()
            * (units.J / units.mol)
            / (self.framework_density * 1e3)
        )

        self.kH_std_dv = 0.0

        # Qst = - < ΔE * exp(-β ΔE) > / <exp(-β ΔE)>  + kB.T # [kJ/mol]
        self.Qst = (
            (self.int_energy_list * self.boltz_fac).mean() / self.boltz_fac.mean()
            - (units.kB * self.T)
        ) / (units.kJ / units.mol)

        self.Qst_std_dv = 0.0

    def update_statistics(self, deltaE) -> None:
        """
        Update the statistics of the Widom insertion method after a new insertion.

        Parameters
        ----------
        deltaE : float
            The change in energy associated with the latest insertion.
        """

        self.int_energy_list = np.append(self.int_energy_list, deltaE)

        

        self.boltz_fac = np.exp(-self.beta * self.int_energy_list)

        

        # kH = β <exp(-β ΔE)> [mol kg-1 Pa-1]
        self.kH = (
            self.beta
            * self.boltz_fac.mean()
            * (units.J / units.mol)
            / (self.framework_density * 1e3)
        )

        # Qst = - < ΔE * exp(-β ΔE) > / <exp(-β ΔE)>  + kB.T # [kJ/mol]
        self.Qst = (
            (self.int_energy_list * self.boltz_fac).mean() / self.boltz_fac.mean()
            - (units.kB * self.T)
        ) / (units.kJ / units.mol)

        # Calculate standard deviation using cross-validation
        if len(self.int_energy_list) > 5:
            cv_int_energy_list = random_n_splits(self.int_energy_list, 5, self.rnd_generator)

            cv_boltz_fac = np.exp(-self.beta * cv_int_energy_list)

            # Calculate standard deviation using cross-validation
            self.kH_std_dv = (
                (
                    self.beta
                    * cv_boltz_fac.mean(axis=-1)
                    * (units.J / units.mol)
                    / (self.framework_density * 1e3)
                )
            ).std()

            self.Qst_std_dv = (
                (
                    (cv_int_energy_list * cv_boltz_fac).mean(axis=-1) / cv_boltz_fac.mean(axis=-1)
                    - (units.kB * self.T)
                )
                / (units.kJ / units.mol)
            ).std()

    def save_results(self, file_name: str = "Widom_Results.json") -> None:
        """
        Save a json file with the main results of the simulation.

        Parameters
        ----------
        file_name : str, optional
            Name of the output json file (default is 'Widom_Results.json').
        """

        results = {
            "temperature_K": self.T,
            "henry_coefficient_mol_kg-1_Pa-1": self.kH,
            "henry_coefficient_std_mol_kg-1_Pa-1": self.kH_std_dv,
            "enthalpy_of_adsorption_kJ_mol-1": self.Qst,
            "enthalpy_of_adsorption_std_kJ_mol-1": self.Qst_std_dv,
            "total_insertions": len(self.int_energy_list),
        }

        with open(os.path.join(self.out_folder, file_name), "w") as f:
            json.dump(results, f, indent=4)

    def restart(self) -> None:
        """
        Restart the simulation from the last state.

        This method loads the last saved state from the trajectory file and restores the simulation to that state.
        It also loads the uptake, total energy, and total adsorbates lists from the saved files if they exist.
        """

        print("Restarting simulation...")

        self.int_energy_list = np.load(
            os.path.join(self.out_folder, f"int_energy_{self.P:.5f}.npy")
        )

        # Set the base iteration to the length of the uptake list
        self.base_iteration = len(self.int_energy_list)

        self.logger.print_restart_info()

    def try_insertion(self) -> tuple[float, ase.Atoms]:
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

        atoms_trial = random_mol_insertion(self.framework, self.adsorbate, self.rnd_generator)

        overlaped = check_overlap(
            atoms=atoms_trial,
            group1_indices=np.arange(self.n_atoms_framework),
            group2_indices=np.arange(
                self.n_atoms_framework, self.n_atoms_framework + self.n_adsorbate_atoms
            ),
            vdw_radii=self.vdw,
        )

        if overlaped:
            return 1000.0, atoms_trial  # Return 1000 energy to indicate overlap

        atoms_trial.calc = self.model
        e_new = atoms_trial.get_potential_energy()

        deltaE = e_new - self.framework_energy - self.adsorbate_energy

        if np.abs(deltaE) > np.abs(self.max_deltaE):
            return 1000.0, atoms_trial  # Return 1000 energy to indicate error

        return deltaE, atoms_trial

    def run(self, N) -> None:

        header = """
Iteration  |  dE (eV)  |  dE (kJ/mol)  | kH [mol kg-1 Pa-1]  |  dH (kJ/mol) | Time (s)
---------------------------------------------------------------------------------------"""
        print(header, file=self.out_file, flush=True)

        for iteration in tqdm(range(1, N + 1), disable=(self.out_file is None), desc="Widom Step"):

            actual_iteration = iteration + self.base_iteration

            step_time_start = datetime.datetime.now()

            accepted = False

            deltaE = 0.0

            atoms_trial = self.framework.copy()

            insert_iter = 0

            while not accepted:
                insert_iter += 1
                deltaE, atoms_trial = self.try_insertion()
                if deltaE < units.kB * self.T or insert_iter > 1000:
                    accepted = True

            if deltaE < self.minimum_energy:
                self.minimum_configuration = atoms_trial.copy()
                self.minimum_energy = deltaE
                tmp_name = "minimum_configuration_{:.2f}.cif".format(
                    deltaE / (units.kJ / units.mol)
                )

                write(
                    os.path.join(os.path.join(self.out_folder, "Movies", tmp_name)),
                    atoms_trial,
                    format="cif",
                )

            self.trajectory.write(atoms_trial)  # type: ignore

            # Append int_energy_list
            self.update_statistics(deltaE)

            self.logger.print_iteration_info(
                [
                    actual_iteration,
                    deltaE,
                    deltaE / (units.kJ / units.mol),
                    self.kH,
                    self.Qst,
                    (datetime.datetime.now() - step_time_start).total_seconds(),
                ],
            )

            if actual_iteration % self.save_every == 0:

                np.save(
                    os.path.join(self.out_folder, f"int_energy_{self.P:.5f}.npy"),
                    np.array(self.int_energy_list),
                )
