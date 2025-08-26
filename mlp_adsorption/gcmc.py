import datetime
import os
from typing import Union

import ase
import numpy as np
from ase import units
from ase.calculators import calculator
from ase.io import Trajectory, read
from tqdm import tqdm

from mlp_adsorption.base_simulator import BaseSimulator
from mlp_adsorption.eos import PengRobinsonEOS
from mlp_adsorption.logger import GCMCLogger
from mlp_adsorption.operations import (
    check_overlap,
    random_insertion_cell,
    random_rotation,
    random_translation,
)


class GCMC(BaseSimulator):
    def __init__(
        self,
        model: calculator.Calculator,
        framework_atoms: ase.Atoms,
        adsorbate_atoms: ase.Atoms,
        temperature: float,
        pressure: float,
        device: str,
        vdw_radii: np.ndarray,
        vdw_factor: float = 0.6,
        save_frequency: int = 100,
        output_to_file: bool = True,
        debug: bool = False,
        fugacity_coeff: float = 1.0,
        random_seed: Union[int, None] = None,
        cutoff_radius: float = 6.0,
        criticalTemperature: Union[float, None] = None,
        criticalPressure: Union[float, None] = None,
        acentricFactor: Union[float, None] = None,
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
        device : str
            Device to run the simulation on, e.g., 'cpu' or 'cuda'.
        vdw_radii : np.ndarray
            Van der Waals radii for the atoms in the framework and adsorbate.
            Should be an array of the same length as the number of atomic numbers in ASE.
        vdw_factor : float, optional
            Factor to scale the Van der Waals radii (default is 0.6).
        save_frequency : int, optional
            Frequency at which to save the simulation state and results (default is 100).
        output_to_file : bool, optional
            If True, writes the output to a file named 'GCMC_Output.out' in the 'results' directory
        debug : bool, optional
            If True, prints detailed debug information during the simulation (default is False).
        fugacity_coeff : float, optional
            Fugacity coefficient to correct the pressure. Default is 1.0.
            Only used if `criticalTemperature`, `criticalPressure`, and `acentricFactor` are not provided.
        random_seed : int | None
            Random seed for reproducibility (default is None).
        cutoff_radius : float
            Interaction potential cut-off radius used to estimate the minimum unit cell (default is 6.0).
        criticalTemperature : float, optional
            Critical temperature of the adsorbate in Kelvin.
        criticalPressure : float, optional
            Critical pressure of the adsorbate in Pascal.
        acentricFactor : float, optional
            Acentric factor of the adsorbate.
        """

        super().__init__(
            model=model,
            framework_atoms=framework_atoms,
            adsorbate_atoms=adsorbate_atoms,
            temperature=temperature,
            pressure=pressure,
            device=device,
            vdw_radii=vdw_radii,
            vdw_factor=vdw_factor,
            save_frequency=save_frequency,
            output_to_file=output_to_file,
            debug=debug,
            fugacity_coeff=fugacity_coeff,
            random_seed=random_seed,
            cutoff_radius=cutoff_radius,
        )

        self.logger = GCMCLogger(simulation=self, output_file=self.out_file)

        self.start_time = datetime.datetime.now()

        # Parameters for calculateing the Peng-Robinson equation of state
        self.criticalTemperature = criticalTemperature
        self.criticalPressure = criticalPressure
        self.acentricFactor = acentricFactor

        # Check if any critical parameters are not None
        if all([self.criticalTemperature, self.criticalPressure, self.acentricFactor]):
            self.eos = PengRobinsonEOS(
                temperature=self.T,
                pressure=self.P,
                criticalTemperature=self.criticalTemperature,  # type: ignore
                criticalPressure=self.criticalPressure,  # type: ignore
                acentricFactor=self.acentricFactor,  # type: ignore
                molarMass=self.adsorbate_mass,
            )
            self.fugacity_coeff = self.eos.get_fugacity_coefficient()

        # Parameters for storing the main results during the simulation
        self.N_ads: int = 0
        self.uptake_list: list[int] = []
        self.total_energy_list: list[float] = []
        self.total_ads_list: list[float] = []

        self.mov_dict: dict = {"insertion": [], "deletion": [], "translation": [], "rotation": []}

        # Base iteration for restarting the simulation. This is for tracking the iteration count only
        self.base_iteration: int = 0

    def restart(self) -> None:
        """
        Restart the simulation from the last state.

        This method loads the last saved state from the trajectory file and restores the simulation to that state.
        It also loads the uptake, total energy, and total adsorbates lists from the saved files if they exist.
        """

        print("Restarting simulation...")
        uptake_restart, total_energy_restart, total_ads_restart = [], [], []

        if os.path.exists(os.path.join(self.out_folder, f"uptake_{self.P:.5f}.npy")):
            uptake_restart = np.load(
                os.path.join(self.out_folder, f"uptake_{self.P:.5f}.npy")
            ).tolist()

        if os.path.exists(os.path.join(self.out_folder, f"total_energy_{self.P:.5f}.npy")):
            total_energy_restart = np.load(
                os.path.join(self.out_folder, f"total_energy_{self.P:.5f}.npy")
            ).tolist()

        if os.path.exists(os.path.join(self.out_folder, f"total_ads_{self.P:.5f}.npy")):
            total_ads_restart = np.load(
                os.path.join(self.out_folder, f"total_ads_{self.P:.5f}.npy")
            ).tolist()

        # Check if the len of all restart elements are the same:
        if not (len(uptake_restart) == len(total_energy_restart) == len(total_ads_restart)):
            raise ValueError(
                f"""
            The lengths of uptake, total energy, and total adsorbates lists do not match.
            Please check the saved files.
            Found lengths: {len(uptake_restart)}, {len(total_energy_restart)}, {len(total_ads_restart)}
            for uptake, total energy, and total ads respectively."""
            )

        self.uptake_list = uptake_restart
        self.total_energy_list = total_energy_restart
        self.total_ads_list = total_ads_restart

        # Set the base iteration to the length of the uptake list
        self.base_iteration = len(self.uptake_list)

        self.logger.print_restart_info()

        self.load_state(os.path.join(self.out_folder, "GCMC_Trajectory.traj"))

    def load_state(self, state_file: str) -> None:
        """
        Load the state of the simulation from a file.

        Parameters
        ----------
        state_file : str
            Path to the file containing the saved state of the simulation.
        """
        print(f"Loading state from {state_file}...")

        if not os.path.exists(state_file):
            raise FileNotFoundError(f"State file '{state_file}' does not exist.")

        if state_file.endswith(".traj"):
            state = Trajectory(state_file, "r")[-1]  # type: ignore
        else:
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

        self.logger.print_load_state_info(
            n_atoms=len(state), average_ads_energy=average_binding_energy
        )

    def _insertion_acceptance(self, deltaE) -> bool:
        """
        Calculate the acceptance probability for insertion of an adsorbate molecule as

        # Pacc (N -> N + 1) = min(1, β * V * f * exp(-β ΔE) / (N + 1))
        """

        exp_value = np.exp(-self.beta * deltaE)

        pre_factor = self.V * self.beta * self.fugacity / (self.N_ads + 1)

        acc = min(1, pre_factor * exp_value)

        rnd_number = np.random.rand()

        if self.debug:
            self.logger.print_debug_movement(
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

        exp_value = np.exp(-self.beta * deltaE)

        pre_factor = self.N_ads / (self.V * self.beta * self.fugacity)

        acc = min(1, pre_factor * exp_value)

        rnd_number = np.random.rand()

        if self.debug:
            self.logger.print_debug_movement(
                movement="Deletion",
                deltaE=deltaE,
                prefactor=pre_factor,
                acc=acc,
                rnd_number=rnd_number,
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def _move_acceptance(self, deltaE, movement_name="Movement") -> bool:
        """
        Calculate the acceptance probability for translation or rotation of an adsorbate molecule as

        Pmove = min(1, exp(-β ΔE))
        """

        exp_value = np.exp(-self.beta * deltaE)
        acc = min(1, exp_value)

        rnd_number = np.random.rand()

        if self.debug:
            self.logger.print_debug_movement(
                movement=movement_name, deltaE=deltaE, prefactor=1, acc=acc, rnd_number=rnd_number
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

        pos[-self.n_adsorbate_atoms :] = random_insertion_cell(
            original_positions=pos[-self.n_adsorbate_atoms :],
            lattice_vectors=atoms_trial.get_cell(),
            rnd_generator=self.rnd_generator,
        )

        atoms_trial.set_positions(pos)
        atoms_trial.wrap()

        overlaped = check_overlap(
            atoms=atoms_trial,
            group1_indices=np.arange(self.n_atoms_framework),
            group2_indices=np.arange(self.n_atoms_framework, self.n_atoms_framework + self.n_adsorbate_atoms),
            vdw_radii=self.vdw,
        )

        if overlaped:
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
        i_start = self.n_atoms_framework + self.n_adsorbate_atoms * i_ads
        i_end = self.n_atoms_framework + self.n_adsorbate_atoms * (i_ads + 1)

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

        i_start = self.n_atoms_framework + self.n_adsorbate_atoms * i_ads
        i_end = self.n_atoms_framework + self.n_adsorbate_atoms * (i_ads + 1)

        pos[i_start:i_end] = random_translation(
            original_positions=pos[i_start:i_end],
            max_translation=1.0,
            rnd_generator=self.rnd_generator,
        )

        atoms_trial.set_positions(pos)  # type: ignore

        overlaped = check_overlap(
            atoms=atoms_trial,
            group1_indices=np.concatenate(
                [np.arange(0, i_start), np.arange(i_end, len(atoms_trial))]
            ),
            group2_indices=np.arange(i_start, i_end),
            vdw_radii=self.vdw,
        )

        if overlaped:
            return False

        e_trial = atoms_trial.get_potential_energy()  # type: ignore

        deltaE = e_trial - self.current_total_energy
        if self._move_acceptance(deltaE=deltaE, movement_name="Translation"):
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
        i_start = self.n_atoms_framework + self.n_adsorbate_atoms * i_ads
        i_end = self.n_atoms_framework + self.n_adsorbate_atoms * (i_ads + 1)

        pos[i_start:i_end] = random_rotation(pos[i_start:i_end], rnd_generator=self.rnd_generator)
        atoms_trial.set_positions(pos)  # type: ignore

        overlaped = check_overlap(
            atoms=atoms_trial,
            group1_indices=np.concatenate(
                [np.arange(0, i_start), np.arange(i_end, len(atoms_trial))]
            ),
            group2_indices=np.arange(i_start, i_end),
            vdw_radii=self.vdw,
        )

        if overlaped:
            return False

        e_trial = atoms_trial.get_potential_energy()  # type: ignore

        deltaE = e_trial - self.current_total_energy

        if self._move_acceptance(deltaE=deltaE, movement_name="Rotation"):
            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_trial
            return True
        else:
            return False

    def run(self, N) -> None:
        """Run the Grand Canonical Monte Carlo simulation for N iterations."""

        self.logger.print_run_header()

        for iteration in tqdm(range(1, N + 1), disable=(self.out_file is None), desc="GCMC Step"):

            actual_iteration = iteration + self.base_iteration

            step_time_start = datetime.datetime.now()

            switch = np.random.rand()

            # Insertion
            if switch < 0.25:
                accepted = self.try_insertion()
                self.mov_dict["insertion"].append(1 if accepted else 0)

            # Deletion
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

            self.logger.print_step_info(
                step=actual_iteration,
                average_ads_energy=average_ads_energy,
                step_time=(datetime.datetime.now() - step_time_start).total_seconds(),
            )

            if actual_iteration % self.save_every == 0:

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
