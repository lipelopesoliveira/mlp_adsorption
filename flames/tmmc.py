from __future__ import annotations

import datetime
import json
import os

import ase
import numpy as np
from ase.calculators import calculator
from ase.io import Trajectory, read
from tqdm import tqdm

from flames import VERSION
from flames.adsorbate import Adsorbate
from flames.base_simulator import BaseSimulator
from flames.logger import TMMCLogger
from flames.operations import check_overlap_vesin, random_mol_insertion


class TMMC(BaseSimulator):
    """
    Base class for transition matrix Monte Carlo (TMMC) simulations using ASE.

    This class implements TMMC deletion/insertion moves, recording the
    deletion/insertion energies of the adsorbate.

    :param model:
        The calculator to use for energy calculations. Can be any ASE-compatible calculator.
        The output of the calculator should be in eV.
    :type model: ase.calculators.calculator.Calculator

    :param framework_atoms:
        The framework structure as an ASE Atoms object.
    :type framework_atoms: ase.Atoms

    :param adsorbates:
        The adsorbate structure(s) as an Adsorbate object or list of Adsorbate objects.
    :type adsorbates: Adsorbate | list[Adsorbate]

    :param temperature:
        Temperature of the ideal reservoir in Kelvin.
    :type temperature: float

    :param pressure:
        Pressure of the ideal reservoir in Pascal.
    :type pressure: float

    :param device:
        Device to run the simulation on, e.g., ``'cpu'`` or ``'cuda'``.
    :type device: str

    :param vdw_radii:
        Van der Waals radii for the atoms in the framework and adsorbate.
        Should be an array of the same length as the number of atomic numbers in ASE.
    :type vdw_radii: np.ndarray

    :param vdw_factor:
        Factor to scale the Van der Waals radii. Default is ``0.6``.
    :type vdw_factor: float, optional

    :max_overlap_tries:
        Maximum tries for the insertion move. Default is ``100``.
    :type max_overlap_tries: int, optional

    :param save_frequency:
        Frequency at which to save the simulation state and results. Default is ``100``.
    :type save_frequency: int, optional

    :param output_to_file:
        If ``True``, writes the output to a file named ``output_{temperature}_{pressure}.out`` in the ``results`` directory. Default is ``True``.
    :type output_to_file: bool, optional

    :param output_folder:
        Folder to save the output files. If ``None``, a folder named ``results_<T>_<P>`` will be created.
    :type output_folder: str or None, optional

    :param debug:
        If ``True``, prints detailed debug information during the simulation. Default is ``False``.
    :type debug: bool, optional

    :param random_seed:
        Random seed for reproducibility. Default is ``None`` and will generate a random seed automatically if not provided.
    :type random_seed: int or None, optional

    :param cutoff_radius:
        Interaction potential cut-off radius used to estimate the minimum unit cell. Default is ``6.0``.
    :type cutoff_radius: float, optional
    """

    def __init__(
        self,
        model: calculator.Calculator,
        framework_atoms: ase.Atoms,
        adsorbates: Adsorbate | list[Adsorbate],
        temperature: float,
        pressure: float,
        device: str,
        vdw_radii: np.ndarray,
        vdw_factor: float = 0.6,
        max_overlap_tries: int = 100,
        max_deltaE: float = 25.0,
        save_frequency: int = 100,
        output_to_file: bool = True,
        output_folder: str | None = None,
        debug: bool = False,
        random_seed: int | None = None,
        cutoff_radius: float = 6.0,
    ) -> None:
        """
        Initialize the transition matrix Monte Carlo (TMMC) simulation.
        """

        super().__init__(
            model=model,
            framework_atoms=framework_atoms,
            adsorbates=adsorbates,
            temperature=temperature,
            pressure=pressure,
            device=device,
            vdw_radii=vdw_radii,
            vdw_factor=vdw_factor,
            max_deltaE=max_deltaE,
            save_frequency=save_frequency,
            save_rejected=False,
            output_to_file=output_to_file,
            output_folder=output_folder,
            debug=debug,
            fugacity_coeff=0.0,
            random_seed=random_seed,
            cutoff_radius=cutoff_radius,
            automatic_supercell=False,
        )

        self.logger = TMMCLogger(simulation=self, output_file=self.out_file)

        self.start_time = datetime.datetime.now()

        # Tracking number of adsorbates per species
        self.n_adsorbates: dict[str, int] = {adsorbate.name: 0 for adsorbate in self.adsorbates}

        # Parameters for storing the main results during the simulation (partitioned by adsorbate)
        self.total_ins_energy_list: dict[str, list[float]] = {
            ads.name: [] for ads in self.adsorbates
        }
        self.total_del_energy_list: dict[str, list[float]] = {
            ads.name: [] for ads in self.adsorbates
        }
        self.volume_list: list[float] = []
        self._del_indices: dict = {}

        # Maximum number of tries to insert a molecule without overlap
        self.max_overlap_tries = max_overlap_tries

        # Base iteration for restarting the simulation. This is for tracking the iteration count only
        self._base_iteration: int = 0

    @property
    def base_iteration(self) -> int:
        """
        Get the base iteration for the TMMC simulation.

        Returns
        -------
        int
            The base iteration count.
        """
        return self._base_iteration

    @base_iteration.setter
    def base_iteration(self, iteration: int) -> None:
        """
        Set the base iteration for the TMMC simulation.

        Parameters
        ----------
        iteration : int
            The base iteration count to set.
        """
        self._base_iteration = iteration

    def get_number_of_adsorbates(self, system: ase.Atoms | None = None) -> dict[str, int]:
        """
        Get the number of adsorbates in the current system.
        It considers the possibility of having multiple adsorbate species in the simulation.
        """
        if not system:
            system = self.current_system

        n_adsorbate_by_type = {}

        for adsorbate in self.adsorbates:
            adsorbate_indices = np.where(system.get_tags() == adsorbate.tag)[0]
            n_adsorbate_by_type[adsorbate.name] = int(
                len(adsorbate_indices) / len(adsorbate.structure)
            )

        return n_adsorbate_by_type

    def get_adsorbates_index(self, tag: int | None = None) -> list[list]:
        """
        Get a list of indices of the adsorbate molecules in the current system.
        """
        adsorbates_list = []

        for adsorbate in self.adsorbates:
            if tag is not None and adsorbate.tag != tag:
                continue

            indices = np.where(self.current_system.get_tags() == adsorbate.tag)[0]

            if len(indices) > 0:
                adsorbates_list.extend(indices.reshape(-1, len(adsorbate.structure)).tolist())

        return adsorbates_list

    def get_macrostate_str(self) -> str:
        """
        Return a string representation of the current multi-component macrostate.
        Example: For a binary mixture with 1 CO2 and 2 H2O, returns "0001_0002".
        """
        return "_".join([f"{self.n_adsorbates[ads.name]:04d}" for ads in self.adsorbates])

    def restart(self) -> None:
        """
        Restart the simulation from the last state.

        This method loads the last saved state from the trajectory file and restores the simulation to that state.
        It also loads the arrays handling the deletion/insertion energies.
        """

        print("Restarting simulation...")

        macrostate_str = self.get_macrostate_str()

        for ads in self.adsorbates:
            ins_file = os.path.join(self.out_folder, f"ins_energy_{ads.name}_{macrostate_str}.npy")
            del_file = os.path.join(self.out_folder, f"del_energy_{ads.name}_{macrostate_str}.npy")

            if os.path.exists(ins_file):
                self.total_ins_energy_list[ads.name] = np.load(ins_file).tolist()
            if os.path.exists(del_file):
                self.total_del_energy_list[ads.name] = np.load(del_file).tolist()

        vol_file = os.path.join(self.out_folder, f"volume_{macrostate_str}.npy")
        if os.path.exists(vol_file):
            self.volume_list = np.load(vol_file).tolist()

        # Set the base iteration to the length of the list (using the first adsorbate as reference)
        first_ads_name = self.adsorbates[0].name
        if len(self.total_ins_energy_list[first_ads_name]) > 0:
            self.base_iteration = len(self.total_ins_energy_list[first_ads_name])

        self.logger.print_restart_info()

        if os.path.exists(os.path.join(self.out_folder, "Movies", "Trajectory.traj")):
            self.load_state(os.path.join(self.out_folder, "Movies", "Trajectory.traj"))

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

        # Workaround to load the labels from Trajectory.info since ASE's Trajectory does not support custom arrays
        if "labels" in state.info.keys():
            state.set_array("labels", state.info["labels"])

        # Trim off the molecule that was temporarily appended during the last saved insertion attempt
        if "inserted_length" in state.info:
            del state[-state.info["inserted_length"] :]

        self.set_state(state)
        self.n_adsorbates = self.get_number_of_adsorbates(state)

        self.logger.print_load_state_info(n_atoms=len(state))

    def save_results(
        self,
        file_name: str | None = None,
    ) -> None:
        """
        Save a json file with the main results of the simulation.

        Parameters
        ----------
        file_name : str
            Name of the output file. Default uses macrostate string format.
        """
        macrostate_str = self.get_macrostate_str()

        if file_name is None:
            file_name = f"results_{self.T}_{macrostate_str}.json"

        results = {
            "simulation": {
                "code_version": VERSION,
                "random_seed": self.random_seed,
                "temperature_K": self.T,
                "macrostate": self.n_adsorbates,
                "n_steps": len(self.volume_list),
                "enlapsed_time_hours": (datetime.datetime.now() - self.start_time).total_seconds()
                / 3600,
            },
        }

        with open(os.path.join(self.out_folder, file_name), "w") as f:
            json.dump(results, f, indent=4)

    def _save_state(self, actual_iteration: int) -> None:
        """
        Save the simulation trajectory and generated energies.
        """
        if actual_iteration % self.save_every == 0:
            if hasattr(self, "_current_ins_atoms"):
                self.trajectory.write(self._current_ins_atoms)  # type: ignore
            else:
                self.trajectory.write(self.current_system)

            macrostate_str = self.get_macrostate_str()

            for ads in self.adsorbates:
                np.save(
                    os.path.join(self.out_folder, f"ins_energy_{ads.name}_{macrostate_str}.npy"),
                    np.array(self.total_ins_energy_list[ads.name]),
                )
                np.save(
                    os.path.join(self.out_folder, f"del_energy_{ads.name}_{macrostate_str}.npy"),
                    np.array(self.total_del_energy_list[ads.name]),
                )

            np.save(
                os.path.join(self.out_folder, f"volume_{macrostate_str}.npy"),
                np.array(self.volume_list),
            )

    def try_insertion(self, adsorbate_tag: int):
        """
        Try to insert a new adsorbate molecule into the framework.
        This method randomly places the adsorbate in the framework and checks for van der Waals overlap.
        If there is no overlap, it calculates the new potential energy and decides whether to accept the insertion
        based on the acceptance criteria.

        If after a number of tries (self.max_overlap_tries) no valid position is found, an exception is thrown.

        Parameters
        ----------
        adsorbate_tag : int
            The tag of the adsorbate molecule being inserted.

        Returns
        -------
        deltaE
            Insertion energy.
        """
        adsorbate = next((ads for ads in self.adsorbates if ads.tag == adsorbate_tag), None)
        if adsorbate is None:
            raise ValueError(f"Adsorbate with tag {adsorbate_tag} not found.")

        for _ in range(self.max_overlap_tries):
            atoms_trial = random_mol_insertion(
                self.current_system, adsorbate.structure, self.rnd_generator
            )

            overlaped = check_overlap_vesin(
                atoms=atoms_trial,
                group1_indices=np.arange(len(self.current_system)),
                group2_indices=np.arange(len(self.current_system), len(atoms_trial)),
                vdw_radii=self.vdw,
            )
            if overlaped:
                continue

            atoms_trial.calc = self.model
            e_new = atoms_trial.get_potential_energy()

            deltaE = e_new - self.current_total_energy - self.adsorbate_energy[adsorbate.name]

            if np.abs(deltaE) > np.abs(self.max_deltaE):
                continue

            atoms_trial.info["ins_energy"] = deltaE
            atoms_trial.info["n_adsorbates"] = self.n_adsorbates
            atoms_trial.info["inserted_length"] = len(adsorbate.structure)
            self._current_ins_atoms = atoms_trial

            return deltaE

        raise ValueError(f"Could not insert molecule {adsorbate.name}.")

    def try_deletion(self, adsorbate_tag: int):
        """
        Try to delete an adsorbate molecule from the framework.
        This method randomly selects an adsorbate molecule and attempts deletion.

        Parameters
        ----------
        adsorbate_tag : int
            The tag of the adsorbate molecule being deleted.

        Returns
        -------
        deltaE
            Deletion energy.
        """
        ads_name = next((ads.name for ads in self.adsorbates if ads.tag == adsorbate_tag), None)

        if self.n_adsorbates[ads_name] == 0:
            return 0.0

        # Randomly select an adsorbate molecule of this specific tag to delete
        ads_indices_list = self.get_adsorbates_index(tag=adsorbate_tag)
        ads_indices = self.rnd_generator.choice(ads_indices_list, axis=0)

        del_idx = tuple(ads_indices)

        if del_idx in self._del_indices:
            deltaE = self._del_indices[del_idx]
        else:
            atoms_trial = self.current_system.copy()
            atoms_trial.calc = self.model  # type: ignore

            # Delete the selected adsorbate atoms from the trial structure
            del atoms_trial[ads_indices[0] : ads_indices[-1] + 1]

            e_new = atoms_trial.get_potential_energy()  # type: ignore
            deltaE = e_new + self.adsorbate_energy[ads_name] - self.current_total_energy
            self._del_indices[del_idx] = deltaE

        if hasattr(self, "_current_ins_atoms"):
            self._current_ins_atoms.info["del_indices"] = del_idx
            self._current_ins_atoms.info["del_energy"] = deltaE

        return deltaE

    def run(self, N: int) -> None:
        """Run the transition matrix Monte Carlo simulation for N iterations."""

        self.logger.print_run_header()

        for iteration in tqdm(range(1, N + 1), disable=(self.out_file is None), desc="TMMC Step"):
            step_time_start = datetime.datetime.now()

            ins_energies = {}
            del_energies = {}

            # TMMC probes insertion and deletion for all components at the fixed macrostate
            for ads in self.adsorbates:
                ins_energy = self.try_insertion(ads.tag)
                self.total_ins_energy_list[ads.name].append(ins_energy)
                ins_energies[ads.name] = ins_energy

                del_energy = self.try_deletion(ads.tag) if self.n_adsorbates[ads.name] > 0 else 0.0
                self.total_del_energy_list[ads.name].append(del_energy)
                del_energies[ads.name] = del_energy

            self.volume_list.append(self.current_system.get_volume())

            self.logger.print_step_info(
                step=iteration + self.base_iteration,
                del_energy=del_energies,
                ins_energy=ins_energies,
                step_time=(datetime.datetime.now() - step_time_start).total_seconds(),
            )
            self._save_state(iteration + self.base_iteration)
