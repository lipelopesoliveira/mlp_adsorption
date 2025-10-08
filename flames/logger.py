import datetime
import itertools
import os
import platform
import sys
from typing import Optional, TextIO

import ase
import numpy as np
import pymser
from ase import units

from flames import VERSION


class BaseLogger:
    """
    Handles all logging and printing for the simulators.
    Separates the presentation logic from the simulation logic.
    """

    def __init__(self, simulation, output_file: Optional[TextIO] = None):
        """
        Initializes the logger.

        Parameters
        ----------
        simulation : Simulator
            The simulator instance to log.
        output_file : TextIO | None, optional
            A file path or stream to write the output to. If None, prints to stdout.
        """
        self.sim = simulation
        self.out_file = output_file

    def _print(self, *args, **kwargs):
        """Internal print function to direct output to file or console."""
        print(*args, **kwargs, file=self.out_file, flush=True)

    def print_header(self):
        """Prints the header for the simulation output."""
        atomic_numbers = set(
            list(self.sim.framework.get_atomic_numbers())
            + list(self.sim.adsorbate.get_atomic_numbers())
        )

        header = f"""
===========================================================================
         _______  __          ___      .___  ___.  _______     _______.
        |   ____||  |        /   \     |   \/   | |   ____|   /       |
        |  |__   |  |       /  ^  \    |  \  /  | |  |__     |   (----`
        |   __|  |  |      /  /_\  \   |  |\/|  | |   __|     \   \    
        |  |     |  `----./  _____  \  |  |  |  | |  |____.----)   |   
        |__|     |_______/__/     \__\ |__|  |__| |_______|_______/    
                                                               
          Flexible Lattice Adsorption by Monte Carlo Engine Simulation
                        powered by Python + ASE
                    Author: Felipe Lopes de Oliveira
===========================================================================

Code version: {VERSION}
Simulation started at {self.sim.start_time.strftime('%Y-%m-%d %H:%M:%S')}
Hostname: {platform.node()}
OS type: {platform.system()}
OS release: {platform.release()}
OS version: {platform.version()}

Python version: {sys.version.split()[0]}
Numpy version: {np.__version__}
ASE version: {ase.__version__}

Current directory: {os.getcwd()}
Random Seed: {self.sim.random_seed}

Model: {self.sim.model.name}
Running on device: {self.sim.device}

===========================================================================

Constants used:
Boltzmann constant:     {units.kB} eV/K
Beta (1/kT):            {self.sim.beta:.3f} eV^-1
Fugacity coefficient:   {self.sim.fugacity_coeff:.9f} (dimensionless)

===========================================================================

Simulation Parameters:
Temperature: {self.sim.T} K
Pressure: {self.sim.P / 1e5:.5f} bar
Fugacity: {self.sim.fugacity / units.J:.3f} Pa
Fugacity: {self.sim.fugacity:.5e} eV/m^3
(1/kB.T) * V * f = {self.sim.V * self.sim.beta * self.sim.fugacity} [-]

===========================================================================

System Information:
Framework: {self.sim.framework.get_chemical_formula()}
Framework: {self.sim.n_atoms_framework} atoms,
Framework mass: {np.sum(self.sim.framework.get_masses())} g/mol, {self.sim.framework_mass} kg
Framework energy: {self.sim.framework_energy} eV
Framework volume: {self.sim.V} m^3
Framework density: {self.sim.framework_density * 1e3} kg/m^3, {self.sim.framework_density} g/cm^3
Framework cell:
    {self.sim.cell[0, 0]:12.7f} {self.sim.cell[0, 1]:12.7f} {self.sim.cell[0, 2]:12.7f}
    {self.sim.cell[1, 0]:12.7f} {self.sim.cell[1, 1]:12.7f} {self.sim.cell[1, 2]:12.7f}
    {self.sim.cell[2, 0]:12.7f} {self.sim.cell[2, 1]:12.7f} {self.sim.cell[2, 2]:12.7f}

Perpendicular cell:
    {self.sim.perpendicular_cell[0, 0]:12.7f} {self.sim.perpendicular_cell[0, 1]:12.7f} {self.sim.perpendicular_cell[0, 2]:12.7f}
    {self.sim.perpendicular_cell[1, 0]:12.7f} {self.sim.perpendicular_cell[1, 1]:12.7f} {self.sim.perpendicular_cell[1, 2]:12.7f}
    {self.sim.perpendicular_cell[2, 0]:12.7f} {self.sim.perpendicular_cell[2, 1]:12.7f} {self.sim.perpendicular_cell[2, 2]:12.7f}

"""
        if self.sim.get_ideal_supercell() != [1, 1, 1]:
            header += f"""
WARNING: Ideal supercell size is {self.sim.get_ideal_supercell()} (x, y, z).
Consider using automatic_supercell=True to create a supercell that
fits the cutoff radius of {self.sim.cutoff} Å or manually create a supercell.

"""

        header += "Atomic positions:\n"

        for atom in self.sim.framework:
            header += "  {:2} {:12.7f} {:12.7f} {:12.7f}\n".format(atom.symbol, *atom.position)

        header += f"""
===========================================================================
Adsorbate: {self.sim.adsorbate.get_chemical_formula()}
Adsorbate: {self.sim.n_adsorbate_atoms} atoms, {self.sim.adsorbate_mass} kg
Adsorbate energy: {self.sim.adsorbate_energy} eV

Atomic positions:
"""
        for atom in self.sim.adsorbate:
            header += "  {:2} {:12.7f} {:12.7f} {:12.7f}\n".format(atom.symbol, *atom.position)

        header += """
===========================================================================
Shortest distances:
"""

        for i, j in list(itertools.combinations(atomic_numbers, 2)):
            header += f"  {ase.Atom(i).symbol:2} - {ase.Atom(j).symbol:2}: {self.sim.vdw[i] + self.sim.vdw[j]:.3f} Å\n"

        header += f"""
===========================================================================
Conversion factors:
    Conversion factor molecules/unit cell -> mol/kg:         {self.sim.conv_factors['mol/kg']:.9f}
    Conversion factor molecules/unit cell -> mg/g:           {self.sim.conv_factors['mg/g']:.9f}
    Conversion factor molecules/unit cell -> cm^3 STP/gr:    {self.sim.conv_factors['cm^3 STP/gr']:.9f}
    Conversion factor molecules/unit cell -> cm^3 STP/cm^3:  {self.sim.conv_factors['cm^3 STP/cm^3']:.9f}
    Conversion factor molecules/unit cell -> %wt:            {self.sim.conv_factors['mg/g'] * 1e-3:.9f}

Partial pressure:
        {self.sim.P:>15.5f} Pascal
        {self.sim.P / 1e5:>15.5f} bar
        {self.sim.P / 101325:>15.5f} atm
        {self.sim.P / (101325 * 760):>15.5f} Torr
===========================================================================
"""
        self._print(header)

    def print_restart_info(self) -> None:
        """Prints information when a simulation is restarted."""
        state = self.sim.current_system
        avg_binding_energy = (
            (
                self.sim.current_total_energy
                - self.sim.framework_energy
                - self.sim.N_ads * self.sim.adsorbate_energy
            )
            / (units.kJ / units.mol)
            / self.sim.N_ads
            if self.sim.N_ads > 0
            else 0
        )
        self._print(f"Restarting simulation from step {self.sim.base_iteration}...")
        self._print(
            f"""
===========================================================================
Restart file requested.
Loaded state with {len(state)} total atoms.
Current total energy: {self.sim.current_total_energy:.3f} eV
Current number of adsorbates: {self.sim.N_ads}
Current average binding energy: {avg_binding_energy:.3f} kJ/mol
===========================================================================
"""
        )

    def print_debug_movement(self, movement, deltaE, prefactor, acc, rnd_number) -> None:
        """
        Print debug information about the current state of the simulation.
        This method is called to provide detailed information about the current state of the system.
        """
        self._print(
            f"""
=======================================================================================================
Movement type: {movement}
Interaction energy: {deltaE} eV, {(deltaE / (units.kJ / units.mol))} kJ/mol
Exponential factor:     {-self.sim.beta * deltaE:.3E}
Exponential:            {np.exp(-self.sim.beta * deltaE):.3E}
Prefactor:              {prefactor:.3E}
Acceptance probability: {acc:.3f}
Random number:          {rnd_number:.3f}
Accepted: {rnd_number < acc}
=======================================================================================================
"""
        )


class GCMCLogger(BaseLogger):
    """
    Handles all logging and printing for a GCMC simulation.
    Separates the presentation logic from the simulation logic.
    """

    def __init__(self, simulation, output_file: Optional[TextIO] = None):
        """
        Initializes the logger.

        Parameters
        ----------
        simulation : GCMC
            The GCMC simulation instance to log.
        output_file : TextIO | None, optional
            A file path or stream to write the output to. If None, prints to stdout.
        """
        self.sim = simulation
        self.out_file = output_file

    def print_run_header(self) -> None:
        """Prints the header for the main GCMC loop."""
        header = """
===========================================================================

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Starting GCMC simulation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 Iteration |  Number of  |  Uptake  |    Tot En.   |Av. Ads. En.|  Pacc  |  Pdel  |  Ptra  |  Prot  |  Time
      -    |  Molecules  | [mmol/g] |     [eV]     |  [kJ/mol]  |    %   |    %   |   %    |   %    |   [s]
---------- | ----------- | -------- | ------------ | ---------- | ------ | ------ | ------ | ------ | ------"""
        self._print(header)

    def print_step_info(self, step, average_ads_energy, step_time) -> None:

        line_str = "{:^11}|{:^13}|{:>9.2f} |{:>13.4f} |{:>11.4f} |{:7.2f} |{:7.2f} |{:7.2f} |{:7.2f} |{:9.2f}"

        self._print(
            line_str.format(
                step,
                self.sim.N_ads,
                self.sim.N_ads * self.sim.conv_factors["mol/kg"],
                self.sim.current_total_energy,
                average_ads_energy,
                (
                    np.average(self.sim.mov_dict["insertion"]) * 100
                    if len(self.sim.mov_dict["insertion"]) > 0
                    else 0
                ),
                (
                    np.average(self.sim.mov_dict["deletion"]) * 100
                    if len(self.sim.mov_dict["deletion"]) > 0
                    else 0
                ),
                (
                    np.average(self.sim.mov_dict["translation"]) * 100
                    if len(self.sim.mov_dict["translation"]) > 0
                    else 0
                ),
                (
                    np.average(self.sim.mov_dict["rotation"]) * 100
                    if len(self.sim.mov_dict["rotation"]) > 0
                    else 0
                ),
                step_time,
            )
        )

    def print_optimization_start(self, target: str) -> None:
        """Prints a header for framework or adsorbate optimization."""
        self._print(
            f"""
===========================================================================
Start optimizing {target} structure...
===========================================================================
"""
        )

    def print_load_state_info(self, n_atoms, average_ads_energy):
        """Prints information about the loading state."""
        self._print(
            f"""
===========================================================================

Restarting GCMC simulation from previous configuration...

Loaded state with {n_atoms} total atoms.

Current total energy: {self.sim.current_total_energy:.3f} eV
Current number of adsorbates: {self.sim.N_ads}
Current average binding energy: {average_ads_energy:.3f} kJ/mol

Current steps are: {self.sim.base_iteration}

===========================================================================
"""
        )

    def print_iteration_info(self, iteration_data: dict) -> None:
        """Prints a single log line for a GCMC iteration."""
        line_str = "{:^11}|{:^13}|{:>9.2f} |{:>13.4f} |{:>11.4f} |{:7.2f} |{:7.2f} |{:7.2f} |{:7.2f} |{:9.2f}"
        self._print(line_str.format(*iteration_data.values()))

    def print_debug_movement(self, movement, deltaE, prefactor, acc, rnd_number) -> None:
        """Prints detailed debug information for a single MC move."""
        self._print(
            f"""
=======================================================================================================
Movement type: {movement}
Current number of adsorbates: {self.sim.N_ads}
Interaction energy: {deltaE} eV, {(deltaE / (units.kJ / units.mol))} kJ/mol
Exponential factor:     {-self.sim.beta * deltaE:.3E}
Exponential:            {np.exp(-self.sim.beta * deltaE):.3E}
Prefactor:              {prefactor:.3E}
Acceptance probability: {acc:.3f}
Random number:          {rnd_number:.3f}
Accepted: {rnd_number < acc}
=======================================================================================================
"""
        )

    def print_summary(self) -> None:
        """Prints the final summary of the simulation results."""

        eq_results = pymser.equilibrate(
            self.sim.uptake_list,
            LLM=True,
            batch_size=int(len(self.sim.uptake_list) / 50),
            ADF_test=False,
            uncertainty="uSD",
            print_results=False,
        )

        avg_uptake = eq_results["average"]
        std_uptake = eq_results["uncertainty"]

        enthalpy, enthalpy_sd = pymser.calc_equilibrated_enthalpy(
            energy=np.array(self.sim.total_ads_list) / units.kB,  # Convert to K
            number_of_molecules=self.sim.uptake_list,
            temperature=self.sim.T,
            eq_index=eq_results["t0"],
            uncertainty="SD",
            ac_time=int(eq_results["ac_time"]),
        )

        self._print(
            f"""
===========================================================================

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Finishing GCMC simulation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    pyMSER Equilibration Results:
    ------------------------------------------------------------------------------
    Start of equilibrated data:          {eq_results['t0']} of {len(self.sim.uptake_list)}
    Total equilibrated steps:            {len(self.sim.uptake_list) - eq_results['t0']}  ({(len(self.sim.uptake_list) - eq_results['t0']) / len(self.sim.uptake_list) * 100:.2f}%)
    Equilibrated:                        {eq_results['t0'] < 0.75 * len(self.sim.uptake_list)}
    Average over equilibrated data:      {eq_results['average']:.4f} ± {eq_results['uncertainty']:.4f} molecules/unit cell
    Number of uncorrelated samples:      {eq_results['uncorr_samples']:.1f}
    Autocorrelation time:                {eq_results['ac_time']:.1f}
    ------------------------------------------------------------------------------
    
    Average properties of the system:
    ------------------------------------------------------------------------------
    Average loading absolute [molecules/unit cell]       {avg_uptake:12.5f} +/- {std_uptake:12.5f} [-]
    Average loading absolute [mol/kg framework]          {avg_uptake * self.sim.conv_factors["mol/kg"]:12.5f} +/- {std_uptake * self.sim.conv_factors["mol/kg"]:12.5f} [-]
    Average loading absolute [mg/g framework]            {avg_uptake * self.sim.conv_factors["mg/g"]:12.5f} +/- {std_uptake * self.sim.conv_factors["mg/g"]:12.5f} [-]
    Average loading absolute [cm^3 (STP)/gr framework]   {avg_uptake * self.sim.conv_factors["cm^3 STP/gr"]:12.5f} +/- {std_uptake * self.sim.conv_factors["cm^3 STP/gr"]:12.5f} [-]
    Average loading absolute [cm^3 (STP)/cm^3 framework] {avg_uptake * self.sim.conv_factors["cm^3 STP/cm^3"]:12.5f} +/- {std_uptake * self.sim.conv_factors["cm^3 STP/cm^3"]:12.5f} [-]
    Average loading absolute [%wt framework]             {avg_uptake * self.sim.conv_factors["mg/g"] * 1e-3:12.5f} +/- {std_uptake * self.sim.conv_factors["mg/g"] * 1e-3:12.5f} [-]


    Enthalpy of adsorption: [kJ/mol]                     {enthalpy:12.5f} +/- {enthalpy_sd:12.5f} [kJ/mol]

===========================================================================
GCMC simulation finished successfully!
===========================================================================

Simulation finished at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Simulation duration: {datetime.datetime.now() - self.sim.start_time}
===========================================================================
"""
        )


class WidomLogger(BaseLogger):
    """
    Handles all logging and printing for a Widom insertion simulation.
    Separates the presentation logic from the simulation logic.
    """

    def __init__(self, simulation, output_file: Optional[TextIO] = None):
        """
        Initializes the logger.

        Parameters
        ----------
        simulation : Widom
            The Widom simulation instance to log.
        output_file : TextIO | None, optional
            A file path or stream to write the output to. If None, prints to stdout.
        """
        self.sim = simulation
        self.out_file = output_file

    def _print(self, *args, **kwargs):
        """Internal print function to direct output to file or console."""
        print(*args, **kwargs, file=self.out_file, flush=True)

    def print_run_header(self):
        """Prints the header for the main Widom loop."""
        header = """
===========================================================================

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Starting Widom simulation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Iteration  |  dE (eV)  |  dE (kJ/mol)  | kH [mol kg-1 Pa-1]  |  dH (kJ/mol) | Time (s)
---------------------------------------------------------------------------------------"""
        self._print(header)

    def print_iteration_info(self, iteration_data: list):
        """Prints a single log line for a Widom iteration."""
        line_str = "{:^10} | {:^9.6f} | {:>13.2f} | {:>19.3e} | {:12.2f} | {:8.2f}"
        self._print(line_str.format(*iteration_data))

    def print_summary(self):
        """
        Print the footer for the simulation output.
        This method is called at the end of the simulation to display the final results and elapsed time.
        """

        self._print(
            f"""
===========================================================================

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Finishing Widom simulation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    Average properties of the system:
    ------------------------------------------------------------------------------
    Henry coefficient: [mol/kg/Pa]                       {self.sim.kH:12.5e} +/- {0.0:12.5e} [-]
    Enthalpy of adsorption: [kJ/mol]                     {self.sim.Qst:12.5f} +/- {0.0:12.5f} [-]

===========================================================================
Simulation finished successfully!
===========================================================================

Simulation finished at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Simulation duration: {datetime.datetime.now() - self.sim.start_time}
===========================================================================
"""
        )
