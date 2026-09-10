=================================
Multi-component GCMC Simulation
=================================


Brief introduction about multi-component GCMC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While single-component Grand Canonical Monte Carlo (GCMC) simulations are ideal for generating pure gas adsorption isotherms, real-world applications rarely involve pure gases. Processes like post-combustion carbon capture, gas separation, and air purification involve gas mixtures.

Multi-component GCMC extends the stochastic μVT ensemble to handle multiple guest species simultaneously. In these simulations, the algorithm not only decides whether to insert or delete molecules from the rigid framework, but also determines *which* chemical species to attempt these moves on. This allows the simulation of competitive adsorption (co-adsorption), providing critical insight into the selectivity of porous materials like MOFs and COFs under mixed-gas conditions.

----

Setting the stage
~~~~~~~~~~~~~~~~~

In this example, we will evaluate the performance of Mg-MOF-74 in the presence of a binary gas mixture. We will model a mixture containing 70% CO\ :sub:`2` and 30% H\ :sub:`2`\ O. This simulates a simplified wet/humid environment or mixed gas stream to observe how water competes with carbon dioxide for the open-metal sites.

On the ``flames/examples/Basic/4-Multicomponent_GCMC`` folder you will find the ``cif`` file for the Mg-MOF-74. The simulation will again be executed with the machine learning potential MACE, using the ``medium-0b2`` pre-trained foundation model. 

The script below will run the simulation at 298 K (25°C) and 1,000,000 Pa (10 bar).

.. code:: python

    import os

    # Hide UserWarning and RuntimeWarning messages
    import warnings

    import ase
    import torch
    from ase.data import vdw_radii
    from ase.io import read
    from mace.calculators import mace_mp

    from flames.adsorbate import Adsorbate
    from flames.gcmc import GCMC

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the framework structure
    framework = read("mg-mof-74.cif")  # type: ignore

    adsorbate_1 = Adsorbate(
        name="CO2",
        structure="co2.xyz",
        eos={"criticalTemperature": 304.1282, "criticalPressure": 7377300.0, "acentricFactor": 0.22394},
        mol_fraction=0.7,
    )

    adsorbate_2 = Adsorbate(
        name="H2O",
        structure="H2O.xyz",
        eos={"criticalTemperature": 628.0, "criticalPressure": 14100000.0, "acentricFactor": 0.5293},
        mol_fraction=0.3,
    )

    model = mace_mp(
        model="medium-0b2",
        dispersion=False,
        damping="zero",  # choices: ["zero", "bj", "zerom", "bjm"]
        dispersion_xc="pbe",
        default_dtype="float32",
        device=device,
    )

    Temperature = 298.0  # in Kelvin
    pressure = 1_000_000  # in Pa = 10 bar
    MCSteps = 100


    print(
        f"Running GCMC simulation for pressure: {pressure:.2f} Pa at temperature: {Temperature:.2f} K"
    )

    gcmc = GCMC(
        model=model,
        framework_atoms=framework,
        adsorbates=[adsorbate_1, adsorbate_2],
        temperature=Temperature,
        pressure=pressure,
        device=device,
        vdw_radii=vdw_radii,
        vdw_factor=0.6,
        save_frequency=1,
        debug=True,
        output_to_file=True,
        random_seed=42,
        cutoff_radius=6.0,
        automatic_supercell=True,
    )

    gcmc.logger.print_header()

    gcmc.run(MCSteps)

    gcmc.logger.print_summary()

    gcmc.save_results()

----

Breaking down the input script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The overall structure of the script is identical to a single-component simulation, but it requires explicit definition of partial thermodynamic properties.

Defining Multiple Adsorbates
----------------------------

.. code:: python

    adsorbate_1 = Adsorbate(
        name="CO2",
        structure="co2.xyz",
        eos={"criticalTemperature": 304.1282, "criticalPressure": 7377300.0, "acentricFactor": 0.22394},
        mol_fraction=0.7,
    )

    adsorbate_2 = Adsorbate(
        name="H2O",
        structure="H2O.xyz",
        eos={"criticalTemperature": 628.0, "criticalPressure": 14100000.0, "acentricFactor": 0.5293},
        mol_fraction=0.3,
    )

Notice that we now instantiate two separate ``Adsorbate`` objects. The most critical new parameter here is ``mol_fraction``. This dictates the bulk phase composition of the gas mixture outside the framework. 
FLAMES uses the specified molar fractions along with the total pressure to calculate the partial pressure and partial fugacity of each component using the appropriate Equation of State (EOS). 

.. warning:: Ensure that the sum of the ``mol_fraction`` parameters across all your defined adsorbates is exactly equal to 1.0.

Running the simulation
----------------------

.. code:: python

    gcmc = GCMC(
        model=model,
        framework_atoms=framework,
        adsorbates=[adsorbate_1, adsorbate_2],
        # ... remaining parameters
    )

When instantiating the ``GCMC`` simulator object, instead of passing a single ``Adsorbate`` variable, we pass a Python list containing all the species we wish to simulate: ``[adsorbate_1, adsorbate_2]``. FLAMES will automatically recognize this as a multicomponent simulation and distribute the Monte Carlo moves accordingly.

Analyzing the output
~~~~~~~~~~~~~~~~~~~~

The system information
----------------------

In the terminal output, the system information block will now calculate bulk properties for each component separately:

.. code:: none

    ===========================================================================
    Adsorbate 1: CO2
    Adsorbate: 3 atoms, 7.307866e-26 kg
    ...
    Equation of State Parameters:
        MolFraction:           0.7000000000 [-]
        Bulk phase pressure:   700000.000000 [Pa]
    ...
    ===========================================================================
    Adsorbate 2: H2O
    Adsorbate: 3 atoms, 2.991500e-26 kg
    ...
    Equation of State Parameters:
        MolFraction:           0.3000000000 [-]
        Bulk phase pressure:   300000.000000 [Pa]
    ===========================================================================

The partial pressure is derived from your total specified pressure and the molar fractions. The acceptance probabilities for insertion/deletion will be weighted by the respective fugacity of each component.

The simulation and results
--------------------------

During the step-by-step reporting, FLAMES handles the tracking of total energy, but the resulting JSON data provides granular details split by species. 

When the simulation completes, your ``results_[TEMPERATURE]_[PRESSURE].json`` file will organize the uptakes (absolute and excess) as dictionaries partitioned by the adsorbate names you provided.

.. code:: python

    {
        "simulation": {
            "code_version": "0.4.5A",
            "temperature_K": 298.0,
            "pressure_Pa": 1000000,
            "n_steps": 100000
        },
        "absolute_uptake": {
            "CO2": {
                "mol/kg": {
                    "mean": 3.841,
                    "sd": 0.312
                },
                "mg/g": {
                    "mean": 169.04,
                    "sd": 13.7
                }
            },
            "H2O": {
                "mol/kg": {
                    "mean": 6.120,
                    "sd": 0.401
                },
                "mg/g": {
                    "mean": 110.25,
                    "sd": 7.2
                }
            }
        },
        "enthalpy": {
            "CO2": { ... },
            "H2O": { ... }
        }
    }

From this JSON, it's easy to calculate the **Selectivity** of the framework for one gas over another by comparing the ratio of adsorbed molar loadings to the ratio of their bulk molar fractions.

.. note:: When analyzing multi-component systems, plotting your convergence curves (Uptake vs. Steps) becomes twice as important. Due to competitive displacement (where a strongly adsorbing species like H2O slowly pushes out a weakly adsorbing one), it often takes significantly more MC steps to reach true equilibration in a mixture than in a pure gas. Ensure the moving averages for *all* component curves have flattened out.