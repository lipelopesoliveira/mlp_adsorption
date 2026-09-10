===========================
Hybrid GCMC/MD Simulation
===========================


Brief introduction about Hybrid GCMC/MD simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard Grand Canonical Monte Carlo (GCMC) simulations typically treat the porous host material as a rigid lattice. While efficient, this rigid approximation fails for materials that exhibit structural flexibility, such as "breathing" MOFs (e.g., MIL-53) or soft porous crystals. Furthermore, in highly packed pores, standard Monte Carlo translation and rotation moves can suffer from extremely low acceptance rates, leaving the system trapped in local energy minima.

Hybrid GCMC/MD bridges this gap by alternating between stochastic GCMC steps and deterministic Molecular Dynamics (MD) trajectories. The GCMC phase handles the exchange of molecules with the theoretical reservoir (insertions and deletions) to establish the correct chemical potential. The MD phase allows the atomic positions of both the framework and the guest molecules to evolve naturally over time, relaxing structural stresses, exploring configurational space, and allowing the unit cell to expand or contract.

----

Setting the stage
~~~~~~~~~~~~~~~~~

In this example, we will run a hybrid simulation on Mg-MOF-74 loaded with CO:sub:`2`. While Mg-MOF-74 is relatively rigid compared to breathing MOFs, a hybrid approach allows the guest molecules to find optimal configurations around the open-metal sites more effectively than random MC moves alone.

The simulation alternates 5 times between 3,000 GCMC steps and 3,000 MD steps.

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

    FrameworkPath = "mg-mof-74.cif"
    AdsorbatePath = "co2.xyz"

    model = mace_mp(
        model="medium-0b2",
        dispersion=False,
        damping="zero",  # choices: ["zero", "bj", "zerom", "bjm"]
        dispersion_xc="pbe",
        default_dtype="float32",
        device=device,
    )

    # Load the framework structure
    framework: ase.Atoms = read(FrameworkPath)  # type: ignore

    # Load the adsorbate structure
    adsorbate = Adsorbate(
        name="CO2",
        structure=AdsorbatePath,
        eos={"criticalTemperature": 304.1282, "criticalPressure": 7377300.0, "acentricFactor": 0.22394},
    )

    Temperature = 298.0  # in Kelvin
    pressure = 100_000  # in Pa = 1 bar
    MCSteps = 3_000
    MDSteps = 3_000

    print(
        f"Running GCMC simulation for pressure: {pressure:.2f} Pa at temperature: {Temperature:.2f} K"
    )

    gcmc = GCMC(
        model=model,
        framework_atoms=framework,
        adsorbates=adsorbate,
        temperature=Temperature,
        pressure=pressure,
        device=device,
        vdw_radii=vdw_radii,
        vdw_factor=0.6,
        save_frequency=1,
        debug=False,
        output_to_file=True,
        cutoff_radius=6.0,
        automatic_supercell=True,
    )

    gcmc.logger.print_header()

    for j in range(5):
        gcmc.run(MCSteps)
        gcmc.md(
            nsteps=MDSteps,
            time_step=0.5,
            ensemble="NPT",
            thermostat="MTK",
            movie_interval=1,
        )

    gcmc.run(MCSteps)
    gcmc.logger.print_summary()
    gcmc.save_results()

----

Breaking down the input script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The initial setup, including the machine learning potential, framework structure, and adsorbate definition, remains unchanged. The critical difference lies in the execution loop at the end of the script.

The Hybrid Execution Loop
-------------------------

.. code:: python

    for j in range(5):
        gcmc.run(MCSteps)
        gcmc.md(
            nsteps=MDSteps,
            time_step=0.5,
            ensemble="NPT",
            thermostat="MTK",
            movie_interval=1,
        )

    gcmc.run(MCSteps)

This loop creates the hybrid alternating cycle. 

1. **``gcmc.run(MCSteps)``**: The system attempts 3,000 GCMC moves (insertions, deletions, translations, rotations). The framework remains frozen during this specific function call.
2. **``gcmc.md(...)``**: The system switches to Molecular Dynamics for 3,000 steps. Particle insertions and deletions are paused. Both the framework atoms and the currently adsorbed molecules move according to Newton's equations of motion, guided by the forces calculated by the MACE potential.
3. **Final ``gcmc.run(MCSteps)``**: The script finishes with a pure GCMC block to ensure the final statistics reflect the equilibrium particle exchange with the reservoir.

Ensembles and MD Drivers Available
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When calling ``gcmc.md()``, you must specify the ``ensemble`` and ``thermostat`` (or barostat) driver. FLAMES supports three main thermodynamic ensembles and multiple algorithms to control temperature and pressure. You can fine-tune these drivers by passing specific keyword arguments (``kwargs``) directly to the ``gcmc.md()`` call.

NVE (Microcanonical Ensemble)
-----------------------------
Simulates an isolated system with a constant Number of particles, Volume, and Energy. No external temperature or pressure control is applied.
 
* **Driver:** ``"velocityverlet"``
* **Kwargs:** No additional parameters required.

NVT (Canonical Ensemble)
------------------------
Maintains a constant Number of particles, Volume, and Temperature. The unit cell remains completely rigid, but the kinetic energy is scaled to match the target temperature.

* **Berendsen** (``thermostat="berendsen"``): 
  Uses a weak coupling scheme to an external heat bath. It is excellent for bringing a system quickly to the target temperature but does not sample the true canonical ensemble.
  
  * ``taut``: Time constant for the thermostat in fs (default: 1.0).

* **Nose-Hoover** (``thermostat="nosehoover"``): 
  Uses a deterministic continuous dynamics method (Nose-Hoover chains) that accurately samples the true NVT ensemble.
  
  * ``tdamp``: Time constant for temperature fluctuations in fs (default: 50.0).
  * ``tchain``: Number of thermostats in the chain (default: 3).
  * ``tloop``: Number of integration loops (default: 1).

* **Langevin** (``thermostat="langevin"``): 
  Uses stochastic dynamics, adding a friction term and random forces to mimic collisions with a fictitious solvent/bath.
  
  * ``friction``: Friction coefficient (default: 0.01).

NPT (Isothermal-Isobaric Ensemble)
----------------------------------
Maintains a constant Number of particles, Pressure, and Temperature. The unit cell vectors are allowed to change in length and angle, which is essential for modeling the swelling or "breathing" of flexible frameworks.

* **MTK** (``thermostat="mtk"``): 
  The Martyna-Tobias-Klein algorithm. This is the **gold standard** for NPT simulations in modern materials science, as it rigorously samples the isothermal-isobaric ensemble.
  
  * ``tdamp``: Thermostat time constant in fs (default: 50.0).
  * ``pdamp``: Barostat time constant in fs (default: 500.0).
  * ``vol_constraint``: If ``True``, the volume is kept constant while the cell shape (angles and relative vector lengths) can fluctuate.
  * ``isotropic``: If ``True``, forces the unit cell to expand/contract equally in all directions (cannot be used with ``vol_constraint=True``).

* **Berendsen** (``thermostat="berendsen"``): 
  Similar to its NVT counterpart, this applies an exponential scaling to both coordinates and box vectors. Good for fast initial volume relaxation.
  
  * ``isotropic``: Whether to use isotropic pressure coupling (default: False).
  * ``compressibility``: Material compressibility in bar\ :sup:`-1` (default: 1e-4).
  * ``taup``: Barostat time constant in fs (default: 500.0).

* **Nose-Hoover / Melchionna** (``thermostat="nosehoover"``): 
  An implementation of the Nose-Hoover dynamics modified by Melchionna for NPT systems.
  
  * ``ttime``: Thermostat time constant in fs (default: 25.0).
  * ``ptime``: Parrinello-Rahman barostat time constant in fs (default: 75.0).
  * ``bulk_modulus``: Material bulk modulus in GPa (default: 30.0).

----

Under the Hood: Terminal Logging and Trajectories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the MD wrapper takes over, the terminal output seamlessly shifts to reflect the dynamics. The ``_md_core`` function attaches an internal logger that prints the system's states at the interval defined by ``output_interval``.

.. code:: none

    ======================================================================================
    Starting NPT MD Simulation using NPT-Mtk

    General Parameters:
        Temperature: 298.00 K
        Time Step: 0.50 fs
        Number of MD Steps: 3000
        Output Interval: 100 steps
        Movie Interval: 1 steps

    Method-Specific Parameters:
        Driver: MTKNPT
        Trajectory File: ./NPT-mtk_298.00K_0.traj
        pdamp: 500.0
        tdamp: 50.0
        tchain: 3
        pchain: 3
    
        Step   |  Pot. Energy   |  Kin. Energy   |  Total Energy  |  Temperature  |  Stress  |   Volume    | Elapsed Time 
        [-]    |      [eV]      |      [eV]      |      [eV]      |      [K]      |   [GPa]  |    [A^3]    |      [s]      
     --------- | -------------- | -------------- | -------------- | ------------- | -------- | ----------- | -------------
            0  |   -2351.243408 |     112.564100 |   -2238.679308 |      298.000  |    -0.00 |     8241.97 |       0.0
          100  |   -2348.112341 |     108.412000 |   -2239.700341 |      286.321  |     0.02 |     8245.12 |       2.1
          ...

Simultaneously, a trajectory file (``.traj`` format) is generated. If you run multiple MD blocks via a ``for`` loop (as seen in the hybrid simulation script), FLAMES prevents overwriting by appending an index to the filename (e.g., ``NPT-Mtk_298.00K_0.traj``, ``NPT-Mtk_298.00K_1.traj``). You can open these trajectory files directly in the ``ase-gui`` to visualize the framework breathing and guest molecule diffusion over time.