========================================
Ewald Sum for Electrostatic Interactions
========================================


Brief introduction about the Ewald sum method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Ewald sum is a method for computing long-range electrostatic interactions in periodic systems. The Ewald sum decomposes the electrostatic potential into two parts: a short-range part that converges rapidly in real space and a long-range part that converges in reciprocal space.
This decomposition allows for efficient computation of the electrostatic interactions while maintaining accuracy.


----

Setting the stage
~~~~~~~~~~~~~~~~~

In this example we will evaluate the Mg-MOF-74, a material well-known for having adsorption of CO :sub:`2` on Open-Metal sites.

On the ``flames/examples/Calculators/ClassicalFF`` folder you will find the ``cif`` file for the Mg-MOF-74 [1]_. contaning the charges calculated with the DDEC method.

Note that the partial charges informations are stored in the ``_atom_site_charge`` tag of the cif file, and the atom types are stored in the ``_atom_site_label`` tag. The calculator will use these informations to calculate the electrostatic interactions and the Lennard-Jones interactions, using the parameters from the json file.

.. code:: none

    data_MgMOF-74_DDEC
    _chemical_name_common                  'MgMOF-74_DDEC'
    _cell_length_a                              6.870000000
    _cell_length_b                             15.112181400
    _cell_length_c                             15.112181400
    _cell_angle_alpha                         117.746240480
    _cell_angle_beta                           98.715798920
    _cell_angle_gamma                          98.715798920

    _symmetry_cell_setting          triclinic
    _symmetry_space_group_name_Hall 'P 1'
    _symmetry_space_group_name_H-M  'P 1'
    _symmetry_Int_Tables_number     1

    _symmetry_equiv_pos_as_xyz 'x,y,z'

    loop_
    _atom_site_type_symbol
    _atom_site_label
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_charge
    Mg    Mg         0.5186500000      0.4093400000      0.7319800000    1.4148400
    Mg    Mg         0.4814500000      0.5907400000      0.2681800000    1.4152260
    Mg    Mg         0.7867500000      0.2681200000      0.6775000000    1.4150490
    Mg    Mg         0.2133500000      0.7319200000      0.3227000000    1.4149810
    Mg    Mg         0.1093700000      0.3226800000      0.5908000000    1.4152670
    Mg    Mg         0.8907700000      0.6774800000      0.4094000000    1.4153640
    H     H          0.1869650000      0.5308500000      0.5511000000    0.1423580
    H     H          0.8131650000      0.4692500000      0.4491000000    0.1422210
    H     H          0.6359650000      0.4490200000      0.9799300000    0.1428330
    H     H          0.3641650000      0.5510200000      0.0203300000    0.1425470
    H     H          0.6561950000      0.0203100000      0.4693300000    0.1427900
    H     H          0.3439950000      0.9799100000      0.5309300000    0.1426060
    C     C          0.7398300000      0.3850400000      0.5624800000    0.6837900
    C     C          0.2604300000      0.6150400000      0.4376800000    0.6836320
    C     C          0.1774300000      0.4376200000      0.8227000000    0.6838090
    C     C          0.8228300000      0.5624200000      0.1775000000    0.6836940
    C     C          0.3548500000      0.1774800000      0.6151000000    0.6839840
    C     C          0.6454500000      0.8226800000      0.3851000000    0.6829560
    C     C          0.6148300000      0.4491400000      0.5376800000   -0.1773490
    C     C          0.3854300000      0.5509400000      0.4624800000   -0.1773120
    C     C          0.0772300000      0.4624300000      0.9115900000   -0.1770960
    C     C         0.9230300000      0.5376300000      0.0885900000   -0.1775100
    C     C         0.1657400000      0.0885600000      0.5509900000   -0.1772920
    C     C         0.8345400000      0.9115600000      0.4491900000   -0.1770670
    C     C         0.4378500000      0.4679400000      0.5698800000    0.3282120
    C     C         0.5622500000      0.5321400000      0.4302800000    0.3285260
    C     C         0.8680500000      0.4302200000      0.8982000000    0.3283210
    C     C         0.1320500000      0.5698200000      0.1020000000    0.3288670
    C     C         0.9699700000      0.1019800000      0.5322000000    0.3285720
    C     C         0.0301700000      0.8981800000      0.4680000000    0.3290870
    C     C         0.3255300000      0.5227400000      0.5351800000   -0.2087990
    C     C         0.6747300000      0.4773400000      0.4649800000   -0.2086970
    C     C         0.7904300000      0.4649200000      0.9877000000   -0.2092150
    C     C         0.2098300000      0.5351200000      0.0125000000   -0.2099140
    C     C         0.8028500000      0.0124800000      0.4774000000   -0.2096940
    C     C         0.1974500000      0.9876800000      0.5228000000   -0.2095000
    O     O          0.6872510000      0.3548400000      0.6208800000   -0.7387460
    O     O          0.3128510000      0.6452400000      0.3792800000   -0.7387570
    O     O          0.0664510000      0.3792200000      0.7341000000   -0.7387300
    O     O          0.9336510000      0.6208200000      0.2661000000   -0.7387400
    O     O          0.3324710000      0.2660800000      0.6453000000   -0.7390080
    O     O          0.6676710000      0.7340800000      0.3549000000   -0.7383650
    O     O          0.9004500000      0.3768400000      0.5316800000   -0.6324900
    O     O          0.0996500000      0.6232400000      0.4684800000   -0.6328050
    O     O          0.3688500000      0.4684200000      0.8453000000   -0.6325900
    O     O         0.6312500000      0.5316200000      0.1549000000   -0.6327200
    O     O         0.5236700000      0.1548800000      0.6233000000   -0.6323300
    O     O         0.4764700000      0.8452800000      0.3769000000   -0.6322220
    O     O         0.3562530000      0.4334400000      0.6283800000   -0.8119520
    O     O         0.6438530000      0.5666400000      0.3717800000   -0.8121590
    O     O         0.7279530000      0.3717200000      0.8052000000   -0.8116690
    O     O         0.2721530000      0.6283200000      0.1950000000   -0.8122470
    O     O         0.9228730000      0.1949800000      0.5667000000   -0.8121360
    O     O         0.0772730000      0.8051800000      0.4335000000   -0.8124180


For the Lennard Jones interactions, the parameters are taken from the UFF force field, and are stored in the ``UFF_lj_params.json`` file. The calculator will use the atom types from the ``_atom_site_label`` tag to assign the parameters to the atoms in the structure.


.. code:: json
    
    {
        "O": {
            "sigma": 3.03315,
            "epsilon": 48.1581
        },
        "N": {
            "sigma": 3.26256,
            "epsilon": 38.9492
        },
        "C": {
            "sigma": 3.47299,
            "epsilon": 47.8562
        },
        "F": {
            "sigma": 3.0932,
            "epsilon": 36.4834
        },
        "B": {
            "sigma": 3.58141,
            "epsilon": 47.8058
        },
        "I": {
            "sigma": 4.01,
            "epsilon": 170.57
        },
        "P": {
            "sigma": 3.69723,
            "epsilon": 161.03
        },
        "S": {
            "sigma": 3.59032,
            "epsilon": 173.107
        },
        "W": {
            "sigma": 2.73,
            "epsilon": 33.71
        },
        "Y": {
            "sigma": 2.98,
            "epsilon": 36.23
        },
        "K": {
            "sigma": 3.4,
            "epsilon": 17.61
        },
        "Cl": {
            "sigma": 3.51932,
            "epsilon": 142.562
        },
        "Br": {
            "sigma": 3.51905,
            "epsilon": 186.191
        },
        "H": {
            "sigma": 2.84642,
            "epsilon": 7.64893
        },
        "Zn": {
            "sigma": 2.46155,
            "epsilon": 62.3992
        },
        "Be": {
            "sigma": 2.44552,
            "epsilon": 42.7736
        },
        "Ca": {
            "sigma": 3.02816,
            "epsilon": 119.766
        },
        "Cr": {
            "sigma": 2.69319,
            "epsilon": 7.54829
        },
        "Fe": {
            "sigma": 2.5943,
            "epsilon": 6.54185
        },
        "Mn": {
            "sigma": 2.63795,
            "epsilon": 6.54185
        },
        "Cu": {
            "sigma": 3.11369,
            "epsilon": 2.5161
        },
        "Co": {
            "sigma": 2.55866,
            "epsilon": 7.04507
        },
        "Ga": {
            "sigma": 3.90481,
            "epsilon": 208.836
        },
        "Ti": {
            "sigma": 2.8286,
            "epsilon": 8.55473
        },
        "Sc": {
            "sigma": 2.93551,
            "epsilon": 9.56117
        },
        "V": {
            "sigma": 2.80099,
            "epsilon": 8.05151
        },
        "Ni": {
            "sigma": 2.52481,
            "epsilon": 7.54829
        },
        "Zr": {
            "sigma": 2.78317,
            "epsilon": 34.7221
        },
        "Mg": {
            "sigma": 2.69141,
            "epsilon": 55.8574
        },
        "Ne": {
            "sigma": 2.88918,
            "epsilon": 21.1352
        },
        "Ag": {
            "sigma": 2.80455,
            "epsilon": 18.1159
        },
        "In": {
            "sigma": 3.97608,
            "epsilon": 301.428
        },
        "Cd": {
            "sigma": 2.53728,
            "epsilon": 114.734
        },
        "Sb": {
            "sigma": 3.93777,
            "epsilon": 225.946
        },
        "Te": {
            "sigma": 3.98232,
            "epsilon": 200.281
        },
        "Al": {
            "sigma": 3.91105,
            "epsilon": 155.998
        },
        "Si": {
            "sigma": 3.80414,
            "epsilon": 155.998
        },
        "As": {
            "sigma": 3.77,
            "epsilon": 155.47
        },
        "La": {
            "sigma": 3.14,
            "epsilon": 8.55
        },
        "Ar": {
            "sigma": 3.34,
            "epsilon": 119.8
        },
        "Au": {
            "sigma": 2.93,
            "epsilon": 19.62
        },
        "Rh": {
            "sigma": 2.61,
            "epsilon": 26.67
        },
        "Li": {
            "sigma": 2.18,
            "epsilon": 12.58
        },
        "Ba": {
            "sigma": 3.3,
            "epsilon": 183.15
        },
        "Sr": {
            "sigma": 3.24,
            "epsilon": 118.24
        },
        "Pd": {
            "sigma": 2.58,
            "epsilon": 24.15
        },
        "Mo": {
            "sigma": 2.72,
            "epsilon": 28.18
        },
        "Na": {
            "sigma": 2.66,
            "epsilon": 15.09
        },
        "Kr": {
            "sigma": 3.636,
            "epsilon": 166.4
        },
        "Xe": {
            "sigma": 4.1,
            "epsilon": 221.0
        },
        "Gd": {
            "sigma": 3.0,
            "epsilon": 4.53
        },
        "Er": {
            "sigma": 3.02,
            "epsilon": 3.52
        },
        "Dy": {
            "sigma": 3.05,
            "epsilon": 3.52
        },
        "U": {
            "sigma": 3.02,
            "epsilon": 11.07
        },
        "Tm": {
            "sigma": 3.01,
            "epsilon": 3.02
        },
        "Lu": {
            "sigma": 3.24,
            "epsilon": 20.63
        },
        "Th": {
            "sigma": 3.03,
            "epsilon": 13.08
        },
        "He": {
            "sigma": 2.64,
            "epsilon": 10.9
        },
        "Pt": {
            "sigma": 2.45,
            "epsilon": 40.25
        },
        "X": {
            "sigma": 3.73,
            "epsilon": 148.0
        },
        "Cs": {
            "sigma": 2.80,
            "epsilon": 27.0
        },
        "Os": {
            "sigma": 3.05,
            "epsilon": 79.0
        },
        "At": {
            "sigma": 89.633,
            "epsilon": 3.097
        },
        "Fr": {
            "sigma": 0.0,
            "epsilon": 0.0
        },
        "Pa": {
            "sigma": 0.0,
            "epsilon": 0.0
        }
    }


The ``ase.calculators.mixing.SumCalculator`` will be used to combine the Lennard-Jones and Ewald calculators, and the resulting calculator will be used to run the GCMC simulation.

.. code:: python

    import json

    import ase
    from ase.calculators import mixing
    from ase.data import vdw_radii
    from numba import get_num_threads, set_num_threads

    from flames.adsorbate import Adsorbate
    from flames.calculators.ewald import CustomEwald
    from flames.calculators.lennard_jones import CustomLennardJones
    from flames.utilities import read_cif
    from flames.widom import Widom

    NUM_THREADS_TO_USE = 25
    set_num_threads(NUM_THREADS_TO_USE)

    print(get_num_threads())


    with open("/home/felipe/PRs/flames/flames/data/UFF_lj_params.json", "r") as f:
        uff_lj_params = json.loads(f.read())

    with open("/home/felipe/PRs/flames/flames/data/TraPPE_lj_params.json", "r") as f:
        trappe_lj_params = json.loads(f.read())

    FrameworkPath = "MgMOF-74_DDEC.cif"
    AdsorbatePath = "co2_labels.xyz"

    ewald = CustomEwald(cutoff=12.0, precision=1e-6)
    lj = CustomLennardJones({**uff_lj_params, **trappe_lj_params}, vdw_cutoff=12.5)

    calc = mixing.SumCalculator([lj, ewald])

    # Load the framework structure
    framework = read_cif(FrameworkPath)  # type: ignore

    # Load the adsorbate structure
    adsorbate = Adsorbate(
        name="CO2",
        structure="co2_labels.xyz",
    )

    Temperature = 298.0

    NSteps = 30000

    widom = Widom(
        model=calc,  # type: ignore
        framework_atoms=framework,
        adsorbate_atoms=adsorbate,
        temperature=Temperature,
        device="cpu",
        vdw_radii=vdw_radii,
        debug=False,
        output_to_file=True,
        random_seed=42,
        cutoff_radius=12.5,
        automatic_supercell=True,
    )

    widom.logger.print_header()

    widom.run(NSteps)
    widom.logger.print_summary()
    widom.save_results()

----


The rest of the simulation proceeds as in the previous examples.

References
~~~~~~~~~~

.. [1] Mason, J.A., Sumida, K., Herm, Z.R., Krishna, R. and Long, J.R., 2011. Evaluating metal-organic frameworks for post-combustion carbon dioxide capture via temperature swing adsorption. Energy & Environmental Science, 4(8), pp.3030-3040. https://pubs.rsc.org/en/content/articlelanding/2011/ee/c1ee01720a
