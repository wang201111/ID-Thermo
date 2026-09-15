"""Ion, interaction, and phase definitions used by ID-Thermo."""

from __future__ import annotations

# Ion order is fixed throughout the solver and all stored parameter arrays.
IONS = ("Na+", "K+", "Mg+2", "Cl-", "SO4-2")
ION_INDEX = {ion: i for i, ion in enumerate(IONS)}
CHARGES = (1.0, 1.0, 2.0, -1.0, -2.0)
CATION_INDEX = (0, 1, 2)
ANION_INDEX = (3, 4)

# Same-charge pair corrections. The order matches model/idthermo_parameters.json.
THETA_NAMES = ("Na-K", "Na-Mg", "K-Mg", "Cl-SO4")
THETA_INDEX = ((0, 1), (0, 2), (1, 2), (3, 4))

# Three-ion interaction corrections generated from the five shared ion factors.
PSI_NAMES = (
    "Na-K-Cl", "Na-K-SO4", "Na-Mg-Cl", "Na-Mg-SO4",
    "K-Mg-Cl", "K-Mg-SO4", "Na-Cl-SO4", "K-Cl-SO4", "Mg-Cl-SO4",
)
PSI_INDEX = (
    (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4),
    (1, 2, 3), (1, 2, 4), (0, 3, 4), (1, 3, 4), (2, 3, 4),
)

# Each ternary subsystem carries one theta-psi block in the local
# identifiability analysis. The phase lists define the SLE candidate set.
TERNARY_SYSTEMS = {
    "KCl_NaCl_H2O": {
        "theta": "Na-K", "psi": "Na-K-Cl", "counter_ion": "Cl-",
        "phases": ("Halite", "Sylvite"),
    },
    "K2SO4_Na2SO4_H2O": {
        "theta": "Na-K", "psi": "Na-K-SO4", "counter_ion": "SO4-2",
        "phases": ("Arcanite", "Thenardite", "Mirabilite", "Glaserite"),
    },
    "MgCl2_NaCl_H2O": {
        "theta": "Na-Mg", "psi": "Na-Mg-Cl", "counter_ion": "Cl-",
        "phases": ("Halite", "Bischofite"),
    },
    "MgSO4_Na2SO4_H2O": {
        "theta": "Na-Mg", "psi": "Na-Mg-SO4", "counter_ion": "SO4-2",
        "phases": ("Thenardite", "Mirabilite", "Epsomite", "Hexahydrite", "Kieserite", "Bloedite"),
    },
    "KCl_MgCl2_H2O": {
        "theta": "K-Mg", "psi": "K-Mg-Cl", "counter_ion": "Cl-",
        "phases": ("Sylvite", "Bischofite", "Carnallite"),
    },
    "K2SO4_MgSO4_H2O": {
        "theta": "K-Mg", "psi": "K-Mg-SO4", "counter_ion": "SO4-2",
        "phases": ("Arcanite", "Epsomite", "Hexahydrite", "Kieserite", "Leonite", "Picromerite", "Langbeinite"),
    },
    "Na2SO4_NaCl_H2O": {
        "theta": "Cl-SO4", "psi": "Na-Cl-SO4", "counter_ion": "Na+",
        "phases": ("Halite", "Thenardite", "Mirabilite"),
    },
    "K2SO4_KCl_H2O": {
        "theta": "Cl-SO4", "psi": "K-Cl-SO4", "counter_ion": "K+",
        "phases": ("Sylvite", "Arcanite"),
    },
    "MgCl2_MgSO4_H2O": {
        "theta": "Cl-SO4", "psi": "Mg-Cl-SO4", "counter_ion": "Mg+2",
        "phases": ("Bischofite", "Epsomite", "Hexahydrite", "Kieserite"),
    },
}

QUATERNARY_SYSTEMS = {
    "Na-K-Cl-SO4": {
        "phases": ("Halite", "Sylvite", "Arcanite", "Thenardite", "Mirabilite", "Glaserite"),
        "theta": ("Na-K", "Cl-SO4"),
        "psi": ("Na-K-Cl", "Na-K-SO4", "Na-Cl-SO4", "K-Cl-SO4"),
    },
    "Na-Mg-Cl-SO4": {
        "phases": ("Halite", "Thenardite", "Mirabilite", "Bischofite", "Epsomite", "Hexahydrite", "Kieserite", "Bloedite"),
        "theta": ("Na-Mg", "Cl-SO4"),
        "psi": ("Na-Mg-Cl", "Na-Mg-SO4", "Na-Cl-SO4", "Mg-Cl-SO4"),
    },
    "K-Mg-Cl-SO4": {
        "phases": ("Sylvite", "Arcanite", "Carnallite", "Bischofite", "Epsomite", "Hexahydrite", "Kieserite", "Leonite", "Picromerite", "Kainite", "Langbeinite"),
        "theta": ("K-Mg", "Cl-SO4"),
        "psi": ("K-Mg-Cl", "K-Mg-SO4", "K-Cl-SO4", "Mg-Cl-SO4"),
    },
}

ALL_PHASES = tuple(sorted(
    {phase for spec in TERNARY_SYSTEMS.values() for phase in spec["phases"]}
    | {phase for spec in QUATERNARY_SYSTEMS.values() for phase in spec["phases"]}
))
