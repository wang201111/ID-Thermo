"""Thermodynamic parameter loading and temperature-dependent Pitzer terms."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import pandas as pd
import torch

from .constants import ALL_PHASES, IONS, ION_INDEX, PSI_INDEX, THETA_INDEX

DTYPE = torch.float64
TR = 298.15
MW_W = 0.0180153
LN10 = math.log(10.0)
PARAMETER_NAMES = ("B0", "B1", "B2", "C0")
CSV_ENCODING = "utf-8-sig"


def project_root() -> Path:
    """Return the repository root for a source checkout."""
    return Path(__file__).resolve().parents[2]


def parse_phase(reaction: str) -> dict[str, float]:
    """Parse ion and water stoichiometry from a stored dissolution reaction."""
    rhs = reaction.split("=", 1)[1].strip()
    tokens = re.findall(
        r"(\d+(?:\.\d+)?)?\s*(H2O|Na\+|K\+|Mg\+2|Cl-|SO4-2)",
        rhs,
    )
    out: dict[str, float] = {}
    for coefficient, species in tokens:
        out[species] = out.get(species, 0.0) + (
            float(coefficient) if coefficient else 1.0
        )
    return out


def temperature_function(
    coefficients: torch.Tensor,
    temperature_k: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the six-coefficient temperature expansion at T in kelvin."""
    basis = torch.stack(
        [
            torch.ones_like(temperature_k),
            1 / temperature_k - 1 / TR,
            torch.log(temperature_k / TR),
            temperature_k - TR,
            temperature_k * temperature_k - TR * TR,
            1 / (temperature_k * temperature_k) - 1 / (TR * TR),
        ],
        dim=-1,
    )
    return torch.einsum("nf,...f->n...", basis, coefficients)


def a_phi(temperature_k: torch.Tensor) -> torch.Tensor:
    """Return the Debye-Huckel A_phi parameter used by the Pitzer model."""
    temperature_c = temperature_k - 273.15
    density = (
        999.83952
        + 16.945176 * temperature_c
        - 7.9870401e-3 * temperature_c**2
        - 46.170461e-6 * temperature_c**3
        + 105.56302e-9 * temperature_c**4
        - 280.54253e-12 * temperature_c**5
    ) / (1 + 16.897850e-3 * temperature_c)
    dielectric = (
        87.740
        - 0.40008 * temperature_c
        + 9.398e-4 * temperature_c**2
        - 1.410e-6 * temperature_c**3
    )
    avogadro = 6.02214076e23
    elementary_charge = 1.602176634e-19
    epsilon_0 = 8.8541878128e-12
    boltzmann = 1.380649e-23
    return (1 / 3) * torch.sqrt(2 * math.pi * avogadro * density) * (
        elementary_charge**2
        / (4 * math.pi * epsilon_0 * dielectric * boltzmann * temperature_k)
    ) ** 1.5


def g_function(x: torch.Tensor) -> torch.Tensor:
    """Pitzer g(x) function for binary interaction terms."""
    x = torch.clamp(x, min=1e-12)
    return 2 * (1 - (1 + x) * torch.exp(-x)) / (x * x)


def g_prime_function(x: torch.Tensor) -> torch.Tensor:
    """Ionic-strength derivative form paired with g(x)."""
    x = torch.clamp(x, min=1e-12)
    return -2 * (1 - (1 + x + 0.5 * x * x) * torch.exp(-x)) / (x * x)


def j_function(x: torch.Tensor) -> torch.Tensor:
    """Approximation used for the higher-order electrostatic mixing term."""
    x = torch.clamp(x, min=1e-12)
    return x / (4 + 4.581 * x ** (-0.7237) * torch.exp(-0.0120 * x**0.528))


def j_prime_function(x: torch.Tensor) -> torch.Tensor:
    """Derivative of :func:`j_function`."""
    x = torch.clamp(x, min=1e-12)
    c1, p, c2, q = 4.581, 0.7237, 0.0120, 0.528
    denominator = 4 + c1 * x ** (-p) * torch.exp(-c2 * x**q)
    derivative = c1 * torch.exp(-c2 * x**q) * (
        -p * x ** (-p - 1) - c2 * q * x ** (q - p - 1)
    )
    return (denominator - x * derivative) / (denominator * denominator)


def electrostatic_theta(
    zi: float,
    zj: float,
    ionic_strength: torch.Tensor,
    aphi: torch.Tensor,
):
    """Return E-theta and its ionic-strength derivative for an ion pair."""
    if zi == zj:
        zero = torch.zeros_like(ionic_strength)
        return zero, zero

    ionic_strength = torch.clamp(ionic_strength, min=1e-14)
    sqrt_i = torch.sqrt(ionic_strength)
    xij = 6 * zi * zj * aphi * sqrt_i
    xii = 6 * zi * zi * aphi * sqrt_i
    xjj = 6 * zj * zj * aphi * sqrt_i
    value = zi * zj / (4 * ionic_strength) * (
        j_function(xij) - 0.5 * j_function(xii) - 0.5 * j_function(xjj)
    )
    derivative = -value / ionic_strength + zi * zj / (8 * ionic_strength**2) * (
        xij * j_prime_function(xij)
        - 0.5 * xii * j_prime_function(xii)
        - 0.5 * xjj * j_prime_function(xjj)
    )
    return value, derivative


@dataclass(frozen=True)
class ThermodynamicDatabase:
    """Numerical thermodynamic tables consumed by the SLE solver."""

    binary_coefficients: torch.Tensor
    theta_coefficients: torch.Tensor
    psi_coefficients: torch.Tensor
    phase_stoichiometry: torch.Tensor
    phase_water: torch.Tensor
    phase_logk_reference: torch.Tensor
    phase_families: tuple[frozenset[str], ...]
    logk_fits: dict[str, dict]

    @classmethod
    def load(cls, data_dir: Path | None = None) -> "ThermodynamicDatabase":
        """Load the coefficient tables shipped with the repository."""
        data_dir = data_dir or (project_root() / "data" / "thermodynamic")
        binary = pd.read_csv(data_dir / "binary_parameters.csv", encoding=CSV_ENCODING)
        theta = pd.read_csv(data_dir / "thereda_theta.csv", encoding=CSV_ENCODING)
        psi = pd.read_csv(data_dir / "thereda_psi.csv", encoding=CSV_ENCODING)
        phases = pd.read_csv(data_dir / "solid_phases.csv", encoding=CSV_ENCODING)
        logk_fits = json.loads(
            (data_dir / "solid_phase_logk_fits.json").read_text(encoding="utf-8")
        )

        # Binary arrays are indexed by cation, anion, parameter type, and the
        # six temperature coefficients used by temperature_function().
        bin_coef = torch.zeros((3, 2, 4, 6), dtype=DTYPE)
        for ci, cation in enumerate(IONS[:3]):
            for ai, anion in enumerate(IONS[3:]):
                rows = binary[(binary.cation == cation) & (binary.anion == anion)]
                for pi, name in enumerate(PARAMETER_NAMES):
                    row = rows[rows.param == name]
                    if len(row):
                        values = row.iloc[0][[f"a{k}" for k in range(1, 7)]].astype(float)
                        bin_coef[ci, ai, pi] = torch.tensor(
                            values.to_numpy(),
                            dtype=DTYPE,
                        )

        theta_coef = torch.zeros((4, 6), dtype=DTYPE)
        for index, (i, j) in enumerate(THETA_INDEX):
            wanted = frozenset((IONS[i], IONS[j]))
            for _, row in theta.iterrows():
                if frozenset((row.ion_i, row.ion_j)) == wanted:
                    theta_coef[index] = torch.tensor(
                        [float(row[f"a{k}"]) for k in range(1, 7)],
                        dtype=DTYPE,
                    )
                    break

        psi_coef = torch.zeros((9, 6), dtype=DTYPE)
        for index, ids in enumerate(PSI_INDEX):
            wanted = frozenset(IONS[i] for i in ids)
            for _, row in psi.iterrows():
                if frozenset((row.ion_i, row.ion_j, row.ion_k)) == wanted:
                    psi_coef[index] = torch.tensor(
                        [float(row[f"a{k}"]) for k in range(1, 7)],
                        dtype=DTYPE,
                    )
                    break

        phase_map = phases.set_index("name")
        stoich = torch.zeros((len(ALL_PHASES), len(IONS)), dtype=DTYPE)
        water = torch.zeros(len(ALL_PHASES), dtype=DTYPE)
        logk_ref = torch.zeros(len(ALL_PHASES), dtype=DTYPE)
        families: list[frozenset[str]] = []

        for phase_index, phase in enumerate(ALL_PHASES):
            row = phase_map.loc[phase]
            parsed = parse_phase(str(row.reaction))
            families.append(
                frozenset(species for species in parsed if species != "H2O")
            )
            for species, coefficient in parsed.items():
                if species == "H2O":
                    water[phase_index] = coefficient
                else:
                    stoich[phase_index, ION_INDEX[species]] = coefficient
            logk_ref[phase_index] = float(row.log_k)

        return cls(
            bin_coef,
            theta_coef,
            psi_coef,
            stoich,
            water,
            logk_ref,
            tuple(families),
            logk_fits,
        )

    def parameter_tables(self, temperature_k: torch.Tensor):
        """Evaluate binary, theta, and psi reference tables at each T."""
        return (
            temperature_function(self.binary_coefficients, temperature_k),
            temperature_function(self.theta_coefficients, temperature_k),
            temperature_function(self.psi_coefficients, temperature_k),
        )

    def solid_logk(self, temperature_k: torch.Tensor) -> torch.Tensor:
        """Evaluate the solid-phase log K functions for all candidate phases."""
        values = []
        for phase_index, phase in enumerate(ALL_PHASES):
            fit = self.logk_fits.get(phase)
            if fit:
                beta = fit["beta"]
                value = beta[0] + beta[1] / temperature_k
                if int(fit["k"]) == 3:
                    value = value + beta[2] * torch.log(temperature_k)
            else:
                value = torch.full_like(
                    temperature_k,
                    float(self.phase_logk_reference[phase_index]),
                )
            values.append(value)
        return torch.stack(values, dim=1)
