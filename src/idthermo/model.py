"""Differentiable Pitzer first-saturation solver and mixing-layer models."""

from __future__ import annotations

from typing import Iterable
import math

import torch
from torch import nn

from .constants import (
    ALL_PHASES, ANION_INDEX, CATION_INDEX, CHARGES, IONS,
    PSI_INDEX, PSI_NAMES, THETA_INDEX, THETA_NAMES,
)
from .datasets import ExperimentBatch
from .thermodynamics import (
    DTYPE, LN10, MW_W, ThermodynamicDatabase, a_phi, electrostatic_theta, g_function, g_prime_function,
)


torch.set_default_dtype(DTYPE)
_DEFAULT_DATABASE = ThermodynamicDatabase.load()


class PitzerSLEModel(nn.Module):
    """Common Pitzer SLE solver with a replaceable theta/psi correction layer."""

    def __init__(self):
        super().__init__()
        self.register_buffer("charges", torch.tensor(CHARGES, dtype=DTYPE), persistent=False)
        self.register_buffer("phase_stoichiometry", _DEFAULT_DATABASE.phase_stoichiometry.clone(), persistent=False)
        self.register_buffer("phase_water", _DEFAULT_DATABASE.phase_water.clone(), persistent=False)

    def delta_theta(self) -> torch.Tensor:
        """Return the four additive theta corrections."""
        raise NotImplementedError

    def delta_psi(self) -> torch.Tensor:
        """Return the nine additive psi corrections."""
        raise NotImplementedError

    def _activities(self, molality: torch.Tensor, batch: ExperimentBatch):
        """Evaluate ion activity coefficients and water activity."""
        binary, theta_reference, psi_reference = batch.parameter_tables
        theta = theta_reference + self.delta_theta()[None, :]
        psi = psi_reference + self.delta_psi()[None, :]

        ionic_strength = 0.5 * torch.sum(molality * (self.charges[None, :] ** 2), dim=1)
        sqrt_ionic = torch.sqrt(torch.clamp(ionic_strength, min=1e-14))
        aphi = a_phi(batch.temperature_k)
        charge_sum = torch.sum(molality * torch.abs(self.charges)[None, :], dim=1)
        molality_sum = molality.sum(dim=1)

        # Binary cation-anion contributions.
        b, bp, bphi, c_parameter = {}, {}, {}, {}
        for ci, cation in enumerate(CATION_INDEX):
            for ai, anion in enumerate(ANION_INDEX):
                p = binary[:, ci, ai, :]
                beta2 = p[:, 2]
                alpha1 = 1.4 if abs(float(self.charges[cation])) == 2 and abs(float(self.charges[anion])) == 2 else 2.0
                alpha2 = 12.0
                b[(cation, anion)] = p[:, 0] + p[:, 1] * g_function(alpha1 * sqrt_ionic) + beta2 * g_function(alpha2 * sqrt_ionic)
                bp[(cation, anion)] = (
                    p[:, 1] * g_prime_function(alpha1 * sqrt_ionic) + beta2 * g_prime_function(alpha2 * sqrt_ionic)
                ) / torch.clamp(ionic_strength, min=1e-14)
                bphi[(cation, anion)] = p[:, 0] + p[:, 1] * torch.exp(-alpha1 * sqrt_ionic) + beta2 * torch.exp(-alpha2 * sqrt_ionic)
                c_parameter[(cation, anion)] = p[:, 3] / (2 * math.sqrt(abs(float(self.charges[cation] * self.charges[anion]))))

        # Same-charge mixing combines the adjustable theta value with the
        # higher-order electrostatic term E-theta(I).
        pair_terms = {}
        for pair_index, (i, j) in enumerate(THETA_INDEX):
            electrostatic, electrostatic_prime = electrostatic_theta(float(self.charges[i]), float(self.charges[j]), ionic_strength, aphi)
            phi = theta[:, pair_index] + electrostatic
            phi_prime = electrostatic_prime
            phi_osmotic = theta[:, pair_index] + electrostatic + ionic_strength * electrostatic_prime
            pair_terms[(i, j)] = pair_terms[(j, i)] = (phi, phi_prime, phi_osmotic)

        # Common Pitzer term F, shared by all ion activity coefficients.
        common = -aphi * (sqrt_ionic / (1 + 1.2 * sqrt_ionic) + (2 / 1.2) * torch.log(1 + 1.2 * sqrt_ionic))
        for cation in CATION_INDEX:
            for anion in ANION_INDEX:
                common = common + molality[:, cation] * molality[:, anion] * bp[(cation, anion)]
        for i, j in THETA_INDEX:
            common = common + molality[:, i] * molality[:, j] * pair_terms[(i, j)][1]

        psi_map = {frozenset(ids): psi[:, index] for index, ids in enumerate(PSI_INDEX)}

        def psi_value(i: int, j: int, k: int):
            return psi_map[frozenset((i, j, k))]

        # Cation and anion activity coefficients are assembled separately to
        # preserve the standard Pitzer summation structure.
        log_gamma = []
        for ion in CATION_INDEX:
            value = float(self.charges[ion])**2 * common
            for anion in ANION_INDEX:
                value = value + molality[:, anion] * (2 * b[(ion, anion)] + charge_sum * c_parameter[(ion, anion)])
            for cation in CATION_INDEX:
                if cation == ion:
                    continue
                value = value + molality[:, cation] * 2 * pair_terms[(ion, cation)][0]
                for anion in ANION_INDEX:
                    value = value + molality[:, cation] * molality[:, anion] * psi_value(ion, cation, anion)
            value = value + molality[:, 3] * molality[:, 4] * psi_value(3, 4, ion)
            for cation in CATION_INDEX:
                for anion in ANION_INDEX:
                    value = value + abs(float(self.charges[ion])) * molality[:, cation] * molality[:, anion] * c_parameter[(cation, anion)]
            log_gamma.append(value)

        for ion in ANION_INDEX:
            value = float(self.charges[ion])**2 * common
            for cation in CATION_INDEX:
                value = value + molality[:, cation] * (2 * b[(cation, ion)] + charge_sum * c_parameter[(cation, ion)])
            for anion in ANION_INDEX:
                if anion == ion:
                    continue
                value = value + molality[:, anion] * 2 * pair_terms[(ion, anion)][0]
                for cation in CATION_INDEX:
                    value = value + molality[:, anion] * molality[:, cation] * psi_value(ion, anion, cation)
            for c1, c2 in ((0, 1), (0, 2), (1, 2)):
                value = value + molality[:, c1] * molality[:, c2] * psi_value(c1, c2, ion)
            for cation in CATION_INDEX:
                for anion in ANION_INDEX:
                    value = value + abs(float(self.charges[ion])) * molality[:, cation] * molality[:, anion] * c_parameter[(cation, anion)]
            log_gamma.append(value)
        log_gamma = torch.stack(log_gamma, dim=1)

        # Osmotic coefficient and water activity use the same parameter table.
        osmotic_accumulator = -aphi * ionic_strength**1.5 / (1 + 1.2 * sqrt_ionic)
        for cation in CATION_INDEX:
            for anion in ANION_INDEX:
                osmotic_accumulator = osmotic_accumulator + molality[:, cation] * molality[:, anion] * (
                    bphi[(cation, anion)] + charge_sum * c_parameter[(cation, anion)]
                )
        for i, j in ((0, 1), (0, 2), (1, 2)):
            third_order = molality[:, 3] * psi_value(i, j, 3) + molality[:, 4] * psi_value(i, j, 4)
            osmotic_accumulator = osmotic_accumulator + molality[:, i] * molality[:, j] * (pair_terms[(i, j)][2] + third_order)
        third_order = sum(molality[:, c] * psi_value(3, 4, c) for c in CATION_INDEX)
        osmotic_accumulator = osmotic_accumulator + molality[:, 3] * molality[:, 4] * (pair_terms[(3, 4)][2] + third_order)
        osmotic_coefficient = 1 + 2 * osmotic_accumulator / torch.clamp(molality_sum, min=1e-30)
        log_water_activity = -osmotic_coefficient * MW_W * molality_sum
        return log_gamma, log_water_activity

    def saturation_indices(self, salinity: torch.Tensor, batch: ExperimentBatch) -> torch.Tensor:
        """Return saturation indices for all candidate solids at each salinity."""
        molality = batch.ray_coefficients * (salinity / (100 - salinity))[:, None]
        log_gamma, log_water_activity = self._activities(molality, batch)
        log_activity = torch.log(torch.clamp(molality, min=1e-300)) + log_gamma
        stoich = self.phase_stoichiometry
        water = self.phase_water
        values = (
            (log_activity[:, None, :] * stoich[None, :, :]).sum(dim=2) / LN10
            + log_water_activity[:, None] * water[None, :] / LN10
            - batch.solid_logk
        )
        required = stoich[None, :, :] > 0
        present = molality[:, None, :] > 1e-14
        valid = torch.all((~required) | present, dim=2) & batch.phase_mask
        return torch.where(valid, values, torch.full_like(values, -1e30))

    def max_saturation_index(self, salinity: torch.Tensor, batch: ExperimentBatch) -> torch.Tensor:
        """Return the saturation envelope max(SI) along each composition ray."""
        return torch.max(self.saturation_indices(salinity, batch), dim=1).values

    def controlling_phases(self, salinity: torch.Tensor, batch: ExperimentBatch) -> list[str]:
        """Return the phase with the largest saturation index at each endpoint."""
        index = torch.argmax(self.saturation_indices(salinity, batch), dim=1).detach().cpu().numpy()
        return [ALL_PHASES[int(i)] for i in index]

    def find_first_root_brackets(self, batch: ExperimentBatch):
        """Bracket the first zero crossing of the saturation envelope."""
        with torch.no_grad():
            # A fixed salinity grid is used only to locate the first crossing;
            # the differentiable root refinement is carried out in forward().
            grid = torch.cat([
                torch.linspace(0.02, 40.0, 30, dtype=DTYPE),
                torch.linspace(41.0, 96.0, 24, dtype=DTYPE),
            ])
            n = len(batch)
            previous_s = torch.full((n,), float(grid[0]), dtype=DTYPE, device=batch.ray_coefficients.device)
            previous_f = self.max_saturation_index(previous_s, batch)
            lower = torch.full((n,), float("nan"), dtype=DTYPE, device=batch.ray_coefficients.device)
            upper = torch.full((n,), float("nan"), dtype=DTYPE, device=batch.ray_coefficients.device)
            found = torch.zeros(n, dtype=torch.bool, device=batch.ray_coefficients.device)
            for grid_value in grid[1:]:
                salinity = torch.full((n,), float(grid_value), dtype=DTYPE, device=batch.ray_coefficients.device)
                current_f = self.max_saturation_index(salinity, batch)
                crossed = (~found) & torch.isfinite(current_f) & (current_f > 0) & torch.isfinite(previous_f) & (previous_f <= 0)
                lower = torch.where(crossed, previous_s, lower)
                upper = torch.where(crossed, salinity, upper)
                found = found | crossed
                update = (~found) & torch.isfinite(current_f)
                previous_s = torch.where(update, salinity, previous_s)
                previous_f = torch.where(update, current_f, previous_f)
            lower = torch.where(found, lower, torch.full_like(lower, 0.02))
            upper = torch.where(found, upper, torch.full_like(upper, 96.0))
        return lower, upper, found

    def forward(self, batch: ExperimentBatch, brackets=None, root_steps: int = 8):
        """Solve first-saturation salinity with differentiable false-position steps."""
        if brackets is None:
            brackets = self.find_first_root_brackets(batch)
        lower, upper, found = brackets
        lower, upper = lower.clone(), upper.clone()
        f_lower = self.max_saturation_index(lower, batch)
        f_upper = self.max_saturation_index(upper, batch)
        # False-position updates keep the endpoint differentiable with respect
        # to the mixing corrections while the bracket decisions stay detached.
        for _ in range(root_steps):
            denominator = f_upper - f_lower
            denominator = torch.where(torch.abs(denominator) > 1e-12, denominator, torch.full_like(denominator, 1e-12))
            salinity = (lower * f_upper - upper * f_lower) / denominator
            salinity = torch.clamp(salinity, lower + 1e-8, upper - 1e-8)
            f_salinity = self.max_saturation_index(salinity, batch)
            left = f_salinity.detach() <= 0
            lower = torch.where(left, salinity, lower)
            f_lower = torch.where(left, f_salinity, f_lower)
            upper = torch.where(~left, salinity, upper)
            f_upper = torch.where(~left, f_salinity, f_upper)
        denominator = f_upper - f_lower
        denominator = torch.where(torch.abs(denominator) > 1e-12, denominator, torch.full_like(denominator, 1e-12))
        salinity = (lower * f_upper - upper * f_lower) / denominator
        return salinity, found, brackets


class IDThermoModel(PitzerSLEModel):
    """Four delta-theta corrections and five shared ion factors."""

    def __init__(self, dtheta: Iterable[float], ion_factors: Iterable[float]):
        super().__init__()
        self.dtheta_values = nn.Parameter(torch.tensor(list(dtheta), dtype=DTYPE))
        self.ion_factors = nn.Parameter(torch.tensor(list(ion_factors), dtype=DTYPE))

    @classmethod
    def from_parameter_dict(cls, parameters: dict) -> "IDThermoModel":
        """Construct the model from model/idthermo_parameters.json content."""
        return cls(
            [parameters["dtheta"][name] for name in THETA_NAMES],
            [parameters["ion_factors"][ion] for ion in IONS],
        )

    def delta_theta(self) -> torch.Tensor:
        return self.dtheta_values

    def delta_psi(self) -> torch.Tensor:
        """Generate all nine psi corrections from five shared ion factors."""
        return torch.stack([
            self.ion_factors[i] * self.ion_factors[j] * self.ion_factors[k]
            for i, j, k in PSI_INDEX
        ])

    def export(self) -> dict:
        """Return the deployed corrections as JSON-serializable mappings."""
        dtheta = self.delta_theta().detach().cpu().numpy()
        factors = self.ion_factors.detach().cpu().numpy()
        dpsi = self.delta_psi().detach().cpu().numpy()
        return {
            "dtheta": {name: float(value) for name, value in zip(THETA_NAMES, dtheta)},
            "ion_factors": {ion: float(value) for ion, value in zip(IONS, factors)},
            "dpsi": {name: float(value) for name, value in zip(PSI_NAMES, dpsi)},
        }


class IndependentMixingModel(PitzerSLEModel):
    """Auxiliary 13-parameter model with independent delta-psi corrections."""

    def __init__(self, dtheta: Iterable[float], dpsi: Iterable[float]):
        super().__init__()
        self.dtheta_values = nn.Parameter(torch.tensor(list(dtheta), dtype=DTYPE))
        self.dpsi_values = nn.Parameter(torch.tensor(list(dpsi), dtype=DTYPE))

    def delta_theta(self) -> torch.Tensor:
        return self.dtheta_values

    def delta_psi(self) -> torch.Tensor:
        return self.dpsi_values


class FixedMixingModel(PitzerSLEModel):
    """Non-trainable direct delta-theta and delta-psi corrections."""

    def __init__(self, dtheta: Iterable[float] | None = None, dpsi: Iterable[float] | None = None):
        super().__init__()
        theta_values = [0.0] * 4 if dtheta is None else list(dtheta)
        psi_values = [0.0] * 9 if dpsi is None else list(dpsi)
        self.register_buffer("dtheta_values", torch.tensor(theta_values, dtype=DTYPE))
        self.register_buffer("dpsi_values", torch.tensor(psi_values, dtype=DTYPE))

    def delta_theta(self) -> torch.Tensor:
        return self.dtheta_values

    def delta_psi(self) -> torch.Tensor:
        return self.dpsi_values
