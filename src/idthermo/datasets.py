"""Load experimental tables and prepare batched SLE calculations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .constants import ALL_PHASES, QUATERNARY_SYSTEMS, TERNARY_SYSTEMS
from .thermodynamics import DTYPE, ThermodynamicDatabase, project_root

CSV_ENCODING = "utf-8-sig"


@dataclass
class ExperimentBatch:
    """Tensor bundle used by the Pitzer first-saturation solver."""

    ray_coefficients: torch.Tensor
    temperature_k: torch.Tensor
    target_salinity: torch.Tensor
    phase_mask: torch.Tensor
    binary_parameters: torch.Tensor
    theta_reference: torch.Tensor
    psi_reference: torch.Tensor
    solid_logk: torch.Tensor
    labels: list[str]
    record_ids: list[str]

    def __len__(self) -> int:
        return int(self.target_salinity.numel())

    @property
    def parameter_tables(self):
        """Return the temperature-resolved Pitzer parameter tables."""
        return self.binary_parameters, self.theta_reference, self.psi_reference

    def with_thereda_mixing(self, enabled: bool) -> "ExperimentBatch":
        """Return a copy with THEREDA theta/psi terms enabled or zeroed."""
        if enabled:
            return self
        return replace(
            self,
            theta_reference=torch.zeros_like(self.theta_reference),
            psi_reference=torch.zeros_like(self.psi_reference),
        )

    def subset(self, indices: np.ndarray | list[int]) -> "ExperimentBatch":
        """Select a set of records without rebuilding thermodynamic tables."""
        idx_np = np.asarray(indices, dtype=int)
        idx = torch.as_tensor(idx_np, dtype=torch.long)
        ids = np.asarray(self.record_ids, dtype=object)[idx_np].tolist()
        labels = np.asarray(self.labels, dtype=object)[idx_np].tolist()
        return ExperimentBatch(
            ray_coefficients=self.ray_coefficients[idx],
            temperature_k=self.temperature_k[idx],
            target_salinity=self.target_salinity[idx],
            phase_mask=self.phase_mask[idx],
            binary_parameters=self.binary_parameters[idx],
            theta_reference=self.theta_reference[idx],
            psi_reference=self.psi_reference[idx],
            solid_logk=self.solid_logk[idx],
            labels=labels,
            record_ids=ids,
        )


def _phase_mask(
    labels: Iterable[str],
    temperature_c: np.ndarray,
    ternary: bool,
    database: ThermodynamicDatabase,
) -> torch.Tensor:
    """Build the candidate-solid mask with the data-supported phase rule."""
    phase_to_index = {name: i for i, name in enumerate(ALL_PHASES)}
    labels = list(labels)
    mask = np.zeros((len(labels), len(ALL_PHASES)), dtype=bool)

    for row_index, (label, temp) in enumerate(zip(labels, temperature_c)):
        phases = (
            TERNARY_SYSTEMS[label]["phases"]
            if ternary
            else QUATERNARY_SYSTEMS[label]["phases"]
        )
        in_support: list[str] = []
        out_support: list[str] = []
        for phase in phases:
            fit = database.logk_fits.get(phase)
            if fit is None or float(fit["T_min"]) <= temp <= float(fit["T_max"]):
                in_support.append(phase)
            else:
                out_support.append(phase)

        # If one hydrate/polymorph in a salt family has a supported fit, suppress
        # unsupported members of the same family. A family is never dropped in
        # full merely because its temperature fit is missing.
        supported_families = {
            database.phase_families[phase_to_index[phase]]
            for phase in in_support
        }
        dropped = {
            phase
            for phase in out_support
            if database.phase_families[phase_to_index[phase]] in supported_families
        }
        for phase in phases:
            if phase not in dropped:
                mask[row_index, phase_to_index[phase]] = True

    return torch.tensor(mask, dtype=torch.bool)


def _make_batch(
    frame: pd.DataFrame,
    label_column: str,
    ternary: bool,
    database: ThermodynamicDatabase,
) -> ExperimentBatch:
    """Convert a first-saturation table to the tensors used by the solver."""
    labels = frame[label_column].astype(str).tolist()
    salinity = frame.salinity_wt_pct.astype(float).to_numpy()
    molality = frame[["m_Na", "m_K", "m_Mg", "m_Cl", "m_SO4"]].astype(float).to_numpy()

    # Each record defines a composition ray. The stored molalities are the
    # measured first-saturation endpoint; this coefficient lets the solver vary
    # salinity while keeping the dry-salt composition ratio fixed.
    ray = molality * ((100.0 - salinity) / salinity)[:, None]

    temperature_c = frame.temperature_C.astype(float).to_numpy()
    temperature_k = torch.tensor(temperature_c + 273.15, dtype=DTYPE)
    binary, theta, psi = database.parameter_tables(temperature_k)

    return ExperimentBatch(
        ray_coefficients=torch.tensor(ray, dtype=DTYPE),
        temperature_k=temperature_k,
        target_salinity=torch.tensor(salinity, dtype=DTYPE),
        phase_mask=_phase_mask(labels, temperature_c, ternary, database),
        binary_parameters=binary,
        theta_reference=theta,
        psi_reference=psi,
        solid_logk=database.solid_logk(temperature_k),
        labels=labels,
        record_ids=frame.record_id.astype(str).tolist(),
    )


def load_ternary_frame(data_dir: Path | None = None) -> pd.DataFrame:
    """Load and validate the 4211-record ternary training table."""
    data_dir = data_dir or (project_root() / "data" / "ternary")
    frame = pd.read_csv(
        data_dir / "first_saturation.csv",
        encoding=CSV_ENCODING,
        low_memory=False,
    )
    if len(frame) != 4211:
        raise RuntimeError(f"Expected 4211 ternary records, found {len(frame)}")

    order = pd.Categorical(
        frame.subsystem,
        categories=list(TERNARY_SYSTEMS),
        ordered=True,
    )
    return (
        frame.assign(_system_order=order)
        .sort_values(["_system_order", "record_id"])
        .drop(columns="_system_order")
        .reset_index(drop=True)
    )


def load_ternary_batch(
    database: ThermodynamicDatabase | None = None,
    data_dir: Path | None = None,
) -> ExperimentBatch:
    """Load the complete ternary training set as one solver batch."""
    database = database or ThermodynamicDatabase.load()
    return _make_batch(load_ternary_frame(data_dir), "subsystem", True, database)


def load_quaternary_frame(
    system: str | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load the held-out quaternary data, optionally for one system."""
    data_dir = data_dir or (project_root() / "data" / "quaternary")
    frame = pd.read_csv(
        data_dir / "first_saturation.csv",
        encoding=CSV_ENCODING,
        low_memory=False,
    )
    if system is None:
        return frame
    if system not in QUATERNARY_SYSTEMS:
        raise KeyError(system)
    return frame[frame.system == system].reset_index(drop=True)


def load_quaternary_batch(
    system: str,
    database: ThermodynamicDatabase | None = None,
    data_dir: Path | None = None,
) -> ExperimentBatch:
    """Load one held-out quaternary system as a solver batch."""
    database = database or ThermodynamicDatabase.load()
    frame = load_quaternary_frame(system, data_dir)
    return _make_batch(frame, "system", False, database)
