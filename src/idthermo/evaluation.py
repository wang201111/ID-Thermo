"""Evaluation utilities for the frozen Pitzer mixing corrections."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import QUATERNARY_SYSTEMS
from .datasets import load_quaternary_batch, load_quaternary_frame
from .model import FixedMixingModel, IDThermoModel
from .thermodynamics import ThermodynamicDatabase, project_root


def load_parameter_file(path: Path | None = None) -> dict:
    """Load the frozen ID-Thermo parameter file."""
    path = path or (project_root() / "model" / "idthermo_parameters.json")
    return json.loads(path.read_text(encoding="utf-8"))


def regression_metrics(prediction, observation) -> dict:
    """Return R2, RMSE, MAE, and signed mean bias."""
    prediction = np.asarray(prediction, dtype=float)
    observation = np.asarray(observation, dtype=float)
    error = prediction - observation
    denominator = np.sum((observation - observation.mean()) ** 2)
    return {
        "n": int(len(observation)),
        "r2": float(1 - np.sum(error**2) / denominator),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def phase_accuracy(
    frame: pd.DataFrame,
    controlling_phases: list[str],
) -> tuple[int, int, float]:
    """Compare predicted controlling phases with available source labels."""
    hits = 0
    labeled = 0
    for observed, predicted in zip(frame.observed_phases, controlling_phases):
        if pd.isna(observed) or not str(observed).strip():
            continue
        accepted = {phase.strip() for phase in str(observed).split(";") if phase.strip()}
        if not accepted:
            continue
        labeled += 1
        hits += int(predicted in accepted)
    return hits, labeled, (hits / labeled if labeled else float("nan"))


def evaluate_model(
    model,
    batch,
    frame: pd.DataFrame,
    root_steps: int = 14,
) -> tuple[dict, np.ndarray, list[str]]:
    """Evaluate one mixing-parameter configuration on a prepared batch."""
    brackets = model.find_first_root_brackets(batch)
    with torch.no_grad():
        prediction, found, _ = model(batch, brackets=brackets, root_steps=root_steps)
        controlling = model.controlling_phases(prediction, batch)

    metrics = regression_metrics(
        prediction.cpu().numpy(),
        frame.salinity_wt_pct.astype(float).to_numpy(),
    )
    hits, labeled, hit_rate = phase_accuracy(frame, controlling)
    metrics.update(
        roots_found=int(found.sum()),
        phase_hits=hits,
        phase_labeled_n=labeled,
        phase_hit=hit_rate,
    )
    return metrics, prediction.cpu().numpy(), controlling


def evaluate_quaternary_system(
    system: str,
    configuration: str,
    database: ThermodynamicDatabase | None = None,
    parameters: dict | None = None,
):
    """Evaluate zero mixing, THEREDA, or ID-Thermo on one held-out system."""
    database = database or ThermodynamicDatabase.load()
    frame = load_quaternary_frame(system)
    batch = load_quaternary_batch(system, database)

    if configuration == "zero":
        model = FixedMixingModel()
        batch = batch.with_thereda_mixing(False)
    elif configuration == "thereda":
        model = FixedMixingModel()
    elif configuration == "idthermo":
        parameters = parameters or load_parameter_file()
        model = IDThermoModel.from_parameter_dict(parameters)
    else:
        raise ValueError(configuration)

    return evaluate_model(model, batch, frame)


def evaluate_all_configurations(parameters: dict | None = None) -> pd.DataFrame:
    """Evaluate the three reference configurations on all quaternary systems."""
    database = ThermodynamicDatabase.load()
    parameters = parameters or load_parameter_file()
    rows = []

    for system in QUATERNARY_SYSTEMS:
        for configuration, label in (
            ("zero", "Zero mixing"),
            ("thereda", "THEREDA"),
            ("idthermo", "ID-Thermo"),
        ):
            metrics, _, _ = evaluate_quaternary_system(
                system,
                configuration,
                database,
                parameters,
            )
            rows.append({"system": system, "configuration": label, **metrics})

    return pd.DataFrame(rows)
