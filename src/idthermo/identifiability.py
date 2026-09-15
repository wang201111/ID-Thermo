"""Local and global identifiability calculations for the mixing layer."""

from __future__ import annotations

import numpy as np
import torch

from .constants import IONS, PSI_NAMES, TERNARY_SYSTEMS, THETA_NAMES
from .datasets import ExperimentBatch, load_ternary_batch, load_ternary_frame
from .evaluation import load_parameter_file
from .model import FixedMixingModel, IDThermoModel


def final_direct_corrections(parameters: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen direct delta-theta and derived delta-psi arrays."""
    parameters = parameters or load_parameter_file()
    dtheta = np.array([parameters["dtheta"][name] for name in THETA_NAMES], dtype=float)
    dpsi = np.array([parameters["dpsi"][name] for name in PSI_NAMES], dtype=float)
    return dtheta, dpsi


def subsystem_sensitivity(
    system: str,
    batch: ExperimentBatch | None = None,
    parameters: dict | None = None,
    step: float = 1e-5,
    root_steps: int = 12,
):
    """Compute the two-column theta-psi sensitivity matrix for one subsystem."""
    full_batch = batch or load_ternary_batch()
    frame = load_ternary_frame()
    selected = np.flatnonzero(frame.subsystem.to_numpy() == system)
    subset = full_batch.subset(selected)

    dtheta, dpsi = final_direct_corrections(parameters)
    theta_index = THETA_NAMES.index(TERNARY_SYSTEMS[system]["theta"])
    psi_index = PSI_NAMES.index(TERNARY_SYSTEMS[system]["psi"])
    base = FixedMixingModel(dtheta, dpsi)
    brackets = base.find_first_root_brackets(subset)

    columns = []
    for kind, index in (("theta", theta_index), ("psi", psi_index)):
        plus_theta, minus_theta = dtheta.copy(), dtheta.copy()
        plus_psi, minus_psi = dpsi.copy(), dpsi.copy()
        if kind == "theta":
            plus_theta[index] += step
            minus_theta[index] -= step
        else:
            plus_psi[index] += step
            minus_psi[index] -= step

        plus = FixedMixingModel(plus_theta, plus_psi)
        minus = FixedMixingModel(minus_theta, minus_psi)
        with torch.no_grad():
            y_plus, found_plus, _ = plus(subset, brackets=brackets, root_steps=root_steps)
            y_minus, found_minus, _ = minus(subset, brackets=brackets, root_steps=root_steps)
        if int(found_plus.sum()) != len(subset) or int(found_minus.sum()) != len(subset):
            raise RuntimeError(f"Root loss in sensitivity calculation for {system}")
        columns.append(((y_plus - y_minus) / (2 * step)).cpu().numpy())

    jacobian = np.column_stack(columns)
    norms = np.linalg.norm(jacobian, axis=0)
    normalized = jacobian / norms
    singular = np.linalg.svd(normalized, compute_uv=False)
    cosine = float(abs(np.dot(jacobian[:, 0], jacobian[:, 1]) / (norms[0] * norms[1])))

    return {
        "subsystem": system,
        "n": int(len(subset)),
        "cosine_similarity": cosine,
        "normalized_condition_number": float(singular[0] / singular[-1]),
        "normalized_singular_values": singular.tolist(),
        "theta": TERNARY_SYSTEMS[system]["theta"],
        "psi": TERNARY_SYSTEMS[system]["psi"],
    }, jacobian


def global_parameter_jacobian(
    batch: ExperimentBatch | None = None,
    parameters: dict | None = None,
    step: float = 1e-5,
    root_steps: int = 16,
):
    """Compute the 4211-by-9 Jacobian of the deployed ID-Thermo model."""
    batch = batch or load_ternary_batch()
    parameters = parameters or load_parameter_file()
    x = np.array(
        [parameters["dtheta"][name] for name in THETA_NAMES]
        + [parameters["ion_factors"][ion] for ion in IONS],
        dtype=float,
    )

    base = IDThermoModel(x[:4], x[4:])
    brackets = base.find_first_root_brackets(batch)
    jacobian = np.empty((len(batch), 9), dtype=float)

    # Central differences are evaluated with one common root bracket so that
    # the derivative reflects the local parameter perturbation, not a change in
    # the bracketing path.
    for column in range(9):
        plus_x, minus_x = x.copy(), x.copy()
        plus_x[column] += step
        minus_x[column] -= step
        plus = IDThermoModel(plus_x[:4], plus_x[4:])
        minus = IDThermoModel(minus_x[:4], minus_x[4:])
        with torch.no_grad():
            y_plus, found_plus, _ = plus(batch, brackets=brackets, root_steps=root_steps)
            y_minus, found_minus, _ = minus(batch, brackets=brackets, root_steps=root_steps)
        if int(found_plus.sum()) != len(batch) or int(found_minus.sum()) != len(batch):
            raise RuntimeError("Root loss in global Jacobian calculation")
        jacobian[:, column] = ((y_plus - y_minus) / (2 * step)).cpu().numpy()

    singular = np.linalg.svd(jacobian, compute_uv=False)
    normalized = jacobian / np.linalg.norm(jacobian, axis=0, keepdims=True)
    normalized_singular = np.linalg.svd(normalized, compute_uv=False)

    return {
        "shape": list(jacobian.shape),
        "rank": int(np.linalg.matrix_rank(jacobian)),
        "singular_values": singular.tolist(),
        "smallest_singular_value": float(singular[-1]),
        "condition_number": float(singular[0] / singular[-1]),
        "column_normalized_condition_number": float(
            normalized_singular[0] / normalized_singular[-1]
        ),
    }, jacobian
