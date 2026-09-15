"""Deterministic training protocol for the nine-parameter ID-Thermo model."""

from __future__ import annotations

from dataclasses import replace
import itertools
import math

import numpy as np
from scipy.optimize import least_squares
import torch

from .constants import PSI_INDEX
from .datasets import ExperimentBatch
from .model import IDThermoModel, IndependentMixingModel

SCORE_ROOT_STEPS = 18
MAX_OUTER_ROUNDS = 30
MIN_OUTER_ROUNDS = 5
PATIENCE = 4
ABSOLUTE_TOLERANCE = 1e-4


def least_squares_score(
    model,
    batch: ExperimentBatch,
    root_steps: int = SCORE_ROOT_STEPS,
) -> tuple[float, float, int]:
    """Return half-SSE, RMSE, and number of successful first-saturation roots."""
    brackets = model.find_first_root_brackets(batch)
    with torch.no_grad():
        prediction, found, _ = model(batch, brackets=brackets, root_steps=root_steps)
    residual = prediction - batch.target_salinity
    half_sse = 0.5 * torch.sum(residual * residual)
    rmse = torch.sqrt(torch.mean(residual * residual))
    return float(half_sse), float(rmse), int(found.sum())


def linear_independent_initialization(batch: ExperimentBatch):
    """Solve a local linearized 4-theta plus 9-psi initialization problem."""
    base = IDThermoModel([0.0] * 4, [0.0] * 5)
    brackets = base.find_first_root_brackets(batch)
    with torch.no_grad():
        baseline, _, _ = base(batch, brackets=brackets, root_steps=8)

    baseline = baseline.cpu().numpy()
    target = batch.target_salinity.cpu().numpy()
    jacobian = np.empty((len(batch), 13), dtype=float)
    step = 1e-4

    # The auxiliary problem temporarily treats the nine psi corrections as
    # independent variables. This gives signs and relative magnitudes before
    # imposing the shared-factor parameterization.
    for index in range(4):
        correction = [step if i == index else 0.0 for i in range(4)]
        model = IDThermoModel(correction, [0.0] * 5)
        with torch.no_grad():
            prediction, _, _ = model(batch, brackets=brackets, root_steps=8)
        jacobian[:, index] = (prediction.cpu().numpy() - baseline) / step

    for index in range(9):
        psi = batch.psi_reference.clone()
        psi[:, index] += step
        modified = replace(batch, psi_reference=psi)
        with torch.no_grad():
            prediction, _, _ = base(modified, brackets=brackets, root_steps=8)
        jacobian[:, 4 + index] = (prediction.cpu().numpy() - baseline) / step

    result = least_squares(
        lambda x: (baseline - target) + jacobian @ x,
        np.zeros(13),
        loss="linear",
        max_nfev=4000,
    )
    info = {
        "half_sse_linearized": float(result.cost),
        "nfev": int(result.nfev),
    }
    return result.x[:4], result.x[4:], jacobian, info


def independent_continuation(
    batch: ExperimentBatch,
    dtheta,
    dpsi,
    epochs: int = 100,
):
    """Refine the auxiliary 13-parameter model through the full SLE solver."""
    model = IndependentMixingModel(dtheta, dpsi)
    brackets = model.find_first_root_brackets(batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)
    trace = []

    for epoch in range(epochs):
        if epoch == 40:
            optimizer.param_groups[0]["lr"] = 3e-4
        if epoch == 75:
            optimizer.param_groups[0]["lr"] = 1e-4
        if epoch and epoch % 20 == 0:
            brackets = model.find_first_root_brackets(batch)

        optimizer.zero_grad()
        prediction, _, _ = model(batch, brackets=brackets, root_steps=4)
        residual = prediction - batch.target_salinity
        loss = 0.5 * torch.sum(residual * residual)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            trace.append([epoch, *least_squares_score(model, batch, root_steps=12)])

    return model, trace


def deterministic_factor_starts(dtheta, dpsi):
    """Construct shared-factor starts from the converged independent psi values."""
    dpsi = np.asarray(dpsi, dtype=float)

    # The first six psi terms involve cation pairs and one anion. Their log
    # magnitudes form a linear system in the five log factor magnitudes.
    design = np.zeros((6, 5), dtype=float)
    for row, key in enumerate(PSI_INDEX[:6]):
        design[row, list(key)] = 1.0
    log_magnitude = np.linalg.lstsq(
        design,
        np.log(np.maximum(np.abs(dpsi[:6]), 1e-12)),
        rcond=None,
    )[0]

    # This direction leaves the first six triple products unchanged. The three
    # anion-pair psi values select useful anchors along it.
    soft_direction = np.array([1.0, 1.0, 1.0, -2.0, -2.0])

    branches = []
    for signs in itertools.product((-1.0, 1.0), repeat=5):
        signs = np.asarray(signs)
        predicted = np.array([
            np.prod(signs[list(key)])
            for key in PSI_INDEX[:6]
        ])
        if np.all(predicted == np.sign(dpsi[:6])):
            branches.append(signs)
    if len(branches) != 2:
        raise RuntimeError(f"Expected two sign branches, found {len(branches)}")

    reference_magnitude = np.array([
        math.exp(np.sum(log_magnitude[list(key)]))
        for key in PSI_INDEX
    ])
    individual_anchors = [
        -(1 / 3) * math.log(abs(dpsi[j]) / reference_magnitude[j])
        for j in range(6, 9)
        if dpsi[j] != 0
    ]
    scale = max(
        1e-12,
        float(
            np.dot(reference_magnitude[6:], np.abs(dpsi[6:]))
            / np.dot(reference_magnitude[6:], reference_magnitude[6:])
        ),
    )
    aggregate_anchor = -(1 / 3) * math.log(scale)
    anchors = sorted(set(
        round(float(value), 12)
        for value in [
            0.0,
            min(individual_anchors),
            aggregate_anchor,
            max(individual_anchors),
        ]
    ))

    starts = []
    for branch_index, signs in enumerate(branches):
        for anchor_index, anchor in enumerate(anchors):
            factors = signs * np.exp(log_magnitude + anchor * soft_direction)
            starts.append({
                "name": f"branch{branch_index}_anchor{anchor_index}",
                "dtheta": list(map(float, dtheta)),
                "ion_factors": factors.tolist(),
                "anchor": anchor,
            })

    return starts, {
        "canonical_log_magnitude": log_magnitude.tolist(),
        "soft_direction": soft_direction.tolist(),
        "individual_anchors": individual_anchors,
        "aggregate_anchor": aggregate_anchor,
        "anchors": anchors,
    }


def optimize_shared_model(batch: ExperimentBatch, start: dict):
    """Optimize one deterministic shared-factor start with repeated LBFGS rounds."""
    model = IDThermoModel(start["dtheta"], start["ion_factors"])

    def parameters():
        return {
            "dtheta": model.dtheta_values.detach().cpu().numpy().tolist(),
            "ion_factors": model.ion_factors.detach().cpu().numpy().tolist(),
        }

    def restore(values):
        with torch.no_grad():
            model.dtheta_values.copy_(
                torch.tensor(values["dtheta"], dtype=torch.float64)
            )
            model.ion_factors.copy_(
                torch.tensor(values["ion_factors"], dtype=torch.float64)
            )

    best = least_squares_score(model, batch)
    best_parameters = parameters()
    trace = [["start", 0, *best]]
    stalled = 0
    converged = False
    reason = "maximum rounds"

    # Root brackets are refreshed before each outer round. Within one LBFGS
    # round they remain fixed so the line search sees a smooth local objective.
    for round_index in range(MAX_OUTER_ROUNDS):
        restore(best_parameters)
        previous_loss = best[0]
        brackets = model.find_first_root_brackets(batch)
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=0.5,
            max_iter=30,
            max_eval=50,
            history_size=50,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            prediction, _, _ = model(batch, brackets=brackets, root_steps=8)
            residual = prediction - batch.target_salinity
            loss = 0.5 * torch.sum(residual * residual)
            loss.backward()
            return loss

        optimizer.step(closure)
        current = least_squares_score(model, batch)
        trace.append(["lbfgs", round_index, *current])

        if current[0] < best[0]:
            best = current
            best_parameters = parameters()
        else:
            restore(best_parameters)

        improvement = previous_loss - best[0]
        stalled = stalled + 1 if improvement <= ABSOLUTE_TOLERANCE else 0
        if round_index + 1 >= MIN_OUTER_ROUNDS and stalled >= PATIENCE:
            converged = True
            reason = f"plateau after {PATIENCE} consecutive rounds"
            break

    restore(best_parameters)
    return model, {
        "converged": converged,
        "stop_reason": reason,
        "outer_rounds": len(trace) - 1,
        "best_half_sse": best[0],
        "rmse": best[1],
        "roots": best[2],
    }, trace


def local_shared_jacobian(
    batch: ExperimentBatch,
    dtheta,
    ion_factors,
    step: float = 1e-5,
):
    """Compute the local 4211-by-9 Jacobian at a shared-model checkpoint."""
    x = np.asarray(list(dtheta) + list(ion_factors), dtype=float)
    base = IDThermoModel(x[:4], x[4:])
    brackets = base.find_first_root_brackets(batch)
    jacobian = np.empty((len(batch), 9), dtype=float)

    for index in range(9):
        plus, minus = x.copy(), x.copy()
        plus[index] += step
        minus[index] -= step
        model_plus = IDThermoModel(plus[:4], plus[4:])
        model_minus = IDThermoModel(minus[:4], minus[4:])
        with torch.no_grad():
            y_plus, found_plus, _ = model_plus(
                batch,
                brackets=brackets,
                root_steps=12,
            )
            y_minus, found_minus, _ = model_minus(
                batch,
                brackets=brackets,
                root_steps=12,
            )
        if int(found_plus.sum()) != len(batch) or int(found_minus.sum()) != len(batch):
            raise RuntimeError("Root loss while computing the local Jacobian")
        jacobian[:, index] = ((y_plus - y_minus) / (2 * step)).cpu().numpy()

    return jacobian
