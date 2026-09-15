"""Feature construction and regression baselines used in the paper comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .constants import ALL_PHASES, IONS, QUATERNARY_SYSTEMS, TERNARY_SYSTEMS
from .datasets import load_quaternary_frame, load_ternary_frame
from .thermodynamics import ThermodynamicDatabase, a_phi, temperature_function

CATIONS = IONS[:3]
ANIONS = IONS[3:]


def ray_fractions(molality: dict[str, float]) -> tuple[list[float], list[float]]:
    """Return normalized cation and anion fractions along a composition ray."""
    cation_sum = sum(molality.get(cation, 0.0) for cation in CATIONS)
    anion_sum = sum(molality.get(anion, 0.0) for anion in ANIONS)
    cation = [
        molality.get(ion, 0.0) / cation_sum if cation_sum > 0 else 0.0
        for ion in CATIONS
    ]
    anion = [
        molality.get(ion, 0.0) / anion_sum if anion_sum > 0 else 0.0
        for ion in ANIONS
    ]
    return cation, anion


def _scalar_temperature_function(coefficients, temperature_k: float) -> float:
    """Evaluate one coefficient vector at one temperature."""
    import torch

    t = torch.tensor([temperature_k], dtype=torch.float64)
    c = torch.tensor(coefficients, dtype=torch.float64)
    return float(temperature_function(c, t)[0])


def thermodynamic_features(
    molality: dict[str, float],
    temperature_k: float,
    phases: tuple[str, ...],
    database: ThermodynamicDatabase,
) -> list[float]:
    """Build the ten database-derived features used in feature group T2."""
    cation_fraction, anion_fraction = ray_fractions(molality)
    binary = database.binary_coefficients.cpu().numpy()
    theta = database.theta_coefficients.cpu().numpy()
    psi = database.psi_coefficients.cpu().numpy()
    features: list[float] = []

    # Composition-weighted B0, B1, and C0 binary terms.
    for parameter_index in (0, 1, 3):
        value = 0.0
        for ci in range(3):
            for ai in range(2):
                value += (
                    cation_fraction[ci]
                    * anion_fraction[ai]
                    * _scalar_temperature_function(
                        binary[ci, ai, parameter_index],
                        temperature_k,
                    )
                )
        features.append(value)

    theta_cations = 0.0
    for pair_index, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        theta_cations += (
            cation_fraction[i]
            * cation_fraction[j]
            * _scalar_temperature_function(theta[pair_index], temperature_k)
        )
    features.append(theta_cations)
    features.append(
        anion_fraction[0]
        * anion_fraction[1]
        * _scalar_temperature_function(theta[3], temperature_k)
    )

    psi_value = 0.0
    psi_indices = (
        (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4),
        (1, 2, 3), (1, 2, 4), (0, 3, 4), (1, 3, 4), (2, 3, 4),
    )
    for index, ids in enumerate(psi_indices):
        if ids[0] < 3 and ids[1] < 3:
            ci, cj, ai = ids[0], ids[1], ids[2] - 3
            weight = (
                cation_fraction[ci]
                * cation_fraction[cj]
                * anion_fraction[ai]
            )
        else:
            ci = ids[0]
            weight = (
                cation_fraction[ci]
                * anion_fraction[0]
                * anion_fraction[1]
            )
        psi_value += weight * _scalar_temperature_function(
            psi[index],
            temperature_k,
        )
    features.append(psi_value)

    import torch

    features.append(
        float(a_phi(torch.tensor([temperature_k], dtype=torch.float64))[0])
    )

    # The solid-phase database enters only through summary values. Solid labels
    # from experimental records are not used as model inputs.
    logk_values = []
    for phase in phases:
        phase_index = ALL_PHASES.index(phase)
        fit = database.logk_fits.get(phase)
        if fit:
            value = fit["beta"][0] + fit["beta"][1] / temperature_k
            if int(fit["k"]) == 3:
                value += fit["beta"][2] * np.log(temperature_k)
        else:
            value = float(database.phase_logk_reference[phase_index])
        stoich = database.phase_stoichiometry[phase_index].cpu().numpy()
        nu = max(float(stoich.sum()), 1.0)
        logk_values.append(value / nu)

    features.extend([
        min(logk_values),
        max(logk_values),
        float(np.mean(logk_values)),
    ])
    return features


def structural_features(molality: dict[str, float]) -> list[float]:
    """Build permutation-invariant pair and triplet products for feature group TG."""
    cation_fraction, anion_fraction = ray_fractions(molality)
    features = []
    for i in range(3):
        for j in range(i + 1, 3):
            features.append(cation_fraction[i] * cation_fraction[j])
    features.append(anion_fraction[0] * anion_fraction[1])
    for i in range(3):
        for j in range(i + 1, 3):
            for k in range(2):
                features.append(
                    cation_fraction[i] * cation_fraction[j] * anion_fraction[k]
                )
    for i in range(3):
        features.append(
            cation_fraction[i] * anion_fraction[0] * anion_fraction[1]
        )
    return features


def build_features(
    molality: dict[str, float],
    temperature_c: float,
    phases: tuple[str, ...],
    database: ThermodynamicDatabase,
):
    """Return the T1, T2, and TG feature vectors for one record."""
    cation_fraction, anion_fraction = ray_fractions(molality)
    temperature_k = temperature_c + 273.15
    composition_temperature = (
        cation_fraction
        + anion_fraction
        + [temperature_c, 1000.0 / temperature_k]
    )
    thermodynamic = composition_temperature + thermodynamic_features(
        molality,
        temperature_k,
        phases,
        database,
    )
    structured = thermodynamic + structural_features(molality)
    return composition_temperature, thermodynamic, structured


def _row_molality(row) -> dict[str, float]:
    columns = ("m_Na", "m_K", "m_Mg", "m_Cl", "m_SO4")
    return {
        ion: float(row[column])
        for ion, column in zip(IONS, columns)
        if float(row[column]) > 0
    }


def featurize_training(database: ThermodynamicDatabase | None = None):
    """Build the three feature matrices for all 4211 ternary records."""
    database = database or ThermodynamicDatabase.load()
    frame = load_ternary_frame()
    matrices = {"T1": [], "T2": [], "TG": []}

    for _, row in frame.iterrows():
        features = build_features(
            _row_molality(row),
            float(row.temperature_C),
            TERNARY_SYSTEMS[row.subsystem]["phases"],
            database,
        )
        for key, values in zip(("T1", "T2", "TG"), features):
            matrices[key].append(values)

    x = {key: np.asarray(value, dtype=float) for key, value in matrices.items()}
    y = frame.salinity_wt_pct.astype(float).to_numpy()
    groups = frame.subsystem.astype(str).to_numpy()
    return x, y, groups


def featurize_quaternary(
    system: str,
    database: ThermodynamicDatabase | None = None,
):
    """Build the three feature matrices for one held-out quaternary system."""
    database = database or ThermodynamicDatabase.load()
    frame = load_quaternary_frame(system)
    matrices = {"T1": [], "T2": [], "TG": []}

    for _, row in frame.iterrows():
        features = build_features(
            _row_molality(row),
            float(row.temperature_C),
            QUATERNARY_SYSTEMS[system]["phases"],
            database,
        )
        for key, values in zip(("T1", "T2", "TG"), features):
            matrices[key].append(values)

    x = {key: np.asarray(value, dtype=float) for key, value in matrices.items()}
    y = frame.salinity_wt_pct.astype(float).to_numpy()
    return x, y


def candidate_models(seed: int = 0):
    """Return the fixed hyperparameter candidates screened by LOSO validation."""
    return {
        "Ridge": [
            make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            for alpha in (0.1, 1.0, 10.0)
        ],
        "RandomForest": [
            RandomForestRegressor(
                n_estimators=250,
                max_depth=None,
                min_samples_leaf=leaf,
                random_state=seed,
                n_jobs=-1,
            )
            for leaf in (1, 4)
        ],
        "GradientBoosting": [
            GradientBoostingRegressor(
                n_estimators=250,
                max_depth=3,
                learning_rate=rate,
                random_state=seed,
            )
            for rate in (0.05, 0.1)
        ],
        "GaussianProcess": [
            make_pipeline(
                StandardScaler(),
                GaussianProcessRegressor(
                    kernel=ConstantKernel(10.0) * RBF(length_scale=2.0)
                    + WhiteKernel(1.0),
                    normalize_y=True,
                    n_restarts_optimizer=0,
                    random_state=seed,
                ),
            )
        ],
        "MLP": [
            make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    alpha=alpha,
                    max_iter=1000,
                    random_state=seed,
                    early_stopping=True,
                    n_iter_no_change=30,
                ),
            )
            for alpha in (1e-3, 1e-1)
        ],
    }


def loso_rmse(
    estimator,
    x,
    y,
    subsystem,
    max_train_records: int | None = None,
) -> float:
    """Mean leave-one-subsystem-out RMSE used for hyperparameter selection."""
    errors = []
    for held_out in np.unique(subsystem):
        train_mask = subsystem != held_out
        test_mask = subsystem == held_out
        x_train, y_train = x[train_mask], y[train_mask]

        if max_train_records is not None and len(y_train) > max_train_records:
            rng = np.random.default_rng(0)
            chosen = rng.choice(len(y_train), max_train_records, replace=False)
            x_train, y_train = x_train[chosen], y_train[chosen]

        prediction = estimator.fit(x_train, y_train).predict(x[test_mask])
        errors.append(np.sqrt(np.mean((prediction - y[test_mask]) ** 2)))

    return float(np.mean(errors))


def baseline_metrics(prediction, observation) -> dict:
    """Metrics reported for the machine-learning comparison."""
    prediction = np.asarray(prediction, dtype=float)
    observation = np.asarray(observation, dtype=float)
    error = prediction - observation
    denominator = np.sum((observation - observation.mean()) ** 2)
    return {
        "r2": float(1 - np.sum(error**2) / denominator),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "med_abs": float(np.median(np.abs(error))),
        "n_gt5": int(np.sum(np.abs(error) > 5)),
    }
