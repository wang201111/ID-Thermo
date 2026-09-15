"""Rebuild the machine-learning baseline comparison from the ternary data."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from idthermo.baselines import (
    baseline_metrics,
    candidate_models,
    featurize_quaternary,
    featurize_training,
    loso_rmse,
)
from idthermo.datasets import CSV_ENCODING

FEATURE_GROUPS = {
    "T1": "composition_temperature",
    "T2": "thermodynamic_features",
    "TG": "structure_enhanced",
}
SYSTEMS = ("Na-K-Cl-SO4", "Na-Mg-Cl-SO4", "K-Mg-Cl-SO4")


def main() -> None:
    """Select each learner by ternary LOSO validation and evaluate transfer."""
    parser = argparse.ArgumentParser(
        description="Reproduce the 3 feature groups x 5 learner comparison."
    )
    parser.add_argument(
        "--feature-group",
        choices=["T1", "T2", "TG", "all"],
        default="all",
    )
    parser.add_argument(
        "--model",
        choices=["Ridge", "RandomForest", "GradientBoosting", "GP", "MLP", "all"],
        default="all",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "ml_baselines_reproduced.csv",
    )
    args = parser.parse_args()

    x_train, y_train, subsystems = featurize_training()
    if len(y_train) != 4211:
        raise RuntimeError(f"Training budget changed: {len(y_train)}")

    tests = {system: featurize_quaternary(system) for system in SYSTEMS}
    if [len(tests[system][1]) for system in SYSTEMS] != [75, 82, 25]:
        raise RuntimeError("Quaternary evaluation sizes changed")

    groups = (
        FEATURE_GROUPS
        if args.feature_group == "all"
        else {args.feature_group: FEATURE_GROUPS[args.feature_group]}
    )
    model_groups = candidate_models(seed=0)
    model_groups["GP"] = model_groups.pop("GaussianProcess")
    if args.model != "all":
        model_groups = {args.model: model_groups[args.model]}

    rows = []
    for feature_key, feature_label in groups.items():
        x = x_train[feature_key]
        for model_name, candidates in model_groups.items():
            started = time.time()

            # The GP protocol uses a fixed 1200-record subset because full
            # repeated GP fitting is prohibitively expensive at n=4211.
            limit = 1200 if model_name == "GP" else None
            scores = [
                loso_rmse(
                    candidate,
                    x,
                    y_train,
                    subsystems,
                    max_train_records=limit,
                )
                for candidate in candidates
            ]
            selected_index = int(np.argmin(scores))
            estimator = candidates[selected_index]

            x_fit, y_fit = x, y_train
            if limit is not None and len(y_fit) > limit:
                rng = np.random.default_rng(0)
                chosen = rng.choice(len(y_fit), limit, replace=False)
                x_fit, y_fit = x_fit[chosen], y_fit[chosen]
            estimator.fit(x_fit, y_fit)

            values = []
            for system in SYSTEMS:
                x_test, y_test = tests[system]
                prediction = estimator.predict(x_test[feature_key])
                metrics = baseline_metrics(prediction, y_test)
                rows.append({
                    "tier": feature_label,
                    "feature_key": feature_key,
                    "model": model_name,
                    "system": system,
                    "n": len(y_test),
                    **metrics,
                    "loso_rmse": float(scores[selected_index]),
                    "candidate_index": selected_index,
                    "train_n": len(y_train) if limit is None else min(limit, len(y_train)),
                })
                values.append(f"{system}={metrics['r2']:.3f}")

            elapsed = time.time() - started
            print(
                f"{feature_key:<2s} {model_name:<16s} "
                f"LOSO={scores[selected_index]:.3f} {' '.join(values)} ({elapsed:.0f}s)"
            )

    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Partial reruns replace only the matching feature/model rows.
    if args.out.exists() and (args.feature_group != "all" or args.model != "all"):
        previous = pd.read_csv(args.out, encoding=CSV_ENCODING)
        keys = set(zip(result.feature_key, result.model))
        keep = [
            tuple(pair) not in keys
            for pair in zip(previous.feature_key, previous.model)
        ]
        result = pd.concat([previous.loc[keep], result], ignore_index=True)

    result.to_csv(args.out, index=False, encoding=CSV_ENCODING)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
