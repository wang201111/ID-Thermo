"""Run post-freeze component diagnostics for the corrected mixing table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from idthermo.constants import PSI_NAMES, QUATERNARY_SYSTEMS, THETA_NAMES
from idthermo.datasets import CSV_ENCODING, load_quaternary_batch, load_quaternary_frame
from idthermo.evaluation import evaluate_model, load_parameter_file
from idthermo.model import FixedMixingModel
from idthermo.thermodynamics import ThermodynamicDatabase


def main() -> None:
    """Evaluate each active correction alone and after removal from the full table."""
    parser = argparse.ArgumentParser(
        description="Post-freeze add-one and leave-one-out diagnostics."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "component_ablation.csv",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    parameters = load_parameter_file()
    dtheta = np.array(
        [parameters["dtheta"][name] for name in THETA_NAMES],
        dtype=float,
    )
    dpsi = np.array(
        [parameters["dpsi"][name] for name in PSI_NAMES],
        dtype=float,
    )
    database = ThermodynamicDatabase.load()
    rows = []

    for system, specification in QUATERNARY_SYSTEMS.items():
        frame = load_quaternary_frame(system)
        batch = load_quaternary_batch(system, database)
        thereda_metrics, _, _ = evaluate_model(FixedMixingModel(), batch, frame)
        full_metrics, _, _ = evaluate_model(
            FixedMixingModel(dtheta, dpsi),
            batch,
            frame,
        )

        active = (
            [("theta", name) for name in specification["theta"]]
            + [("psi", name) for name in specification["psi"]]
        )
        for parameter_type, name in active:
            add_theta = np.zeros(4, dtype=float)
            add_psi = np.zeros(9, dtype=float)
            drop_theta = dtheta.copy()
            drop_psi = dpsi.copy()

            if parameter_type == "theta":
                index = THETA_NAMES.index(name)
                add_theta[index] = dtheta[index]
                drop_theta[index] = 0.0
                value = dtheta[index]
            else:
                index = PSI_NAMES.index(name)
                add_psi[index] = dpsi[index]
                drop_psi[index] = 0.0
                value = dpsi[index]

            add_metrics, _, _ = evaluate_model(
                FixedMixingModel(add_theta, add_psi),
                batch,
                frame,
            )
            drop_metrics, _, _ = evaluate_model(
                FixedMixingModel(drop_theta, drop_psi),
                batch,
                frame,
            )
            rows.append({
                "system": system,
                "parameter_type": parameter_type,
                "parameter": name,
                "correction": value,
                "thereda_r2": thereda_metrics["r2"],
                "full_r2": full_metrics["r2"],
                "add_one_r2": add_metrics["r2"],
                "leave_one_out_r2": drop_metrics["r2"],
                "add_one_delta_r2": add_metrics["r2"] - thereda_metrics["r2"],
                "full_context_delta_r2": full_metrics["r2"] - drop_metrics["r2"],
            })

    result = pd.DataFrame(rows)
    result.to_csv(args.out, index=False, encoding=CSV_ENCODING)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
