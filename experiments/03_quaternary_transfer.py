"""Evaluate the frozen parameter table on the held-out quaternary systems."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from idthermo.constants import QUATERNARY_SYSTEMS
from idthermo.datasets import CSV_ENCODING, load_quaternary_batch, load_quaternary_frame
from idthermo.evaluation import evaluate_all_configurations, evaluate_model, load_parameter_file
from idthermo.model import FixedMixingModel, IDThermoModel
from idthermo.thermodynamics import ThermodynamicDatabase


def main() -> None:
    """Write system-level, pointwise, and source-level transfer results."""
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen ID-Thermo table on three held-out quaternary systems."
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("model") / "idthermo_parameters.json",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    parameters = load_parameter_file(args.parameters)
    table = evaluate_all_configurations(parameters)
    table.to_csv(
        args.out_dir / "quaternary_transfer.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    print(table.to_string(index=False))

    database = ThermodynamicDatabase.load()
    model = IDThermoModel.from_parameter_dict(parameters)
    thereda_model = FixedMixingModel()
    point_rows = []
    source_rows = []

    for system in QUATERNARY_SYSTEMS:
        frame = load_quaternary_frame(system)
        batch = load_quaternary_batch(system, database)
        _, prediction, controlling = evaluate_model(model, batch, frame)
        _, thereda_prediction, _ = evaluate_model(thereda_model, batch, frame)

        for (_, row), predicted, thereda_value, phase in zip(
            frame.iterrows(),
            prediction,
            thereda_prediction,
            controlling,
        ):
            point_rows.append({
                "record_id": row.record_id,
                "system": system,
                "source_id": row.source_id,
                "temperature_C": float(row.temperature_C),
                "observed_salinity_wt_pct": float(row.salinity_wt_pct),
                "thereda_predicted_salinity_wt_pct": float(thereda_value),
                "idthermo_predicted_salinity_wt_pct": float(predicted),
                "idthermo_residual_wt_pct": float(predicted - row.salinity_wt_pct),
                "controlling_phase": phase,
            })

        observed = frame.salinity_wt_pct.astype(float).to_numpy()
        for source_id, group in frame.groupby("source_id", sort=False):
            index = group.index.to_numpy()
            thereda_rmse = float(
                ((thereda_prediction[index] - observed[index]) ** 2).mean() ** 0.5
            )
            idthermo_rmse = float(
                ((prediction[index] - observed[index]) ** 2).mean() ** 0.5
            )
            source_rows.append({
                "system": system,
                "source_id": source_id,
                "n": len(index),
                "temperature_min_C": float(group.temperature_C.min()),
                "temperature_max_C": float(group.temperature_C.max()),
                "thereda_rmse_wt_pct": thereda_rmse,
                "idthermo_rmse_wt_pct": idthermo_rmse,
                "delta_rmse_wt_pct": idthermo_rmse - thereda_rmse,
            })

    pd.DataFrame(point_rows).to_csv(
        args.out_dir / "quaternary_predictions.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    pd.DataFrame(source_rows).to_csv(
        args.out_dir / "source_metrics.csv",
        index=False,
        encoding=CSV_ENCODING,
    )


if __name__ == "__main__":
    main()
