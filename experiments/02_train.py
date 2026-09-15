"""Train ID-Thermo from the ternary first-saturation data only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from idthermo.constants import IONS, THETA_NAMES
from idthermo.datasets import load_ternary_batch
from idthermo.training import (
    deterministic_factor_starts,
    independent_continuation,
    linear_independent_initialization,
    local_shared_jacobian,
    optimize_shared_model,
)


def write_json(path: Path, value) -> None:
    """Write a reproducibility record with stable UTF-8 JSON formatting."""
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def main() -> None:
    """Run the auxiliary fit, shared-factor multistart, and SVD refinement."""
    parser = argparse.ArgumentParser(
        description="Train ID-Thermo on the 4211 ternary first-saturation records."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "training_reproduction",
    )
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise RuntimeError(f"Output directory must be empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    batch = load_ternary_batch()
    protocol = {
        "training_records": len(batch),
        "loss": "ordinary least squares; 0.5 * sum((predicted-observed)^2)",
        "parameter_regularization": "none",
        "quaternary_data_used_for_training": False,
        "deployed_parameterization": "4 delta-theta corrections + 5 shared ion factors",
        "delta_psi": "F_i * F_j * F_k",
        "selection_rule": "minimum converged 4211-record half-SSE",
    }
    write_json(args.out_dir / "protocol.json", protocol)

    # Stage A: solve an auxiliary model in which the nine delta-psi values are
    # independent. This supplies a stable sign and magnitude pattern for the
    # factorized model; it is not the deployed parameterization.
    dtheta0, dpsi0, _, linear_info = linear_independent_initialization(batch)
    write_json(
        args.out_dir / "01_linear_independent_initialization.json",
        {"dtheta": dtheta0.tolist(), "dpsi": dpsi0.tolist(), **linear_info},
    )

    independent, continuation_trace = independent_continuation(batch, dtheta0, dpsi0)
    dtheta_stage = independent.dtheta_values.detach().cpu().numpy()
    dpsi_stage = independent.dpsi_values.detach().cpu().numpy()
    write_json(
        args.out_dir / "02_independent_continuation.json",
        {
            "dtheta": dtheta_stage.tolist(),
            "dpsi": dpsi_stage.tolist(),
            "trace": continuation_trace,
        },
    )

    # Stage B: convert the auxiliary solution to deterministic starts for the
    # 4 + 5 shared-factor model and select by the ternary training objective.
    starts, start_meta = deterministic_factor_starts(dtheta_stage, dpsi_stage)
    write_json(
        args.out_dir / "03_deterministic_shared_starts.json",
        {**start_meta, "candidates": starts},
    )

    coarse = []
    for start in starts:
        model, convergence, trace = optimize_shared_model(batch, start)
        coarse.append({
            "candidate": start,
            "convergence": convergence,
            "parameters": model.export(),
            "trace": trace,
        })
    coarse.sort(
        key=lambda item: (
            item["convergence"]["best_half_sse"],
            item["candidate"]["name"],
        )
    )
    write_json(
        args.out_dir / "04_shared_multistart.json",
        {"selected": coarse[0]["candidate"]["name"], "candidates": coarse},
    )

    # The weakest singular direction supplies four deterministic local starts
    # around the best shared solution. Quaternary data are never read here.
    winner = coarse[0]
    winner_dtheta = [
        winner["parameters"]["dtheta"][name]
        for name in THETA_NAMES
    ]
    winner_factors = [
        winner["parameters"]["ion_factors"][ion]
        for ion in IONS
    ]
    jacobian = local_shared_jacobian(batch, winner_dtheta, winner_factors)
    _, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
    sigma_parameter = winner["convergence"]["rmse"] / singular[-1]
    write_json(
        args.out_dir / "05_local_jacobian.json",
        {
            "singular_values": singular.tolist(),
            "sigma_parameter": float(sigma_parameter),
            "shape": list(jacobian.shape),
        },
    )

    x = np.asarray(winner_dtheta + winner_factors, dtype=float)
    soft_direction = vt[-1]
    refinement_starts = []
    for scale in (-2, -1, 1, 2):
        candidate = x + scale * sigma_parameter * soft_direction
        refinement_starts.append({
            "name": f"soft_direction_{scale:+d}sigma",
            "dtheta": candidate[:4].tolist(),
            "ion_factors": candidate[4:].tolist(),
        })
    write_json(
        args.out_dir / "06_soft_direction_starts.json",
        {"candidates": refinement_starts},
    )

    refined = []
    for start in refinement_starts:
        model, convergence, trace = optimize_shared_model(batch, start)
        refined.append({
            "candidate": start,
            "convergence": convergence,
            "parameters": model.export(),
            "trace": trace,
        })
    refined.sort(
        key=lambda item: (
            item["convergence"]["best_half_sse"],
            item["candidate"]["name"],
        )
    )
    write_json(
        args.out_dir / "07_soft_direction_multistart.json",
        {"selected": refined[0]["candidate"]["name"], "candidates": refined},
    )

    all_candidates = coarse + refined
    all_candidates.sort(
        key=lambda item: (
            item["convergence"]["best_half_sse"],
            item["candidate"]["name"],
        )
    )
    best = all_candidates[0]
    final = {
        "model": "ID-Thermo",
        "parameterization": (
            "four constant delta-theta corrections plus five shared ion factors; "
            "delta-psi(i,j,k)=F_i*F_j*F_k"
        ),
        **best["parameters"],
        "training_records": len(batch),
        "training_half_sse": best["convergence"]["best_half_sse"],
        "training_rmse_wt_pct": best["convergence"]["rmse"],
    }
    output = args.out_dir / "idthermo_parameters.json"
    write_json(output, final)
    print(f"Wrote {output}")
    print("SHA256", hashlib.sha256(output.read_bytes()).hexdigest())
    print(
        json.dumps(
            {
                "training_half_sse": final["training_half_sse"],
                "training_rmse_wt_pct": final["training_rmse_wt_pct"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
