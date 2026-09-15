"""Reproduce the local and global identifiability calculations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from idthermo.constants import TERNARY_SYSTEMS
from idthermo.datasets import CSV_ENCODING, load_ternary_batch
from idthermo.identifiability import global_parameter_jacobian, subsystem_sensitivity


def main() -> None:
    """Evaluate nine local theta-psi blocks and the global 9-parameter Jacobian."""
    parser = argparse.ArgumentParser(
        description="Reproduce the identifiability analysis on the 4211 ternary records."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    batch = load_ternary_batch()
    rows = []
    for subsystem in TERNARY_SYSTEMS:
        summary, _ = subsystem_sensitivity(subsystem, batch=batch)
        rows.append(summary)
        print(
            f"{subsystem:<24s} n={summary['n']:4d} "
            f"cosine={summary['cosine_similarity']:.6f} "
            f"condition={summary['normalized_condition_number']:.6f}"
        )

    pd.DataFrame(rows).to_csv(
        args.out_dir / "identifiability_by_subsystem.csv",
        index=False,
        encoding=CSV_ENCODING,
    )

    global_summary, jacobian = global_parameter_jacobian(batch=batch)
    (args.out_dir / "identifiability_global.json").write_text(
        json.dumps(global_summary, indent=2),
        encoding="utf-8",
    )
    np.save(args.out_dir / "identifiability_global_jacobian.npy", jacobian)
    print(json.dumps(global_summary, indent=2))


if __name__ == "__main__":
    main()
