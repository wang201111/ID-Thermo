import json
from pathlib import Path

import numpy as np

from idthermo.constants import IONS, PSI_INDEX, PSI_NAMES


def test_delta_psi_is_factor_product():
    parameters = json.loads((Path("model") / "idthermo_parameters.json").read_text(encoding="utf-8"))
    factors = np.array([parameters["ion_factors"][ion] for ion in IONS], dtype=float)
    for name, (i, j, k) in zip(PSI_NAMES, PSI_INDEX):
        assert abs(parameters["dpsi"][name] - factors[i] * factors[j] * factors[k]) < 1e-14
