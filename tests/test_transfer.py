from idthermo.evaluation import evaluate_all_configurations


def test_headline_idthermo_results():
    table = evaluate_all_configurations()
    expected = {
        "Na-K-Cl-SO4": (0.868349, 1.531990),
        "Na-Mg-Cl-SO4": (0.728966, 1.608341),
        "K-Mg-Cl-SO4": (0.810708, 1.373553),
    }
    subset = table[table.configuration == "ID-Thermo"].set_index("system")
    for system, (r2, rmse) in expected.items():
        assert abs(float(subset.loc[system, "r2"]) - r2) < 2e-6
        assert abs(float(subset.loc[system, "rmse"]) - rmse) < 2e-6
