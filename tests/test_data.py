from pathlib import Path

import pandas as pd

from idthermo.datasets import CSV_ENCODING, load_quaternary_frame, load_ternary_frame
from idthermo.thermodynamics import project_root


def test_ternary_record_counts():
    frame = load_ternary_frame()
    expected = {
        "MgSO4_Na2SO4_H2O": 735,
        "K2SO4_MgSO4_H2O": 672,
        "Na2SO4_NaCl_H2O": 561,
        "MgCl2_MgSO4_H2O": 523,
        "KCl_NaCl_H2O": 523,
        "KCl_MgCl2_H2O": 358,
        "K2SO4_KCl_H2O": 351,
        "K2SO4_Na2SO4_H2O": 296,
        "MgCl2_NaCl_H2O": 192,
    }
    assert len(frame) == 4211
    assert frame.subsystem.value_counts().to_dict() == expected


def test_ternary_source_accounting():
    frame = load_ternary_frame()
    independent = frame[frame.independent_source_entry.astype(bool)]
    expected = {
        "MgSO4_Na2SO4_H2O": 38,
        "K2SO4_MgSO4_H2O": 53,
        "Na2SO4_NaCl_H2O": 43,
        "MgCl2_MgSO4_H2O": 34,
        "KCl_NaCl_H2O": 40,
        "KCl_MgCl2_H2O": 28,
        "K2SO4_KCl_H2O": 29,
        "K2SO4_Na2SO4_H2O": 25,
        "MgCl2_NaCl_H2O": 26,
    }
    per_subsystem = independent.groupby("subsystem").source_id.nunique().to_dict()
    assert per_subsystem == expected
    assert sum(per_subsystem.values()) == 316
    assert independent.source_id.nunique() == 294
    assert independent.source_id.notna().all()


def test_source_registry_is_normalized():
    root = project_root()
    frame = load_ternary_frame()
    sources = pd.read_csv(
        root / "data" / "ternary" / "source_entries.csv",
        encoding=CSV_ENCODING,
    )
    assert len(sources) == 294
    assert sources.source_id.nunique() == 294
    assert "source_reference" not in frame.columns
    assert "handbook_page_start" in frame.columns
    assert set(frame.source_id.dropna()) == set(sources.source_id)
    assert all(str(value).isascii() for value in sources.citation)


def test_handbook_mapping():
    handbooks = pd.read_csv(
        project_root() / "data" / "ternary" / "handbooks.csv",
        encoding=CSV_ENCODING,
    )
    assert handbooks.source_compilation.tolist() == [
        "Zdanovskii1953",
        "Linke1965",
        "Pelsh1973",
        "Seidell1919",
    ]
    assert handbooks.paper_reference.tolist() == ["[28]", "[29]", "[30]", "[31]"]


def test_quaternary_record_counts_and_sources():
    frame = load_quaternary_frame()
    assert frame.system.value_counts().to_dict() == {
        "Na-Mg-Cl-SO4": 82,
        "Na-K-Cl-SO4": 75,
        "K-Mg-Cl-SO4": 25,
    }
    sources = pd.read_csv(
        project_root() / "data" / "quaternary" / "source_entries.csv",
        encoding=CSV_ENCODING,
    )
    assert len(sources) == 9
    assert int(sources.n_records.sum()) == 182


def test_public_csv_files_have_utf8_bom():
    root = project_root()
    for path in sorted((root / "data").rglob("*.csv")):
        assert path.read_bytes().startswith(b"\xef\xbb\xbf"), path
