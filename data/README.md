# Data

The public data directory contains the inputs used by the reproduction scripts.
All CSV files are encoded as UTF-8 with a byte-order mark for direct opening in
Excel on Windows.

## Ternary training data

`ternary/first_saturation.csv` contains the 4211 first-saturation records used
for ID-Thermo training and identifiability analysis.

Main fields:

| Field | Meaning |
| --- | --- |
| `record_id` | Stable row identifier |
| `subsystem` | Ternary subsystem key |
| `temperature_C` | Experimental temperature in degC |
| `salinity_wt_pct` | Total salt mass fraction at first saturation, wt% |
| `water_wt_pct` | Water mass fraction, wt% |
| `m_Na` ... `m_SO4` | Ion molalities at the reported first-saturation point |
| `ionic_strength_mol_kg` | Ionic strength at that point |
| `observed_solid_phase` | Reported coexisting solid phase or phases |
| `source_compilation` | Handbook key listed in `handbooks.csv` |
| `handbook_page_start` | First handbook page for the compiled record when available |
| `source_id` | Independent literature-source key when one can be assigned |
| `independent_source_entry` | Whether the row contributes to independent-source accounting |

Bibliographic text is not repeated in every measurement row.
`ternary/source_entries.csv` stores one citation for each `source_id`. The 294
unique source IDs are the independent entries used in the source-count
statistics reported in the manuscript.

`ternary/handbooks.csv` maps the compilation keys to references [28]-[31]:

- `Zdanovskii1953` - Zdanovskii et al. (1953)
- `Linke1965` - W. F. Linke (1965)
- `Pelsh1973` - A. D. Pel'sh (1973)
- `Seidell1919` - A. Seidell (1919)

Older source citations printed in Cyrillic in the handbooks are stored as Latin
transliterations in `source_entries.csv`. No numerical data are changed by this
normalization.

## Quaternary evaluation data

`quaternary/first_saturation.csv` contains the held-out transfer sets. These
records are not used in ID-Thermo training.

| System | Records |
| --- | ---: |
| Na-K-Cl-SO4 | 75 |
| Na-Mg-Cl-SO4 | 82 |
| K-Mg-Cl-SO4 | 25 |

`quaternary/source_entries.csv` lists the nine source entries used for the
source-level comparison in the manuscript, including the Shen (2016), Zhang
(2022), and Cui (2020) evaluation datasets.

## Thermodynamic tables

`thermodynamic/` contains the coefficient tables read by the solver:

- `binary_parameters.csv` - binary Pitzer parameter coefficients
- `thereda_theta.csv` - THEREDA same-charge pair parameters
- `thereda_psi.csv` - THEREDA triplet parameters
- `solid_phases.csv` - solid-phase reactions and reference log K values
- `solid_phase_logk_fits.json` - temperature functions used in the SLE solver

These files are calculation inputs. They are kept separate from the experimental
first-saturation tables so that changes to thermodynamic inputs can be audited
independently from changes to experimental data.
