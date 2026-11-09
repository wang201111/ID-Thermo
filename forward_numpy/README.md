# ID-Thermo NumPy/SciPy forward calculator

This standalone calculator implements the submitted nine-parameter model's corrected Pitzer mixing table and first-saturation SLE calculation. It requires **NumPy and SciPy only**. It neither trains nor recalibrates the liquid or solid parameters. All parameter files are relative to this folder; PyTorch, pandas, an installation of the main ID-Thermo package and a network connection are unnecessary at runtime.

From the repository root, install the dependencies in a Python 3.10 or newer environment and run the example:

```sh
cd forward_numpy
python -m pip install -r requirements.txt
python forward.py --input example.csv --output example_predictions.json
python forward.py --input example.csv --output rounded_predictions.json --rounded
python verify.py
python verify_first_root.py
```

The two example commands refuse to overwrite an existing output. `verify.py` deliberately blocks all imports of Torch and pandas, evaluates all 182 frozen numerical cases, and compares concentration and controlling phase with the strict float64 reference. It writes `verification_result.json` and `verification_predictions.json`. The reference values are frozen float64 calculations. In the 182-case check, all controlling phases matched and the maximum absolute concentration difference was 2.33e-10 wt%.

Each output JSON record includes the predicted salinity in wt%, the controlling solid phase, whether a root was found, and the input and root-search diagnostics.

CSV columns are `record_id,system,temperature_C,q_Na,q_K,q_Mg,q_Cl,q_SO4`. The ionic ray is

\[
m_i(S)=q_i S/(100-S),
\]

where (S) is total dissolved anhydrous salt mass percent and (m_i) is molality in mol/kg water. The five `q` entries are mol/kg dry dissolved salts. Use consistent anhydrous salt masses, **not hydrated crystal masses**, in the helper:

```python
from forward import ray_from_salt_weights, predict_ray
q = ray_from_salt_weights({'NaCl': 1.0, 'K2SO4': 1.0})
result = predict_ray(q, 25.0, 'Na-K-Cl-SO4')
```

An arbitrary unsaturated feed of known molalities and total salt mass percent may also define `q_i = m_i * (100-S_feed)/S_feed`; the feed need not be at saturation. Do not insert a guessed saturation concentration to define a new composition.

Supported system identifiers are `Na-K-Cl-SO4`, `Na-Mg-Cl-SO4`, and `K-Mg-Cl-SO4`; all are aqueous systems. The calculator rejects incompatible ions, negative inputs, grossly unbalanced ionic rays and temperatures outside 0–100 °C. This computational range is not a validated industrial operating envelope. The reported quaternary evidence spans 0–100 °C for Na–K, 0–55 °C for Na–Mg, and only 15 °C for K–Mg; it does not establish accuracy at every composition or pressure within these intervals. The output flags temperature extrapolation from those observed ranges.

`parameters.json` contains the baseline binary/theta/psi temperature expansions, frozen G13 solid fits, candidate phases, and full-precision corrections. The optional `--parameters path.json` accepts an independently supplied file of the same schema. `--rounded` applies the four-decimal corrections printed in Table 4 directly to theta and psi. It does not round ion factors and then multiply them. In the 182-case check, rounding shifts predictions by at most 0.028896 wt%; full-precision values should be used to reproduce the manuscript numerically.

The original SG-fam support rule is retained: an out-of-window hydrate is removed if its salt family has an in-window representative; otherwise it is retained as an extrapolated family representative. The default search resolves each candidate phase on a 0.1 wt% salinity grid, refines local SI maxima to recover narrow positive intervals, and refines crossings with Brent's method before taking the earliest root. The archived solver supplies only a search upper bound. `--root-mode legacy` explicitly selects the archived 54-point scan. Failure to find a crossing is returned explicitly. Neither finite search mathematically certifies that arbitrarily narrow or tangential crossings are absent; changes to the model or range require renewed root validation. Missing solids, solid solutions and unmodeled reactions are not supplied by the liquid correction.

`verify_first_root.py` exercises three nearby parameter points from a real training-only stress case. An archived Torch bisection implementation missed an earlier hydrate crossing in two of them; its results are retained in the fixture. The archived NumPy Brent implementation already finds the earlier root in these particular cases. This distinction matters: the fixture diagnoses a numerical search failure, not a failure of the final submitted model. The diagnostic thirteen-parameter point is not the default parameter table.

Some archived evaluation compositions have nonzero charge residuals (up to about 0.724% relative to the sum of ion molalities). Their supplied ionic rays also do not all agree exactly with the reported total salt mass: using the helper's molar masses, the mass associated with 1 kg nominal dry salts ranges from 943.65 to 1055.86 g, and 14 cases deviate by more than 1%. The verification retains those inputs to reproduce the archived calculation; it does not repair them. New inputs are checked for electroneutrality and mass normalization (relative mass tolerance 0.001). Reproduction of all archived cases through the CLI requires **both** explicit options `--allow-unbalanced --allow-mass-imbalance`, and every affected output remains flagged with its residual. The historical `salinity_wt_pct` output on an inconsistent ray is the archived salinity coordinate, not a newly certified mass-consistent composition. Use these options only for traceable reproduction, not to hide inconsistent new inputs.

`PROVENANCE.json` records the original core and parameter origins. Within `pitzer_core.py`, the only modification is the standard-library JSON loader replacing pandas CSV loading. Its thermodynamic equations are preserved; the file still contains unused historical gradient helper functions. `first_root.py` adds the enhanced numerical root search described above. The included verification fixtures preserve the archived evaluation inputs and reference predictions so that the numerical comparison is reproducible.
