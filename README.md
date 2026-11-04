# chiral-ml-paper1 — code, curated data and figures

Code, curated data and figures for the preprint

> **Where machine learning for enantioselectivity on polysaccharide chiral stationary phases
> actually fails: a leave-one-paper-out reappraisal of 10,525 enantiomer pairs**
> Lang Cui (崔朗), Kunming University of Science and Technology, Kunming 650500, China.

The preprint and its Supplementary Information are included as `preprint/Paper1_main.pdf` and
`preprint/Paper1_SI.pdf`.

## Contents

| Path | What it is |
|---|---|
| `data/curated/` | curated modelling tables: 10,525 enantiomer pairs, 25,866 records, 25 columns, 8 split indices (`splits/`), descriptor matrices, and every result CSV quoted in the paper |
| `scripts/` | analysis pipeline that produces the paper's numbers and figures |
| `figures/v3/` | the 7 publication figures (PNG + PDF) |
| `preprint/` | main manuscript PDF and Supplementary Information PDF |
| `notes/`, `scratch/` | empty output directories; scripts write evidence notes and intermediates here |

## Environment

Tested with Python 3.14.3 on Windows 11; Python >= 3.11 should work.

```
numpy pandas scikit-learn rdkit matplotlib scipy
```

On Windows set `PYTHONIOENCODING=utf-8` before running any script.

## Quick start

The curated tables are already included. To rebuild them from the upstream CMRT release, download
CMRT (Zenodo 10.5281/zenodo.7623903) into the location expected by `scripts/01_curate_cmrt.py`
and run:

```bash
python scripts/01_curate_cmrt.py          # rebuild data/curated/ from raw CMRT (optional)
python scripts/_qa_01_curate.py           # structural gate: expected 67 checks pass
python scripts/26_paper_number_gate.py    # number gate: expected 95 pass, 0 fail
python scripts/31_run_all_gates.py --fast # all gates in one command
python scripts/38_make_figures_v3.py      # regenerate the 7 figures
```

## Scripts

| Script(s) | What they produce |
|---|---|
| `01_curate_cmrt.py`, `_qa_01_curate.py` | curated tables, split indices, structural gate |
| `06_chirality_descriptors.py`, `06c_chiral_frame_descriptors.py` | L1a-FF and L1-CF chirality descriptors |
| `12_build_cmcds.py` | L1b descriptors from CMCDS |
| `06b`, `06d`, `14` (`*_signal_test`) | three-generation downstream ablation |
| `07_csp_descriptors.py` | CSP structural descriptors |
| `08_external_overlap.py`, `09_build_external.py`, `11_build_external_dg25.py`, `10_external_validate.py` | external datasets and validation |
| `13_failure_decomposition.py` | per-fold failure-mode decomposition |
| `15_leave_one_doi.py` | S2c fold indices, R²_aff ≡ η² |
| `16_zeroorder_scan.py` | zero-order correlation scan |
| `17_reproducibility_ceiling.py`, `22_interaction_ceiling.py` | between-paper ceiling, additive decomposition |
| `18_s2c_emo.py`, `32_s3_emo.py` | EMO across protocols |
| `19_paper_learning_curve.py` | within-/across-laboratory learning curves |
| `20_external_odrh.py`, `24_external_all_columns.py` | external validation by column |
| `21_condition_reproducibility.py`, `36_condition_pair_bootstrap.py` | model-free condition reproducibility |
| `25_t0_sensitivity.py`, `27_recurate_phi.py` | dead-time / porosity sensitivity |
| `26_paper_number_gate.py`, `31_run_all_gates.py` | number and gate chain |
| `33_review_round2_analyses.py`, `33b`, `33c`, `33d` | round-2 statistics, additive-rule accuracy, per-column ML |
| `37_study_blocked_null.py` | S2b permutation nulls |
| `78_within_paper_attribution.py`, `79_protocol_emo_ci.py`, `80_s2c_null_parallel.py` | H1 attribution, paper-clustered CIs, S2c permutation null |
| `38_make_figures_v3.py` | the 7 publication figures |
| `_baseline_cp25.py` | shared classifier/feature builder used by several scripts |

## Licenses

- **Code** (`scripts/`): MIT — see `LICENSE`.
- **Curated data and figures** (`data/`, `figures/`): CC BY 4.0 — see `DATA_LICENSE.md`.
  They are derived from CMRT (Xu et al., *Nat. Commun.* 2023, 14:3095;
  DOI 10.1038/s41467-023-38853-3; Zenodo 10.5281/zenodo.7623903); attribution is required.

## Citation

If you use these data or code, please cite the preprint (DOI to be added once posted) and the CMRT paper:

> Xu H, Lin J, Zhang D, Mo F. *Nat. Commun.* 2023, 14:3095.
> https://doi.org/10.1038/s41467-023-38853-3
