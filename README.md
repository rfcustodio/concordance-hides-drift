# When Concordance Hides Drift

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23809378.svg)](https://doi.org/10.5281/zenodo.23809378)

Code, simulated data, figure sources and a machine-readable label schema for the
article

> Ralo e Borges CL, Cerri GG, Custódio R. *When Concordance Hides Drift: Jointly
> Monitoring Clinical AI Performance and Clinician Interception After
> Deployment, with an Application to Medical Imaging.* Submitted to the
> Journal of Imaging Informatics in Medicine, 2026.

Everything needed to reproduce the numbers, tables and figures of the article
is in this repository.

## Contents

| Path | Content |
| --- | --- |
| `src/cdrift/risk.py` | Error and interception model; exact risk decomposition (Proposition 3.1); concordance and decision-referenced estimators (Propositions 3.3 and 3.4) |
| `src/cdrift/sim.py` | Vectorised Monte Carlo simulation of an AI-assisted reading service: trust dynamics, workload, deskilling, algorithmic drift, CUSUM monitoring, remediation |
| `src/cdrift/params.py` | All parameters with their default values |
| `experiments/run_all.py` | Experiments E1 to E6; writes CSV files to `results/` |
| `experiments/postprocess.py` | Tables derived for plotting and for the article |
| `results/` | Outputs of the run reported in the article (seed 2026, 400 replicates, 1,000 calibration replicates) and `run_metadata.json` |
| `figures/` | Every figure as TikZ/pgfplots source reading `results/`; `all_figures.pdf` is the compiled set |
| `schema/` | JSON Schema of the Dynamic AI Safety Label; each surveillance estimate states its estimand, population and denominator, and an estimate that cannot be obtained (for example the harmful-override rate without a random audit) is reported as not estimable or not available |
| `examples/` | A fictional label that validates against the schema |
| `tests/` | Unit tests of the identities and of the false-alarm calibration |

## Figures in the article

| Article | Source | Experiment |
| --- | --- | --- |
| Fig. 1 | `figures/fig1_framework.tikz` | conceptual |
| Fig. 2 | `figures/fig2_landscape.tikz` | E1 |
| Fig. 3 | `figures/fig3_identifiability.tikz` | E2 |
| Fig. 4 | `figures/fig4_incorporation.tikz` | E3 |
| Fig. 5 | `figures/fig5_excess.tikz` | E4 |
| Fig. 6 | `figures/fig6_timecourse.tikz` | E4, scenario S4 |
| Fig. 7 | `figures/fig7_audit.tikz` | E5 |
| Fig. 8 | `figures/fig8_cycle.tikz` | conceptual |
| Fig. S1 (not in the article) | `figures/figS1_decomposition.tikz` | E4, scenario S3 |

Table 2 of the article is `results/e4_detection_table.tex`; the full summary is
`results/e4_summary.csv` and `results/summary.md`.

## Reproduce

Python 3.10 or later.

```bash
pip install -r requirements.txt        # exact versions used: requirements-lock.txt
make test                               # identities and calibration (about 5 s)
make results                            # all experiments (about 3 min on one core)
make figures                            # needs TeX Live with pgfplots 1.18
```

`python -I experiments/run_all.py --quick --out /tmp/quick` runs a smoke test
in under a minute.

## Model in brief

For each case stratum and truth class, the AI errs with probability `e`; the
clinician verifies with probability `v`, reads correctly unassisted with
probability `h`, and when she disagrees with the AI keeps her own read with
probability `rho_r` if she is right and `rho_w` if she is wrong. Interception is
`c = v h rho_r`, harmful override is `o = v (1 - h) rho_w`, and risk is
`R = E[S {e (1 - c) + (1 - e) o}]`. Trust rises with observed agreement and falls
faster with observed disagreement; verification falls as trust and workload
rise; unassisted accuracy decays with sustained non-verification.

## Monitoring strategies compared

| Strategy | Signal | Detects |
| --- | --- | --- |
| `report_referenced` | AI output against the final decision | algorithmic drift only, and only while verification is preserved |
| `concordance` | AI and final decision disagreement rate (two-sided) | any change that moves concordance; blind to masked drift; cannot attribute |
| `delayed_truth` | AI against pathology or follow-up, positives, 90-day lag | algorithmic drift |
| `dual_audit` | AI-blinded expert reads on a random fraction, plus optional sentinels | both, attributable; costly at useful power |
| `dual_truth_sentinel` | delayed truth for the AI, projected sentinel cases for interception | both, attributable, low cost |

Sentinel cases are synthetic or retrospective studies with a known AI error,
placed in a quality assurance worklist, as in threat image projection for
baggage screening. They are never used on live patient studies.

## Limitations

All parameters are illustrative. The simulation shows what monitoring designs
can and cannot identify under stated assumptions; it does not estimate the size
of drift in any real deployment.

## Funding

C.L.R.B. was supported by a doctoral scholarship from the São Paulo Research
Foundation (FAPESP), grant 2024/14707-3.

## Licence

Code: MIT (`LICENSE`). Figures, data, schema and example: CC BY 4.0
(`LICENSE-figures-data.txt`).

## How to cite

See `CITATION.cff`. All versions are archived on Zenodo; the concept DOI
[10.5281/zenodo.23809377](https://doi.org/10.5281/zenodo.23809377) always
resolves to the latest version.
