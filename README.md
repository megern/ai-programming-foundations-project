# AI Programming Foundations - Red Wine Quality Workflow

## Version 1.0.0

End-to-end grouped regression workflow. Groups identical feature vectors before 60/20/20 train/validation/test splitting; deduplicates complete numeric rows; fits scaling only on training; chooses ridge alpha from a fixed four-value set using validation; scores test after selection against a training-mean baseline. Saves coefficients, scaling, split hashes and metrics. predict(model, features) is the inference API. Standard library only. Dataset snapshots and descriptive fixed-split results are not independent population-generalization evidence.

See the [complete local workflow](docs/v1-workflow.md) for installation, commands, review artifacts and limits. Earlier preview notes below describe their historical tagged revisions.

## Check potential train/test feature leakage (local development)

`wine-data-audit /absolute/train.csv --compare-with /absolute/test.csv` validates
both snapshots and compares exact numeric feature vectors with the quality
target excluded. It detects repeated features even when column order, numeric
spelling or the target label differs. Output includes overlapping feature counts,
affected test-row counts, full-row duplicates, conflicting labels and both exact
snapshot hashes. Raw records are not exported.

This is a diagnostic for the project's twelve-column wine schema. Repeated
features may be legitimate observations; no exact overlap is not proof of
independence. Approximate duplicates and semantic/data-collection overlap are
outside this check.

## Public preview v0.2.0

Python 3.11+; commands below use a local Mac/Linux shell. Source and reports are public; this is a research/development preview, not a production-security certification.

```sh
git clone --branch v0.2.0 https://github.com/megern/ai-programming-foundations-project.git
cd ai-programming-foundations-project
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install .
wine-data-audit --help
```

The installed entry point was exercised outside the checkout in a separate local environment. This revision has 4 passing local tests. No hosted CI or cloud compute is used. Report an issue with the command, Python/OS versions, sanitized input and expected/actual behavior; remove credentials and private records first. Benchmarks describe the supplied fixtures and hardware.

## Project description
This repository contains a reproducible Python workflow that explores relationships between physicochemical measurements and quality scores for Portuguese red Vinho Verde wines. The work includes data ingestion, documented cleaning, exploratory analysis, three visualizations, and a cited written report. It is designed as a clear starting point for later machine-learning and deep-learning work.

**Public repository:** https://github.com/megern/ai-programming-foundations-project

## What I built
- `data_workflow.ipynb`: notebook with saved computed outputs, setup, ingestion, cleaning functions, EDA, visualizations, and interpretation.
- `module_summary.pdf`: written summary with in-text citations and references.
- `winequality-red.csv`: project dataset.
- `figures/`: saved figures used in the notebook and report.

## Dataset
**Wine Quality - Red Vinho Verde** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality). The red-wine file has 1,599 observations and 12 columns before duplicate removal. It contains physicochemical tests and a sensory quality score.

## How to run
1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `data_workflow.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
4. Run all cells from top to bottom.

`requirements.txt` was generated from the project environment with `pip freeze > requirements.txt`.

## Cleaning and bias awareness
The workflow standardizes column names and removes exact duplicates. Keeping duplicates would give repeated observations more influence in summaries and visualizations, which can distort descriptive conclusions. The dataset also lacks grape variety, producer, price, and complete sensory-context variables; therefore, conclusions should not be generalized to all wines or interpreted as causal.

## Future integration reflections
**ML workflow changes:** A predictive model would add a train/validation/test split before any scaling or feature selection, then evaluate error and calibration by quality class. The cleaning functions would remain reusable preprocessing steps.

**Neural-network preparation:** Numeric features would be standardized using training-set statistics only. The ordinal quality target would be modeled either as regression or an ordinal classification task, while class imbalance would be measured before selecting a loss function.

**Agentic automation potential:** An agent could validate incoming columns and ranges, run this notebook's quality checks, refresh figures, compare new summary statistics against a reference dataset, and open a human-review task when data drift or unexpected duplicates appear.

## Validate the dataset snapshot

```sh
python3 audit_data.py
python3 -m unittest discover -s .
```

The standard-library audit validates twelve distinct normalized columns, row widths, finite nonnegative numbers and integer quality scores in [0, 10]. It reports the exact CSV SHA-256, quality distribution and numeric duplicate counts without including raw rows. The bundled snapshot has 1,599 rows, 1,359 unique numeric rows and 240 duplicates; see `results/data-audit.json`. The artifact builder now validates inputs before analysis.

Saved notebook outputs are generated from computed tables and figures by the builder; it does not execute a notebook kernel. Run the notebook from top to bottom to independently reproduce the analysis. Snapshot validation does not establish population representativeness or causal conclusions.


### Descriptive group sensitivity

```sh
wine-data-audit winequality-red.csv --group-summary
```

`--group-summary` reports alcohol mean, sample standard deviation and sample counts for each quality group, before and after numeric duplicate removal. Small groups remain visible; singleton sample SD is null. See `results/group-summary-20261011.json`. These descriptive statistics are not causal effects, predictive model performance or confidence intervals. The lightweight installed command requires an explicit CSV path; analysis/notebook dependencies remain in `requirements.txt`.
