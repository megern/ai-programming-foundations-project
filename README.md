# AI Programming Foundations - Red Wine Quality Workflow

## Project description
This repository contains a reproducible Python workflow that explores relationships between physicochemical measurements and quality scores for Portuguese red Vinho Verde wines. The work includes data ingestion, documented cleaning, exploratory analysis, three visualizations, and a cited written report. It is designed as a clear starting point for later machine-learning and deep-learning work.

**Public repository:** https://github.com/megrn/ai-programming-foundations-project

## What I built
- `data_workflow.ipynb`: executed notebook with setup, ingestion, cleaning functions, EDA, visualizations, and interpretation.
- `module_summary.pdf`: written summary with in-text citations and references.
- `winequality-red.csv`: project dataset.
- `figures/`: saved figures used in the notebook and report.

## Dataset
**Wine Quality - Red Vinho Verde** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality). The red-wine file has 1,599 observations and 12 columns before duplicate removal. It contains physicochemical tests and a sensory quality score.

## How to run
1. Create and activate a Python 3.9+ virtual environment.
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
