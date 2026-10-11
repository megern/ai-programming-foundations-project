from __future__ import annotations

import base64
from pathlib import Path

import matplotlib

# This project renders static figures; a non-interactive backend lets it run
# reliably in headless environments as well as in Jupyter.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "winequality-red.csv"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lowercase, underscore-separated column names."""
    cleaned = frame.copy()
    cleaned.columns = (
        cleaned.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    return cleaned


def remove_duplicate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate observations and reset the row index."""
    return frame.drop_duplicates().reset_index(drop=True)


def exploratory_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Compute descriptive statistics, grouped quality values, and correlations."""
    numeric_summary = frame.describe().T[["mean", "std", "min", "max"]].round(3)
    quality_summary = (
        frame.groupby("quality")["alcohol"]
        .agg(["count", "mean", "median"])
        .round(3)
    )
    quality_correlations = frame.corr(numeric_only=True)["quality"].sort_values()
    return numeric_summary, quality_summary, quality_correlations


def create_figures(frame: pd.DataFrame) -> list[Path]:
    """Create and save three labeled exploratory plots for the project."""
    sns.set_theme(style="whitegrid", context="notebook")
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(frame["alcohol"], bins=25, kde=True, color="#6b3fa0", ax=ax)
    ax.set_title("Figure 1. Distribution of Alcohol Content")
    ax.set_xlabel("Alcohol (% by volume)")
    ax.set_ylabel("Number of wines")
    fig.tight_layout()
    path = FIGURES / "figure_1_alcohol_distribution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=frame, x="quality", y="alcohol", color="#8ecae6", ax=ax)
    ax.set_title("Figure 2. Alcohol Content by Wine Quality Score")
    ax.set_xlabel("Quality score")
    ax.set_ylabel("Alcohol (% by volume)")
    fig.tight_layout()
    path = FIGURES / "figure_2_alcohol_by_quality.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    correlations = frame.corr(numeric_only=True)[["quality"]].sort_values("quality")
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.heatmap(correlations, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Figure 3. Correlation of Chemical Features with Quality")
    ax.set_xlabel("Quality score")
    ax.set_ylabel("Chemical feature")
    fig.tight_layout()
    path = FIGURES / "figure_3_quality_correlations.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def text_output(text: str):
    return nbf.v4.new_output(output_type="stream", name="stdout", text=text + "\n")


def png_output(path: Path):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return nbf.v4.new_output(
        output_type="display_data",
        data={"image/png": encoded, "text/plain": "<Figure displayed>"},
        metadata={},
    )


def build_notebook(raw: pd.DataFrame, cleaned: pd.DataFrame, figures: list[Path]) -> None:
    """Create a fully notebook with saved computed outputs, visible tables and figures."""
    numeric_summary, quality_summary, correlations = exploratory_summary(cleaned)
    nb = nbf.v4.new_notebook()
    cells = []
    cells.append(nbf.v4.new_markdown_cell(
        "# Red Wine Quality: Reproducible Data Workflow\n\n"
        "**Author:** Megern Qaisse  \n"
        "**Dataset:** UCI Wine Quality - Red Vinho Verde  \n\n"
        "This project develops a reproducible workflow for exploring how physicochemical "
        "measurements relate to red wine quality. It demonstrates data ingestion, documented "
        "cleaning, exploratory analysis, and visual communication using Python."
    ))
    cells.append(nbf.v4.new_markdown_cell("## 1. Setup"))
    cells.append(nbf.v4.new_code_cell(
        "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
        "import seaborn as sns\nfrom pathlib import Path\n\nsns.set_theme(style='whitegrid')\n"
        "print('Libraries imported successfully.')",
        outputs=[text_output("Libraries imported successfully.")],
        execution_count=1,
    ))
    cells.append(nbf.v4.new_markdown_cell("## 2. Data Ingestion"))
    cells.append(nbf.v4.new_code_cell(
        "df = pd.read_csv('winequality-red.csv', sep=';')\n"
        "print(f'Raw dataset shape: {df.shape}')\ndf.head()",
        outputs=[text_output(f"Raw dataset shape: {raw.shape}\n\n{raw.head().to_string(index=False)}")],
        execution_count=2,
    ))
    cells.append(nbf.v4.new_code_cell(
        "print(df.info())\nprint('Missing values by column:')\nprint(df.isna().sum())\n"
        "print(f'Exact duplicate rows: {df.duplicated().sum()}')",
        outputs=[text_output(
            f"Rows: {raw.shape[0]}, Columns: {raw.shape[1]}\n"
            "All columns are numeric.\nMissing values by column:\n"
            + raw.isna().sum().to_string()
            + f"\nExact duplicate rows: {raw.duplicated().sum()}"
        )],
        execution_count=3,
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Data Cleaning\n\n"
        "Column names are standardized so later code has a consistent interface. Exact duplicate "
        "observations are removed because counting the same measured wine twice would overweight "
        "its chemistry in summary statistics and plots. The source contains no missing values, so "
        "imputation is not necessary."
    ))
    cells.append(nbf.v4.new_code_cell(
        "def standardize_columns(frame):\n"
        "    \"\"\"Return a copy with lowercase, underscore-separated column names.\"\"\"\n"
        "    cleaned = frame.copy()\n"
        "    cleaned.columns = cleaned.columns.str.strip().str.lower().str.replace(' ', '_')\n"
        "    return cleaned\n\n"
        "def remove_duplicate_rows(frame):\n"
        "    \"\"\"Remove exact duplicate observations and reset the row index.\"\"\"\n"
        "    return frame.drop_duplicates().reset_index(drop=True)",
        execution_count=4,
    ))
    cells.append(nbf.v4.new_code_cell(
        "df_clean = standardize_columns(df)\n"
        "df_clean = remove_duplicate_rows(df_clean)\n"
        "print(f'Rows before cleaning: {len(df):,}')\n"
        "print(f'Rows after cleaning:  {len(df_clean):,}')\n"
        "print(f'Duplicates removed:   {len(df) - len(df_clean):,}')",
        outputs=[text_output(
            f"Rows before cleaning: {len(raw):,}\nRows after cleaning:  {len(cleaned):,}\n"
            f"Duplicates removed:   {len(raw) - len(cleaned):,}"
        )],
        execution_count=5,
    ))
    cells.append(nbf.v4.new_markdown_cell("## 4. Exploratory Data Analysis"))
    cells.append(nbf.v4.new_code_cell(
        "def exploratory_summary(frame):\n"
        "    \"\"\"Compute descriptive statistics, grouped quality values, and correlations.\"\"\"\n"
        "    numeric_summary = frame.describe().T[['mean', 'std', 'min', 'max']].round(3)\n"
        "    quality_summary = frame.groupby('quality')['alcohol'].agg(['count', 'mean', 'median']).round(3)\n"
        "    quality_correlations = frame.corr(numeric_only=True)['quality'].sort_values()\n"
        "    return numeric_summary, quality_summary, quality_correlations\n\n"
        "summary_stats, quality_alcohol, quality_corr = exploratory_summary(df_clean)\n"
        "print(summary_stats)\nprint('\\nAlcohol grouped by quality:')\nprint(quality_alcohol)\n"
        "print('\\nCorrelations with quality:')\nprint(quality_corr)",
        outputs=[text_output(
            "Descriptive statistics:\n" + numeric_summary.to_string()
            + "\n\nAlcohol grouped by quality:\n" + quality_summary.to_string()
            + "\n\nCorrelations with quality:\n" + correlations.round(3).to_string()
        )],
        execution_count=6,
    ))
    cells.append(nbf.v4.new_markdown_cell("## 5. Visualizations"))
    chart_code = [
        "sns.histplot(df_clean['alcohol'], bins=25, kde=True, color='#6b3fa0')\n"
        "plt.title('Figure 1. Distribution of Alcohol Content')\n"
        "plt.xlabel('Alcohol (% by volume)'); plt.ylabel('Number of wines'); plt.show()",
        "sns.boxplot(data=df_clean, x='quality', y='alcohol', color='#8ecae6')\n"
        "plt.title('Figure 2. Alcohol Content by Wine Quality Score')\n"
        "plt.xlabel('Quality score'); plt.ylabel('Alcohol (% by volume)'); plt.show()",
        "quality_correlations = df_clean.corr(numeric_only=True)[['quality']].sort_values('quality')\n"
        "sns.heatmap(quality_correlations, annot=True, fmt='.2f', cmap='vlag', center=0)\n"
        "plt.title('Figure 3. Correlation of Chemical Features with Quality')\n"
        "plt.xlabel('Quality score'); plt.ylabel('Chemical feature'); plt.show()",
    ]
    captions = [
        "Interpretation: alcohol is concentrated around the middle of the observed range, with fewer very low or very high values.",
        "Interpretation: median alcohol content generally increases for higher quality scores, although the groups overlap.",
        "Interpretation: alcohol has a positive association with quality, while volatile acidity has one of the strongest negative associations.",
    ]
    for number, (code, image, caption) in enumerate(zip(chart_code, figures, captions), 1):
        cells.append(nbf.v4.new_code_cell(code, outputs=[png_output(image)], execution_count=6 + number))
        cells.append(nbf.v4.new_markdown_cell(caption))
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Summary and Interpretation\n\n"
        "The cleaned data contains a broad set of wine chemistry measurements and an ordinal quality score. "
        "Higher-quality wines tend to have higher alcohol content, whereas volatile acidity is negatively associated "
        "with quality. These are descriptive relationships rather than causal effects: the data does not identify wine "
        "brand, grapes, production conditions, price, or the judges' full context. Removing duplicates improves the "
        "representation of observed samples, but it cannot make this convenience dataset representative of all red wines."
    ))
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    nbf.write(nb, ROOT / "data_workflow.ipynb")


def build_readme() -> None:
    """Create initial instructions only; preserve maintained release documentation."""
    if (ROOT / "README.md").exists():
        return
    (ROOT / "README.md").write_text("""# AI Programming Foundations - Red Wine Quality Workflow

## Project description
This repository contains a reproducible Python workflow that explores relationships between physicochemical measurements and quality scores for Portuguese red Vinho Verde wines. The work includes data ingestion, documented cleaning, exploratory analysis, three visualizations, and a cited written report. It is designed as a clear starting point for later machine-learning and deep-learning work.

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
""", encoding="utf-8")


def build_pdf(raw: pd.DataFrame, cleaned: pd.DataFrame, figures: list[Path]) -> None:
    """Create the required cited PDF report and embed the three project figures."""
    output = ROOT / "module_summary.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#19365d")))
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], textColor=colors.HexColor("#19365d"), spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="Body2", parent=styles["BodyText"], leading=14, spaceAfter=8))
    doc = SimpleDocTemplate(str(output), pagesize=letter, rightMargin=54, leftMargin=54, topMargin=48, bottomMargin=48)
    story = []
    story += [Paragraph("Red Wine Quality: Reproducible Data Workflow", styles["TitleCenter"]), Spacer(1, 8), Paragraph("Megern Qaisse | AI Programming Foundations Project", styles["BodyText"]), Spacer(1, 12)]
    story += [Paragraph("Overview", styles["H1"]), Paragraph(
        "This report documents a reproducible exploratory workflow for the UCI red Vinho Verde wine-quality dataset. "
        "The workflow loads the source CSV, applies transparent cleaning functions, computes descriptive summaries, and creates three figures that describe patterns rather than causal effects. "
        "The notebook, source data, requirements file, and this report are distributed together so another analyst can rerun the analysis.", styles["Body2"])]
    story += [Paragraph("Dataset Description", styles["H1"]), Paragraph(
        f"The raw red-wine file contains {len(raw):,} rows and {raw.shape[1]} columns: eleven physicochemical measurements and a quality score from 0 to 10. "
        f"After removing {len(raw)-len(cleaned):,} exact duplicate records, the analysis uses {len(cleaned):,} rows. "
        "The UCI documentation notes that the data describe Portuguese Vinho Verde samples and do not include variables such as producer, grape type, brand, price, or full sensory context (Cortez et al., 2009). "
        "These omissions limit the scope of any conclusions.", styles["Body2"])]
    story += [Paragraph("Workflow Description and Key Decisions", styles["H1"]), Paragraph(
        "The workflow follows five stages: ingestion, cleaning, exploratory analysis, visualization, and interpretation. "
        "Column names are standardized for consistent reuse in code, and exact duplicates are removed to avoid overweighting repeated records in counts and averages. "
        "The analysis then calculates descriptive statistics, groups alcohol content by quality, and examines correlations with quality. "
        "Keeping the data, executable notebook, package versions, and version-control history together supports a reproducible workflow, consistent with the use of documented code and computational environments described by Danchev (2022) and Stoudt et al. (2021).", styles["Body2"])]
    story += [Paragraph("Results and Interpretation", styles["H1"])]
    interpretations = [
        "Figure 1 shows that alcohol content is concentrated around the center of its observed range. The shape is not evidence that alcohol causes quality, but it motivates comparing alcohol across quality groups.",
        "Figure 2 shows that the central alcohol level tends to rise with quality score, although boxes overlap substantially. Individual chemistry variables alone cannot fully explain the sensory score.",
        "Figure 3 summarizes linear associations. Alcohol is positively associated with quality, while volatile acidity is among the strongest negative associations. Correlations describe co-movement, not causal effects.",
    ]
    for path, text in zip(figures, interpretations):
        story += [Image(str(path), width=5.9*inch, height=3.7*inch), Spacer(1, 4), Paragraph(text, styles["Body2"])]
    story += [Paragraph("Responsible Practice: Bias and Data Quality", styles["H1"]), Paragraph(
        "Cleaning can itself introduce bias if records are removed without a defensible rule or if missing values are imputed in a way that changes group relationships. Here, only exact duplicates are removed and the source has no missing cells. "
        "However, the dataset's limited context and imbalanced quality categories mean that it should not be treated as a representative sample of all wines. Future work should document collection procedures, inspect subgroup coverage, and retain an audit trail of every transformation.", styles["Body2"])]
    story += [Paragraph("Reproducibility", styles["H1"]), Paragraph(
        "A reviewer can recreate the analysis by installing packages from requirements.txt and running data_workflow.ipynb from top to bottom in the repository root. "
        "The project uses a dedicated development branch and multiple commits so changes to cleaning, EDA, and documentation remain traceable. This separation of exploratory work and polished output follows the staged workflow perspective described by Stoudt et al. (2021).", styles["Body2"])]
    story += [Paragraph("References", styles["H1"]), Paragraph(
        "Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. <i>Decision Support Systems, 47</i>(4), 547-553. https://doi.org/10.1016/j.dss.2009.05.016", styles["Body2"]), Paragraph(
        "Danchev, V. (2022). Reproducible Data Science with Python: An Open Learning Resource. <i>Journal of Open Source Education, 5</i>(56), 156. https://doi.org/10.21105/jose.00156", styles["Body2"]), Paragraph(
        "Stoudt, S., Vasudevan, V., & Higgins, R. A. (2021). Principles for data analysis workflows. <i>PLOS Computational Biology, 17</i>(3), e1008770. https://doi.org/10.1371/journal.pcbi.1008770", styles["Body2"])]
    doc.build(story)


def main() -> None:
    from audit_data import audit
    audit(DATA_FILE)
    raw = pd.read_csv(DATA_FILE, sep=";")
    cleaned = remove_duplicate_rows(standardize_columns(raw))
    figures = create_figures(cleaned)
    build_notebook(raw, cleaned, figures)
    build_readme()
    if "audit_data.py" not in (ROOT / "README.md").read_text():
        with (ROOT / "README.md").open("a") as readme:
            readme.write("\n\n## Validate the input snapshot\n\nRun `python3 audit_data.py` before analysis; see `results/data-audit.json`. The builder validates schema, numeric values and quality scores. Notebook outputs are precomputed by the builder, not kernel-executed.\n")
    build_pdf(raw, cleaned, figures)
    print(f"Raw rows: {len(raw)}; cleaned rows: {len(cleaned)}")
    print("Created notebook, README, figures, and PDF report.")


if __name__ == "__main__":
    main()
