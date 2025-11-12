import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MODEL_FILES = [
    ("gpt5", "GPT-5"),
    ("pi0", "Pi0"),
    ("magma", "Magma"),
]

DATASET_ORDER = [
    "openx_bimanual",
    "openx_quadrupedal",
    "openx_mobile_manipulation",
    "openx_single_arm",
    "openx_wheeled_robot",
]

METRICS = [
    {
        "metric_key": "approximate_baseline_relative_mae",
        "value_label": "Baseline Normalized MAE",
        "title": "Baseline Normalized MAE on OpenX",
        "output_filename": "openx_approximate_baseline_relative_mae.png",
    },
    {
        "metric_key": "approximate_baseline_relative_mse",
        "value_label": "Baseline Normalized MSE",
        "title": "Baseline Normalized MSE on OpenX",
        "output_filename": "openx_approximate_baseline_relative_mse.png",
    },
]


def load_model_data(results_dir: Path) -> dict[str, dict]:
    """Load JSON results for the configured models."""
    data = {}
    for file_stem, _ in MODEL_FILES:
        json_path = results_dir / f"{file_stem}.json"
        with json_path.open("r") as handle:
            data[file_stem] = json.load(handle)
    return data


def extract_datasets(model_data: dict[str, dict]) -> list[str]:
    """Return the ordered list of datasets shared across models."""
    available = set()
    for model_key, _ in MODEL_FILES:
        available.update(model_data[model_key].keys())
    ordered = [dataset for dataset in DATASET_ORDER if dataset in available]
    return ordered


def format_dataset_label(dataset: str) -> str:
    """Create a readable label for an OpenX dataset name."""
    return dataset.replace("openx_", "").replace("_", " ").title()


def build_metric_frame(
    model_data: dict[str, dict],
    datasets: list[str],
    metric_key: str,
    value_label: str,
) -> pd.DataFrame:
    """Collect approximate baseline relative metric values for each model and dataset."""
    records = []
    for dataset in datasets:
        label = format_dataset_label(dataset)
        for model_key, display_name in MODEL_FILES:
            value = model_data[model_key][dataset]["approximate_relative_metrics"][metric_key]
            records.append(
                {
                    "Dataset": label,
                    "Model": display_name,
                    value_label: value,
                }
            )

    df = pd.DataFrame(records)
    df["Dataset"] = pd.Categorical(
        df["Dataset"],
        categories=[format_dataset_label(ds) for ds in datasets],
        ordered=True,
    )
    df["Model"] = pd.Categorical(
        df["Model"],
        categories=[name for _, name in MODEL_FILES],
        ordered=True,
    )
    return df


def plot_metrics(
    datasets: list[str],
    metric_frame: pd.DataFrame,
    value_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Produce a grouped bar chart showing metrics per dataset."""
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=metric_frame,
        x="Dataset",
        y=value_label,
        hue="Model",
        order=[format_dataset_label(ds) for ds in datasets],
        hue_order=[name for _, name in MODEL_FILES],
    )
    ax.set_title(title)
    ax.set_ylabel(value_label)
    ax.set_xlabel("Dataset")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    current_file = Path(__file__).resolve()
    results_dir = current_file.parent / "results" / "corrected_baseline_mae_approx"

    model_data = load_model_data(results_dir)
    datasets = extract_datasets(model_data)

    for metric in METRICS:
        metric_frame = build_metric_frame(
            model_data,
            datasets,
            metric["metric_key"],
            metric["value_label"],
        )
        output_path = results_dir / metric["output_filename"]
        plot_metrics(datasets, metric_frame, metric["value_label"], metric["title"], output_path)


if __name__ == "__main__":
    main()

