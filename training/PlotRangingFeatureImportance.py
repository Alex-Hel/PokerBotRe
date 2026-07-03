from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ranging_action_model.cbm"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "ranging_action_model_metadata.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "models" / "ranging_feature_importance.png"
DEFAULT_CSV_PATH = PROJECT_ROOT / "models" / "ranging_feature_importance.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-scope",
        choices=("legacy", "preflop", "postflop"),
        default="legacy",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    try:
        from catboost import CatBoostClassifier
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Plotting feature importance requires catboost and matplotlib. Run with "
            f"`{PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'} PlotRangingFeatureImportance.py`."
        ) from exc

    model_path = args.model or default_model_path(args.model_scope)
    metadata_path = args.metadata or default_metadata_path(args.model_scope)
    output_path = args.output or default_output_path(args.model_scope)
    csv_output_path = args.csv_output or default_csv_path(args.model_scope)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = list(metadata["feature_names"])

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    importances = model.get_feature_importance()

    rows = sorted(
        zip(feature_names, (float(value) for value in importances)),
        key=lambda item: item[1],
        reverse=True,
    )
    write_csv(csv_output_path, rows)
    write_plot(output_path, rows[: args.top], plt, title=f"{args.model_scope.title()} Ranging Feature Importance")

    print(f"saved graph: {output_path}")
    print(f"saved csv: {csv_output_path}")


def default_model_path(model_scope: str) -> Path:
    if model_scope == "legacy":
        return DEFAULT_MODEL_PATH
    return PROJECT_ROOT / "models" / f"{model_scope}_ranging_action_model.cbm"


def default_metadata_path(model_scope: str) -> Path:
    if model_scope == "legacy":
        return DEFAULT_METADATA_PATH
    return PROJECT_ROOT / "models" / f"{model_scope}_ranging_action_model_metadata.json"


def default_output_path(model_scope: str) -> Path:
    if model_scope == "legacy":
        return DEFAULT_OUTPUT_PATH
    return PROJECT_ROOT / "models" / f"{model_scope}_ranging_feature_importance.png"


def default_csv_path(model_scope: str) -> Path:
    if model_scope == "legacy":
        return DEFAULT_CSV_PATH
    return PROJECT_ROOT / "models" / f"{model_scope}_ranging_feature_importance.csv"


def write_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["feature", "importance"])
        writer.writerows(rows)


def write_plot(path: Path, rows: list[tuple[str, float]], plt, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [name for name, _ in reversed(rows)]
    values = [value for _, value in reversed(rows)]

    height = max(6.0, len(rows) * 0.35)
    plt.figure(figsize=(12, height))
    plt.barh(names, values, color="#2f6f9f")
    plt.xlabel("Feature importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
