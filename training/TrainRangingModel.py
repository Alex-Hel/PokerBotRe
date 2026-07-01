from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

from models.RangingModel import ACTION_BUCKETS, FEATURE_NAMES


DEFAULT_INPUTS = (
    Path("../data/interpreted/internal_ranging_rows.csv"),
    Path("../data/interpreted/external_ranging_rows.csv"),
)
DEFAULT_MODEL_PATH = Path("../models/ranging_action_model.cbm")
DEFAULT_METADATA_PATH = Path("../models/ranging_action_model_metadata.json")
CATEGORICAL_FEATURES = ("hole_card_class", "street", "position_bucket")
HOLE_CARD_FEATURES = (
    "hole_card_class",
    "hole_high_rank",
    "hole_low_rank",
    "hole_rank_gap",
    "hole_pair",
    "hole_suited",
    "hole_broadway_fraction",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help="Interpreted CSV file. Can be passed multiple times.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for quick smoke tests.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use CatBoost balanced class weights.",
    )
    parser.add_argument(
        "--include-hole-card-features",
        action="store_true",
        help="Include actual hole-card identity/strength features during training.",
    )
    args = parser.parse_args()

    inputs = args.input or list(DEFAULT_INPUTS)
    feature_names = selected_feature_names(args.include_hole_card_features)
    categorical_features = selected_categorical_features(feature_names)
    rows, labels = load_training_rows(
        inputs,
        feature_names=feature_names,
        max_rows=args.max_rows,
    )
    if not rows:
        raise RuntimeError("No training rows found.")

    train_rows, train_labels, test_rows, test_labels = split_rows(
        rows,
        labels,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    model = train_model(
        train_rows=train_rows,
        train_labels=train_labels,
        test_rows=test_rows,
        test_labels=test_labels,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        balanced=args.balanced,
        seed=args.seed,
        feature_names=feature_names,
        categorical_features=categorical_features,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output))

    metadata = build_metadata(
        inputs=inputs,
        output=args.output,
        labels=labels,
        train_count=len(train_rows),
        test_count=len(test_rows),
        args=args,
        feature_names=feature_names,
        categorical_features=categorical_features,
    )
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"saved model: {args.output}")
    print(f"saved metadata: {args.metadata_output}")


def load_training_rows(
    paths: list[Path],
    feature_names: tuple[str, ...],
    max_rows: int | None = None,
) -> tuple[list[list[object]], list[str]]:
    rows: list[list[object]] = []
    labels: list[str] = []
    action_counts = Counter()

    for path in paths:
        if not path.exists():
            print(f"skipping missing input: {path}")
            continue

        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for raw_row in reader:
                label = raw_row.get("action_bucket", "")
                if label not in ACTION_BUCKETS:
                    continue

                rows.append(feature_row(raw_row, feature_names))
                labels.append(label)
                action_counts[label] += 1

                if len(rows) % 10000 == 0:
                    print(f"loaded {len(rows)} rows")
                if max_rows is not None and len(rows) >= max_rows:
                    print(f"loaded {len(rows)} rows from capped dataset")
                    print(f"action counts: {dict(action_counts)}")
                    return rows, labels

    print(f"loaded {len(rows)} rows")
    print(f"action counts: {dict(action_counts)}")
    return rows, labels


def feature_row(raw_row: dict[str, str], feature_names: tuple[str, ...]) -> list[object]:
    return [
        normalize_feature_value(name, raw_row.get(name, ""))
        for name in feature_names
    ]


def normalize_feature_value(name: str, value: str) -> object:
    if name in CATEGORICAL_FEATURES:
        return value or "unknown"

    if value in {"", "None", "none", "nan", "NaN"}:
        return math.nan

    try:
        return float(value)
    except ValueError:
        return math.nan


def split_rows(
    rows: list[list[object]],
    labels: list[str],
    test_fraction: float,
    seed: int,
) -> tuple[list[list[object]], list[str], list[list[object]], list[str]]:
    combined = list(zip(rows, labels))
    rng = random.Random(seed)
    rng.shuffle(combined)

    test_count = int(len(combined) * test_fraction)
    test = combined[:test_count]
    train = combined[test_count:]

    train_rows = [row for row, _ in train]
    train_labels = [label for _, label in train]
    test_rows = [row for row, _ in test]
    test_labels = [label for _, label in test]
    return train_rows, train_labels, test_rows, test_labels


def train_model(
    train_rows: list[list[object]],
    train_labels: list[str],
    test_rows: list[list[object]],
    test_labels: list[str],
    iterations: int,
    depth: int,
    learning_rate: float,
    balanced: bool,
    seed: int,
    feature_names: tuple[str, ...],
    categorical_features: tuple[str, ...],
):
    try:
        from catboost import CatBoostClassifier, Pool
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CatBoost is required to train the ranging action model. Install it with "
            "`pip install catboost` inside the project venv."
        ) from exc

    categorical_indices = [
        index
        for index, name in enumerate(feature_names)
        if name in categorical_features
    ]

    train_pool = Pool(
        train_rows,
        label=train_labels,
        feature_names=list(feature_names),
        cat_features=categorical_indices,
    )
    test_pool = Pool(
        test_rows,
        label=test_labels,
        feature_names=list(feature_names),
        cat_features=categorical_indices,
    )

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        random_seed=seed,
        auto_class_weights="Balanced" if balanced else None,
        allow_writing_files=False,
        verbose=100,
    )
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    predictions = model.predict(test_pool)
    accuracy = sum(
        1
        for predicted, expected in zip(predictions, test_labels)
        if str(predicted[0]) == expected
    ) / max(len(test_labels), 1)
    print(f"validation accuracy: {accuracy:.4f}")
    return model


def build_metadata(
    inputs: list[Path],
    output: Path,
    labels: list[str],
    train_count: int,
    test_count: int,
    args,
    feature_names: tuple[str, ...],
    categorical_features: tuple[str, ...],
) -> dict:
    return {
        "model_path": str(output),
        "input_files": [str(path) for path in inputs],
        "feature_names": list(feature_names),
        "categorical_features": list(categorical_features),
        "categorical_feature_indices": [
            index
            for index, name in enumerate(feature_names)
            if name in categorical_features
        ],
        "excluded_features": [
            feature for feature in FEATURE_NAMES if feature not in feature_names
        ],
        "action_buckets": list(ACTION_BUCKETS),
        "label_counts": dict(Counter(labels)),
        "train_count": train_count,
        "test_count": test_count,
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "balanced": args.balanced,
        "include_hole_card_features": args.include_hole_card_features,
        "seed": args.seed,
    }


def selected_feature_names(include_hole_card_features: bool) -> tuple[str, ...]:
    if include_hole_card_features:
        return tuple(FEATURE_NAMES)
    return tuple(
        feature
        for feature in FEATURE_NAMES
        if feature not in HOLE_CARD_FEATURES
    )


def selected_categorical_features(feature_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        feature
        for feature in CATEGORICAL_FEATURES
        if feature in feature_names
    )


if __name__ == "__main__":
    main()
