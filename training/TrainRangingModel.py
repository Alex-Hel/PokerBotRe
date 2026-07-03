from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.RangingModel import (
    ACTION_BUCKETS,
    ALL_INTERPRETED_FEATURE_NAMES,
    FEATURE_NAMES,
    POSTFLOP_FEATURE_NAMES,
    PREFLOP_FEATURE_NAMES,
)


DEFAULT_INPUTS = (
    PROJECT_ROOT / "data" / "interpreted" / "ranging_rows.csv",
)
CATEGORICAL_FEATURES = (
    "card_bucket",
    "suitedness",
    "street",
    "seat_bucket",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help="Interpreted CSV file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--model-scope",
        choices=("postflop", "preflop", "all"),
        default="postflop",
        help="Which rows/features to train.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--random-strength", type=float, default=0.75)
    parser.add_argument("--bagging-temperature", type=float, default=1.0)
    parser.add_argument("--rsm", type=float, default=1.0)
    parser.add_argument("--od-wait", type=int, default=100)
    parser.add_argument("--test-fraction", type=float, default=0.05)
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
    args = parser.parse_args()

    inputs = args.input or list(DEFAULT_INPUTS)
    feature_names = selected_feature_names(args.model_scope)
    street_filter = selected_street_filter(args.model_scope)
    categorical_features = selected_categorical_features(feature_names)
    rows, labels = load_training_rows(
        inputs,
        feature_names=feature_names,
        street_filter=street_filter,
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
        l2_leaf_reg=args.l2_leaf_reg,
        random_strength=args.random_strength,
        bagging_temperature=args.bagging_temperature,
        rsm=args.rsm,
        od_wait=args.od_wait,
        balanced=args.balanced,
        seed=args.seed,
        feature_names=feature_names,
        categorical_features=categorical_features,
    )

    output = args.output or default_model_path(args.model_scope)
    metadata_output = args.metadata_output or default_metadata_path(args.model_scope)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output))

    metadata = build_metadata(
        inputs=inputs,
        output=output,
        labels=labels,
        train_count=len(train_rows),
        test_count=len(test_rows),
        args=args,
        street_filter=street_filter,
        feature_names=feature_names,
        categorical_features=categorical_features,
    )
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"saved model: {output}")
    print(f"saved metadata: {metadata_output}")


def load_training_rows(
    paths: list[Path],
    feature_names: tuple[str, ...],
    street_filter: str | None = None,
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
                if street_filter == "preflop" and raw_row.get("street") != "preflop":
                    continue
                if street_filter == "postflop" and raw_row.get("street") == "preflop":
                    continue

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
    l2_leaf_reg: float,
    random_strength: float,
    bagging_temperature: float,
    rsm: float,
    od_wait: int,
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
            f"`{PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'} -m pip install catboost` "
            "or run training with "
            f"`{PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'} TrainRangingModel.py`. "
            f"Current interpreter: {sys.executable}"
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
        l2_leaf_reg=l2_leaf_reg,
        random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        rsm=rsm,
        od_type="Iter",
        od_wait=od_wait,
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
    street_filter: str | None,
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
            feature for feature in ALL_INTERPRETED_FEATURE_NAMES if feature not in feature_names
        ],
        "action_buckets": list(ACTION_BUCKETS),
        "label_counts": dict(Counter(labels)),
        "train_count": train_count,
        "test_count": test_count,
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_strength": args.random_strength,
        "bagging_temperature": args.bagging_temperature,
        "rsm": args.rsm,
        "od_wait": args.od_wait,
        "balanced": args.balanced,
        "model_scope": args.model_scope,
        "street_filter": street_filter,
        "seed": args.seed,
    }


def selected_feature_names(model_scope: str) -> tuple[str, ...]:
    if model_scope == "preflop":
        return tuple(PREFLOP_FEATURE_NAMES)
    if model_scope == "postflop":
        return tuple(POSTFLOP_FEATURE_NAMES)
    return tuple(ALL_INTERPRETED_FEATURE_NAMES)


def selected_street_filter(model_scope: str) -> str | None:
    if model_scope in {"preflop", "postflop"}:
        return model_scope
    return None


def default_model_path(model_scope: str) -> Path:
    return PROJECT_ROOT / "models" / f"{model_scope}_ranging_action_model.cbm"


def default_metadata_path(model_scope: str) -> Path:
    return PROJECT_ROOT / "models" / f"{model_scope}_ranging_action_model_metadata.json"


def selected_categorical_features(feature_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        feature
        for feature in CATEGORICAL_FEATURES
        if feature in feature_names
    )


if __name__ == "__main__":
    main()
