
import argparse
import csv
import json
import time

from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from .dataset import load_split
from .model import HTMLFeatureUnion


def calculate_metrics(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_phishing": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_phishing": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_phishing": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "false_positive_rate": float(
            fp / max(1, fp + tn)
        ),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def create_pipeline(args, remove_host=False, remove_images=False):

    return Pipeline([
        (
            "features",
            HTMLFeatureUnion(
                max_text_features=args.max_text_features,
                min_df=2,
                remove_host_length=remove_host,
                remove_external_image_count=remove_images
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=1.0,
                solver="lbfgs"
            )
        )
    ])


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", default="artifacts_v2")
    parser.add_argument("--max-text-features", type=int, default=3000)

    args = parser.parse_args()

    start = time.time()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("URLSight HTML/DOM MODEL V2")
    print("=" * 60)

    # STEP 1: Load datasets

    print("\n[1/6] Loading trainval dataset...")

    trainval, skipped_train = load_split(
        args.dataset_root, "trainval"
    )

    print("Usable trainval samples:", len(trainval))

    print("\n[2/6] Loading test dataset...")

    test, skipped_test = load_split(
        args.dataset_root, "test"
    )

    print("Usable test samples:", len(test))

    # STEP 2: Create the same grouped split as the baseline

    print("\n[3/6] Creating grouped train/validation split...")

    y = np.array([r["label"] for r in trainval])

    groups = []

    for record in trainval:

        hostname = (
            urlparse(record["page_url"]).hostname or ""
        ).lower()

        groups.append(
            hostname if hostname else record["sample_id"]
        )

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    train_idx, val_idx = next(
        splitter.split(
            np.zeros(len(y)),
            y,
            groups=np.array(groups)
        )
    )

    X_train = [trainval[i] for i in train_idx]
    X_val = [trainval[i] for i in val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))

    # STEP 3: Run controlled feature experiments

    experiments = {
        "V2_A_scaled": {
            "remove_host": False,
            "remove_images": False
        },
        "V2_B_without_host_length": {
            "remove_host": True,
            "remove_images": False
        },
        "V2_C_without_external_image_features": {
            "remove_host": False,
            "remove_images": True
        }
    }

    results = {}
    best_name = None
    best_score = (-1, -1)

    print("\n[4/6] Running validation experiments...")

    for name, config in experiments.items():

        print("\n" + "-" * 55)
        print("Experiment:", name)

        model = create_pipeline(
            args,
            remove_host=config["remove_host"],
            remove_images=config["remove_images"]
        )

        experiment_start = time.time()

        print("Fitting model...")

        model.fit(X_train, y_train)

        print("Predicting validation samples...")

        predictions = model.predict(X_val)

        result = calculate_metrics(y_val, predictions)

        result["training_seconds"] = round(
            time.time() - experiment_start, 2
        )

        results[name] = result

        print(json.dumps(result, indent=2))

        score = (
            result["f1_phishing"],
            result["recall_phishing"]
        )

        if score > best_score:
            best_score = score
            best_name = name

    # Save validation results

    with open(output / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # STEP 4: Select model using validation only

    print("\n[5/6] Selected model:", best_name)

    selected_config = experiments[best_name]

    final_model = create_pipeline(
        args,
        remove_host=selected_config["remove_host"],
        remove_images=selected_config["remove_images"]
    )

    # Refit on all trainval data

    print("Refitting selected model on all trainval samples...")

    X_all = trainval
    y_all = np.array([r["label"] for r in trainval])

    final_model.fit(X_all, y_all)

    model_path = output / "best_html_dom_model_v2.joblib"

    joblib.dump(final_model, model_path)

    print("Saved:", model_path)

    # STEP 5: Evaluate selected model once on held-out test set

    print("\n[6/6] Evaluating selected model on held-out test set...")

    X_test = test
    y_test = np.array([r["label"] for r in test])

    test_predictions = final_model.predict(X_test)

    test_results = calculate_metrics(
        y_test,
        test_predictions
    )

    test_results["selected_experiment"] = best_name
    test_results["test_samples"] = len(test)

    with open(output / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # Save skipped sample information

    with open(
        output / "skipped_samples.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["split", "sample", "reason"]
        )

        writer.writeheader()

        for row in skipped_train:
            writer.writerow({**row, "split": "trainval"})

        for row in skipped_test:
            writer.writerow({**row, "split": "test"})

    print("\n=== FINAL TEST RESULTS ===")
    print(json.dumps(test_results, indent=2))

    print("\nTotal time:", round(time.time() - start, 2), "seconds")

    print("\nV2 training completed successfully.")


if __name__ == "__main__":
    main()
