import argparse
import json
import time
import csv

from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from .dataset import load_split
from .model import HTMLFeatureUnion


def metrics(y_true, y_pred):

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision_phishing": float(
            precision_score(
                y_true, y_pred,
                pos_label=1,
                zero_division=0
            )
        ),

        "recall_phishing": float(
            recall_score(
                y_true, y_pred,
                pos_label=1,
                zero_division=0
            )
        ),

        "f1_phishing": float(
            f1_score(
                y_true, y_pred,
                pos_label=1,
                zero_division=0
            )
        ),

        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),

        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["legitimate", "phishing"],
            zero_division=0,
            output_dict=True
        )
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts"
    )

    parser.add_argument(
        "--max-text-features",
        type=int,
        default=3000
    )

    args = parser.parse_args()

    overall_start = time.time()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60, flush=True)
    print("URLSight HTML/DOM PHISHING DETECTION TRAINING", flush=True)
    print("=" * 60, flush=True)

    # -------------------------------------------------
    # STEP 1: LOAD TRAINING DATA
    # -------------------------------------------------

    print("\n[STEP 1/7] Loading trainval dataset...", flush=True)

    trainval, skipped_train = load_split(
        args.dataset_root, "trainval"
    )

    if len(trainval) < 20:
        raise SystemExit(
            f"Not enough usable training samples: {len(trainval)}"
        )

    print(
        f"Trainval loaded: {len(trainval)} usable samples",
        flush=True
    )

    # -------------------------------------------------
    # STEP 2: LOAD TEST DATA
    # -------------------------------------------------

    print("\n[STEP 2/7] Loading test dataset...", flush=True)

    test, skipped_test = load_split(
        args.dataset_root, "test"
    )

    print(
        f"Test loaded: {len(test)} usable samples",
        flush=True
    )

    # Save skipped samples

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
        writer.writerows(skipped_train + skipped_test)

    # -------------------------------------------------
    # STEP 3: TRAIN / VALIDATION SPLIT
    # -------------------------------------------------

    print(
        "\n[STEP 3/7] Creating hostname-grouped train/validation split...",
        flush=True
    )

    y = np.array([
        record["label"] for record in trainval
    ])

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

    print(f"Training samples: {len(X_train)}", flush=True)
    print(f"Validation samples: {len(X_val)}", flush=True)

    print(
        f"Training phishing: {sum(y_train == 1)} | "
        f"Legitimate: {sum(y_train == 0)}",
        flush=True
    )

    print(
        f"Validation phishing: {sum(y_val == 1)} | "
        f"Legitimate: {sum(y_val == 0)}",
        flush=True
    )

    # -------------------------------------------------
    # STEP 4: MODEL TRAINING
    # -------------------------------------------------

    print(
        "\n[STEP 4/7] Starting model comparison...",
        flush=True
    )

    candidates = {

        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0
        ),

        "linear_svm": LinearSVC(
            class_weight="balanced",
            C=1.0,
            random_state=42
        )
    }

    results = {}

    best_name = None
    best_score = -1
    best_model = None

    for name, classifier in candidates.items():

        print(
            f"\n>>> Starting {name}...",
            flush=True
        )

        model_start = time.time()

        pipeline = Pipeline([
            (
                "features",
                HTMLFeatureUnion(
                    max_text_features=args.max_text_features,
                    min_df=2
                )
            ),
            ("classifier", classifier)
        ])

        print(
            "    Fitting DOM and TF-IDF feature extractors...",
            flush=True
        )

        pipeline.fit(X_train, y_train)

        print(
            "    Predicting validation samples...",
            flush=True
        )

        predictions = pipeline.predict(X_val)

        result = metrics(y_val, predictions)
        results[name] = result

        elapsed = time.time() - model_start

        print(f"\n{name} completed in {elapsed:.2f}s", flush=True)

        print(
            json.dumps(
                {
                    k: v for k, v in result.items()
                    if k != "classification_report"
                },
                indent=2
            ),
            flush=True
        )

        if result["f1_phishing"] > best_score:

            best_score = result["f1_phishing"]
            best_name = name
            best_model = pipeline

    # -------------------------------------------------
    # STEP 5: REFIT SELECTED MODEL
    # -------------------------------------------------

    print(
        f"\n[STEP 5/7] Best validation model: {best_name}",
        flush=True
    )

    print(
        "Refitting selected model using all trainval samples...",
        flush=True
    )

    refit_start = time.time()

    best_model.fit(trainval, y)

    print(
        f"Refitting completed in {time.time() - refit_start:.2f}s",
        flush=True
    )

    model_path = output / "best_html_dom_model.joblib"

    joblib.dump(best_model, model_path)

    print(f"Model saved: {model_path}", flush=True)

    # -------------------------------------------------
    # STEP 6: FINAL TEST EVALUATION
    # -------------------------------------------------

    print(
        "\n[STEP 6/7] Evaluating on untouched test dataset...",
        flush=True
    )

    if test:

        y_test = np.array([
            record["label"] for record in test
        ])

        test_predictions = best_model.predict(test)

        test_results = metrics(
            y_test,
            test_predictions
        )

        with open(
            output / "test_results.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                {
                    "selected_model": best_name,
                    "test_samples": len(test),
                    "metrics": test_results
                },
                f,
                indent=2
            )

        print(
            json.dumps(
                {
                    k: v for k, v in test_results.items()
                    if k != "classification_report"
                },
                indent=2
            ),
            flush=True
        )

    else:

        print(
            "No usable test samples. Test evaluation skipped.",
            flush=True
        )

    # -------------------------------------------------
    # STEP 7: SAVE REPORTS
    # -------------------------------------------------

    print(
        "\n[STEP 7/7] Saving training reports...",
        flush=True
    )

    summary = {
        "selected_model": best_name,
        "validation_results": results,
        "usable_trainval": len(trainval),
        "usable_test": len(test),
        "skipped_trainval": len(skipped_train),
        "skipped_test": len(skipped_test),
        "labels": {
            "0": "legitimate",
            "1": "phishing"
        }
    }

    with open(
        output / "validation_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(summary, f, indent=2)

    total_time = time.time() - overall_start

    print("\n" + "=" * 60, flush=True)
    print("TRAINING COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 60, flush=True)

    print(
        f"Total time: {total_time / 60:.2f} minutes",
        flush=True
    )

    print(f"Selected model: {best_name}", flush=True)
    print(f"Saved model: {model_path.resolve()}", flush=True)
    print(f"Reports folder: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
