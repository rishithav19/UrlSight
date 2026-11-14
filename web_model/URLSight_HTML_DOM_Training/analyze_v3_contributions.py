
import joblib
import numpy as np

from pathlib import Path
from html_dom.features import extract_dom_features


model = joblib.load(
    "artifacts_v3/best_html_dom_model_v3.joblib"
)

transformer = model.named_steps["features"]
classifier = model.named_steps["classifier"]

sites = [
    ("Wikipedia", "https://www.wikipedia.org"),
    ("Google", "https://www.google.com"),
    ("GitHub", "https://github.com")
]

# Get the fitted LinearSVC estimators from calibration folds
calibrated_models = classifier.calibrated_classifiers_

dom_names = list(
    transformer.dom_vectorizer.get_feature_names_out()
)

text_names = [
    "TEXT:" + name
    for name in transformer.text_vectorizer.get_feature_names_out()
]

feature_names = dom_names + text_names

for site_name, url in sites:

    path = Path(
        f"webscraper_output/{site_name.lower()}/page.html"
    )

    if not path.exists():
        print(f"\n{site_name}: HTML not found")
        continue

    html = path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    dom_features, html_text = extract_dom_features(
        html, url
    )

    record = {
        "dom_features": dom_features,
        "html_text": html_text
    }

    transformed = transformer.transform([record])

    print("\n" + "=" * 65)
    print(site_name)
    print("=" * 65)

    probabilities = model.predict_proba([record])[0]
    classes = list(classifier.classes_)

    phishing_index = classes.index(1)

    print(
        "Prediction:",
        "PHISHING" if model.predict([record])[0] == 1
        else "LEGITIMATE"
    )

    print(
        f"Phishing probability: "
        f"{probabilities[phishing_index]:.4%}"
    )

    # Inspect each calibrated LinearSVC estimator
    for fold_index, calibrated_model in enumerate(
        calibrated_models
    ):

        estimator = calibrated_model.estimator

        coefficients = estimator.coef_[0]

        if list(estimator.classes_)[1] != 1:
            coefficients = -coefficients

        contributions = np.asarray(
            transformed.multiply(coefficients).toarray()
        ).ravel()

        positive = np.argsort(contributions)[::-1][:10]
        negative = np.argsort(contributions)[:10]

        print(f"\n--- Calibration fold {fold_index + 1} ---")

        print("\nFeatures pushing towards PHISHING:")

        for i in positive:
            if contributions[i] > 0:
                print(
                    f"{feature_names[i]}: "
                    f"{contributions[i]:.4f}"
                )

        print("\nFeatures pushing towards LEGITIMATE:")

        for i in negative:
            if contributions[i] < 0:
                print(
                    f"{feature_names[i]}: "
                    f"{contributions[i]:.4f}"
                )
