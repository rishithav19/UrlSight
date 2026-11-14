
import joblib
import numpy as np
from pathlib import Path
from html_dom.features import extract_dom_features

models = {
    "V1": joblib.load("artifacts/best_html_dom_model.joblib"),
    "V2": joblib.load("artifacts_v2/best_html_dom_model_v2.joblib")
}

sites = [
    ("Wikipedia", "https://www.wikipedia.org"),
    ("Google", "https://www.google.com"),
    ("GitHub", "https://github.com")
]

for site_name, url in sites:

    path = Path(
        f"webscraper_output/{site_name.lower()}/page.html"
    )

    if not path.exists():
        print(f"\n{site_name}: HTML file not found")
        continue

    html = path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    dom_features, html_text = extract_dom_features(html, url)

    record = {
        "dom_features": dom_features,
        "html_text": html_text
    }

    print("\n" + "=" * 65)
    print(site_name)
    print("=" * 65)

    for model_name, model in models.items():

        transformer = model.named_steps["features"]
        classifier = model.named_steps["classifier"]

        transformed = transformer.transform([record])

        classes = list(classifier.classes_)
        phishing_index = classes.index(1)

        # Handle binary and multiclass coefficient formats.
        if len(classes) == 2:

            coefficients = classifier.coef_[0]

            if classes[1] != 1:
                coefficients = -coefficients

        else:
            coefficients = classifier.coef_[phishing_index]

        # DOM and text feature names
        dom_names = list(
            transformer.dom_vectorizer.get_feature_names_out()
        )

        text_names = [
            "TEXT:" + name
            for name in transformer.text_vectorizer.get_feature_names_out()
        ]

        feature_names = dom_names + text_names

        # Contribution = transformed feature value * coefficient
        contributions = np.asarray(
            transformed.multiply(coefficients).toarray()
        ).ravel()

        positive_indices = np.argsort(contributions)[::-1][:10]
        negative_indices = np.argsort(contributions)[:10]

        print(f"\n========== {model_name} ==========")

        probabilities = model.predict_proba([record])[0]
        prediction = int(model.predict([record])[0])

        print(
            "Prediction:",
            "PHISHING" if prediction == 1 else "LEGITIMATE"
        )

        print(
            f"Phishing probability: "
            f"{probabilities[phishing_index]:.4%}"
        )

        print("\nTop features pushing towards PHISHING:")

        for i in positive_indices:
            if contributions[i] > 0:
                print(
                    f"{feature_names[i]}: "
                    f"{contributions[i]:.4f}"
                )

        print("\nTop features pushing towards LEGITIMATE:")

        for i in negative_indices:
            if contributions[i] < 0:
                print(
                    f"{feature_names[i]}: "
                    f"{contributions[i]:.4f}"
                )
