"""Smoke-test the integrated Stage C URL detector."""
from url_detector.src.predict_url import PhishingPredictor


def main():
    predictor = PhishingPredictor()
    test_urls = [
        "https://www.google.com",
        "https://accounts.google.com/signin",
        "http://192.168.1.10/login.exe",
    ]

    for url in test_urls:
        result = predictor.predict(url)
        print(f"\nURL: {url}")
        print("Prediction:", result["label_name"])
        print("Native class:", result["predicted_label"], "(0=phishing, 1=legitimate)")
        print("Decision margin:", round(result["decision_margin"], 4))
        print("Features:", len(result["raw_features"]))


if __name__ == "__main__":
    main()
