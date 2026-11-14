"""Check that all Stage C inference artifacts are present and loadable."""
from url_detector.src.predict_url import PhishingPredictor

if __name__ == "__main__":
    predictor = PhishingPredictor()
    print("Stage C predictor loaded successfully.")
    print("Model directory:", predictor.models_dir)
    print("Selected deep features:", len(predictor.selected_indices))
    print("Device:", predictor.device)
