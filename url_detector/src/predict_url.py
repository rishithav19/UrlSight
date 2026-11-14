"""
Stage C Deep Hybrid Phishing URL Inference Pipeline.

Takes an arbitrary URL string as input, extracts 16 lexical and structural
features exactly matching the LegitPhish dataset definition, processes them
through the trained 15-epoch spatio-temporal extractors (CNN + GRU + FMRCNN),
selects the 101 BKA-optimized deep latent features, and classifies the URL
using the final RBF-SVM.

Security Notice:
- Pure offline lexical analysis. Does NOT visit, ping, or connect to the target URL.
- Predictions are probabilistic machine learning inferences and DO NOT guarantee safety.

Usage:
    python src/predict_url.py "http://example.com"
    python src/predict_url.py --url "http://malicious-login-verify.com"
    python src/predict_url.py --json "https://google.com"
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import torch

# Ensure local src directory is on sys.path
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from feature_extractor import extract_features_from_url, extract_features_vector, FEATURE_NAMES
from deep_hybrid_models import CNNExtractor, GRUExtractor, FMRCNNExtractor

class PhishingPredictor:
    """Encapsulates the frozen Stage C Deep Hybrid inference pipeline."""

    def __init__(self, project_dir=None, device='cpu'):
        if project_dir is None:
            project_dir = PROJECT_DIR
        self.project_dir = project_dir
        self.models_dir = os.path.join(project_dir, "models")
        self.device = device
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads all required scalers, neural checkpoints, feature masks, and final SVM."""
        required = {
            "deep_scaler": os.path.join(self.models_dir, "deep_models_scaler.joblib"),
            "cnn_weights": os.path.join(self.models_dir, "cnn_extractor.pth"),
            "gru_weights": os.path.join(self.models_dir, "gru_extractor.pth"),
            "fmrcnn_weights": os.path.join(self.models_dir, "fmrcnn_extractor.pth"),
            "selected_features": os.path.join(self.models_dir, "final_selected_deep_features.joblib"),
            "hybrid_scaler": os.path.join(self.models_dir, "final_hybrid_scaler.joblib"),
            "hybrid_svm": os.path.join(self.models_dir, "final_hybrid_svm.joblib")
        }

        # Check for missing artifacts
        missing = [name for name, path in required.items() if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                f"Missing required Stage C artifacts: {missing}. "
                "Ensure Stage C full training and downstream pipelines have completed."
            )

        # 1. Input Scaler fitted on real training data (16 features)
        self.input_scaler = joblib.load(required["deep_scaler"])

        # 2. Deep neural extractors
        self.cnn = CNNExtractor(in_features=16, latent_dim=512).to(self.device)
        self.cnn.load_state_dict(torch.load(required["cnn_weights"], map_location=self.device))
        self.cnn.eval()

        self.gru = GRUExtractor(in_features=16, hidden_dim=128, latent_dim=256).to(self.device)
        self.gru.load_state_dict(torch.load(required["gru_weights"], map_location=self.device))
        self.gru.eval()

        self.fmrcnn = FMRCNNExtractor(in_features=16, conv_filters=128, gru_units=128, latent_dim=256).to(self.device)
        self.fmrcnn.load_state_dict(torch.load(required["fmrcnn_weights"], map_location=self.device))
        self.fmrcnn.eval()

        # 3. Feature selection mask (101 BKA-selected indices out of 1024)
        feat_data = joblib.load(required["selected_features"])
        self.selected_indices = feat_data["original_1024_indices"]

        # 4. Downstream scaler & Final RBF-SVM
        self.hybrid_scaler = joblib.load(required["hybrid_scaler"])
        self.final_svm = joblib.load(required["hybrid_svm"])

    def validate_url(self, url: str) -> str:
        """Sanitizes and validates input URL string without visiting."""
        if not isinstance(url, str):
            raise ValueError(f"Input URL must be a string, got {type(url).__name__}")
        url = url.strip()
        if not url:
            raise ValueError("Input URL cannot be empty or purely whitespace.")
        if len(url) < 3:
            raise ValueError(f"Input URL '{url}' is too short to be a valid URL (min length 3).")
        if any(ord(c) < 32 or ord(c) == 127 for c in url):
            raise ValueError("Input URL contains illegal control characters or ASCII non-printables.")
        return url

    def predict(self, url: str) -> dict:
        """
        Executes end-to-end inference on a single URL string.
        Returns a dictionary containing predicted class, confidence, features, and notice.
        """
        clean_url = self.validate_url(url)

        # 1. Feature Extraction (16 features matching LegitPhish exactly)
        raw_features_dict = extract_features_from_url(clean_url)
        raw_features_vector = np.array([float(raw_features_dict[k]) for k in FEATURE_NAMES], dtype=np.float32).reshape(1, -1)

        # 2. Input Scaling (fitted on training data only)
        X_scaled = self.input_scaler.transform(raw_features_vector)

        # 3. Deep Feature Extraction
        # Shapes: CNN (1, 1, 16), GRU (1, 16, 1), FMRCNN (1, 1, 16)
        t_cnn = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)
        t_gru = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            f_cnn = self.cnn.extract_features(t_cnn).cpu().numpy()       # (1, 512)
            f_gru = self.gru.extract_features(t_gru).cpu().numpy()       # (1, 256)
            f_fmrcnn = self.fmrcnn.extract_features(t_cnn).cpu().numpy() # (1, 256)

        # 4. Fusion (1, 1024)
        F_fused = np.hstack([f_cnn, f_gru, f_fmrcnn])
        assert F_fused.shape == (1, 1024), f"Unexpected fused shape: {F_fused.shape}"

        # 5. Feature Selection (101 BKA-selected indices)
        X_selected = F_fused[:, self.selected_indices]
        assert X_selected.shape == (1, len(self.selected_indices)), f"Unexpected selected shape: {X_selected.shape}"

        # 6. Downstream Scaling & Final SVM Classification
        X_final = self.hybrid_scaler.transform(X_selected)
        pred_label = int(self.final_svm.predict(X_final)[0])
        decision_margin = float(self.final_svm.decision_function(X_final)[0])

        # Positive class convention: 0 = Phishing, 1 = Legitimate
        # decision_function is positive for class 1 and negative for class 0
        phishing_confidence_score = -decision_margin
        is_phishing = (pred_label == 0)

        label_name = "PHISHING" if is_phishing else "LEGITIMATE"

        return {
            "url": clean_url,
            "predicted_label": pred_label,
            "label_name": label_name,
            "is_phishing": is_phishing,
            "decision_margin": decision_margin,
            "phishing_confidence_score": phishing_confidence_score,
            "raw_features": raw_features_dict,
            "selected_features_count": len(self.selected_indices),
            "safety_warning": (
                "WARNING: Machine learning detections are assistive and probabilistic, not a guarantee "
                "of safety. A prediction of 'LEGITIMATE' does NOT guarantee the website is safe from "
                "malware, zero-day threats, or server compromise. Always verify domains independently."
            )
        }

def print_result_card(res: dict):
    """Prints a structured, readable terminal card with prediction and features."""
    border = "=" * 75
    divider = "-" * 75

    print("\n" + border)
    print("STAGE C DEEP HYBRID PHISHING DETECTION REPORT")
    print(border)
    print(f"URL: {res['url']}")
    
    if res['is_phishing']:
        badge = "[!] PREDICTED CLASS: PHISHING (Class 0)"
    else:
        badge = "[*] PREDICTED CLASS: LEGITIMATE (Class 1)"
    
    print(divider)
    print(badge)
    print(f"SVM Decision Margin:        {res['decision_margin']:+.4f}")
    print(f"Phishing Confidence Score: {res['phishing_confidence_score']:+.4f} (higher -> more phishing-like)")
    print(f"Latent Features Used:      {res['selected_features_count']} deep dimensions (from CNN 512 + GRU 256 + FMRCNN 256)")
    print(divider)
    print("Key Extracted URL Attributes:")
    rf = res['raw_features']
    print(f"  * Length: {rf['url_length']} chars | Entropy: {rf['url_entropy']:.4f} | Tokens: {rf['token_count']}")
    print(f"  * HTTPS: {'Yes' if rf['https_flag'] else 'No'} | Raw IP: {'Yes' if rf['has_ip_address'] else 'No'} | Domain Hyphen: {'Yes' if rf['has_hyphen_in_domain'] else 'No'}")
    print(f"  * Subdomains: {rf['subdomain_count']} | Query Params: {rf['query_param_count']} | Digits %: {rf['percentage_numeric_chars']:.2f}%")
    print(f"  * Popular TLD: {'Yes' if rf['tld_popularity'] else 'No'} | Suspicious Ext: {'Yes' if rf['suspicious_file_extension'] else 'No'}")
    print(divider)
    print("SAFETY NOTICE:")
    print(res['safety_warning'])
    print(border + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Predict whether a URL is Phishing (0) or Legitimate (1) using the trained Stage C Deep Hybrid Pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/predict_url.py "https://www.paypal.com"
  python src/predict_url.py "http://secure-login-account-update.tk/login.php"
  python src/predict_url.py --json "http://192.168.1.1/admin.exe"
        """
    )
    parser.add_argument("url", nargs="?", help="URL string to classify (e.g. 'https://example.com')")
    parser.add_argument("--url", dest="named_url", help="URL string to classify")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()
    target_url = args.url or args.named_url

    if not target_url:
        try:
            target_url = input("Enter a URL to analyze: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(0)

    try:
        predictor = PhishingPredictor()
        result = predictor.predict(target_url)
        if args.json:
            print(json.dumps(result, indent=4))
        else:
            print_result_card(result)
    except Exception as e:
        print(f"\n[ERROR]: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
