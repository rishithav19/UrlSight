"""
Stage C Component: Deep Spatio-Temporal Feature Extractors (CNN + GRU + FMRCNN)
Phishing URL Detection Pipeline.

Architectures Implemented:
1. CNN Feature Extractor:
   - 1D Convolutional blocks capturing local n-gram patterns across adjacent URL attributes.
   - Output representation: 512-dimensional continuous latent vector.
   - Auxiliary head: Binary classification for end-to-end representation learning.
2. GRU Feature Extractor:
   - Recurrent Gated Recurrent Units capturing sequential transitions across URL features.
   - Output representation: 256-dimensional continuous latent vector.
3. FMRCNN Feature Extractor (Spatiotemporal Cascade Adaptation):
   - Cascaded 1D Conv blocks followed by a temporal GRU layer and dense projection.
   - Output representation: 256-dimensional continuous latent vector.
4. Deep Feature Fusion:
   - Concatenates CNN (512) + GRU (256) + FMRCNN (256) -> 1,024 fused deep latent features.

Usage:
    python src/deep_hybrid_models.py --smoke-test
    python src/deep_hybrid_models.py --full-train
"""

import os
import sys
import copy
import json
import time
import argparse
import joblib
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ====================================================================
# 1. CNN Feature Extractor (512-dim output)
# ====================================================================
class CNNExtractor(nn.Module):
    """
    1D CNN for spatial feature extraction across ordered URL features.
    Specification: Output dimension = 512.
    """
    def __init__(self, in_features=16, latent_dim=512):
        super(CNNExtractor, self).__init__()
        # Conv block 1: in_channels=1, out_channels=128, kernel_size=2, padding=1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)

        # Conv block 2: in_channels=128, out_channels=256, kernel_size=2, padding=1
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)

        # Output length after two padding=1 convolutions on length 16 is 18
        conv_out_len = in_features + 2  # 18
        self.flatten_dim = 256 * conv_out_len

        # Dense projection to 512 dimensions
        self.fc_latent = nn.Linear(self.flatten_dim, latent_dim)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.4)

        # Auxiliary classifier head for supervised training
        self.classifier = nn.Linear(latent_dim, 1)

    def extract_features(self, x):
        # x shape: (batch_size, 1, 16)
        h = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        h = self.drop2(self.relu2(self.bn2(self.conv2(h))))
        h_flat = h.view(h.size(0), -1)
        latent = self.drop3(self.relu3(self.fc_latent(h_flat)))
        return latent

    def forward(self, x):
        latent = self.extract_features(x)
        logits = self.classifier(latent)
        return logits, latent


# ====================================================================
# 2. GRU Feature Extractor (256-dim output)
# ====================================================================
class GRUExtractor(nn.Module):
    """
    Recurrent GRU for sequential/temporal dependency extraction across URL features.
    Specification: Output dimension = 256.
    """
    def __init__(self, in_features=16, hidden_dim=128, latent_dim=256):
        super(GRUExtractor, self).__init__()
        # Input to GRU: (batch_size, seq_len=16, input_size=1)
        self.gru1 = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True, num_layers=1)
        self.gru2 = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, num_layers=1)
        self.drop = nn.Dropout(0.3)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(latent_dim, 1)

    def extract_features(self, x):
        # x shape: (batch_size, 16, 1)
        out1, _ = self.gru1(x)
        out2, h_n = self.gru2(out1)
        # Use last time step output: out2[:, -1, :]
        last_step = out2[:, -1, :]
        latent = self.relu(self.fc_latent(self.drop(last_step)))
        return latent

    def forward(self, x):
        latent = self.extract_features(x)
        logits = self.classifier(latent)
        return logits, latent


# ====================================================================
# 3. FMRCNN Feature Extractor (256-dim output)
# ====================================================================
class FMRCNNExtractor(nn.Module):
    """
    Adapted FMRCNN (Fast Mask Recurrent CNN) for 1D tabular sequence features.
    Cascades 1D spatial convolutions with temporal GRU recurrence and dense projection.
    Specification: Output dimension = 256.
    """
    def __init__(self, in_features=16, conv_filters=128, gru_units=128, latent_dim=256):
        super(FMRCNNExtractor, self).__init__()
        # Spatial feature stage (Conv1D)
        self.conv1 = nn.Conv1d(1, conv_filters, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(conv_filters, conv_filters * 2, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm1d(conv_filters * 2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)

        # Recurrent temporal stage (GRU over convolutional channels)
        # Length after 2 convs on 16 is 18. Channels = 256.
        # Transpose (B, C=256, L=18) -> (B, L=18, C=256)
        self.gru = nn.GRU(input_size=conv_filters * 2, hidden_size=gru_units, batch_first=True)
        self.drop_gru = nn.Dropout(0.3)

        # Dense projection to 256 dimensions
        self.fc_latent = nn.Linear(gru_units, latent_dim)
        self.relu_fc = nn.ReLU()
        self.classifier = nn.Linear(latent_dim, 1)

    def extract_features(self, x):
        # x shape: (batch_size, 1, 16)
        h = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        h = self.drop2(self.relu2(self.bn2(self.conv2(h))))
        # Permute for GRU: (B, C, L) -> (B, L, C)
        h_perm = h.permute(0, 2, 1)
        out_gru, _ = self.gru(h_perm)
        last_step = self.drop_gru(out_gru[:, -1, :])
        latent = self.relu_fc(self.fc_latent(last_step))
        return latent

    def forward(self, x):
        latent = self.extract_features(x)
        logits = self.classifier(latent)
        return logits, latent


# ====================================================================
# Data Preparation & Pipeline
# ====================================================================
def load_and_prepare_data(smoke_test=False, project_dir=None):
    if project_dir is None:
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    train_path = os.path.join(project_dir, "legitphish_train.csv")
    val_path = os.path.join(project_dir, "legitphish_validation.csv")
    synth_path = os.path.join(project_dir, "models", "synthetic_samples_class1.joblib")

    print("[1/5] Loading training data and DPCA synthetic samples...")
    loaded = False
    if pd is not None:
        try:
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            feature_cols = [c for c in train_df.columns if c not in ['URL', 'ClassLabel']]
            X_train_real = train_df[feature_cols].values.astype(np.float32)
            y_train_real = train_df['ClassLabel'].values.astype(int)
            X_val = val_df[feature_cols].values.astype(np.float32)
            y_val = val_df['ClassLabel'].values.astype(int)
            loaded = True
        except Exception:
            loaded = False

    if not loaded:
        def _read_csv_table(path):
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                cols = [c for c in header if c not in ['URL', 'ClassLabel']]
                col_idx = [header.index(c) for c in cols]
                lbl_idx = header.index('ClassLabel')
                rows = list(reader)
                X = np.array([[float(r[i]) for i in col_idx] for r in rows], dtype=np.float32)
                y = np.array([int(r[lbl_idx]) for r in rows], dtype=int)
            return X, y, cols

        X_train_real, y_train_real, feature_cols = _read_csv_table(train_path)
        X_val, y_val, _ = _read_csv_table(val_path)

    # Load synthetic Class 1 samples generated by RaLSGAN
    if os.path.exists(synth_path):
        synth_data = joblib.load(synth_path)
        X_synth = synth_data['synthetic_features']
        y_synth = np.ones(len(X_synth), dtype=int)
        print(f"Loaded {len(X_synth)} RaLSGAN synthetic Class 1 samples.")
    else:
        print("Warning: Synthetic samples file not found, using real training set only.")
        X_synth = np.empty((0, len(feature_cols)))
        y_synth = np.empty((0,), dtype=int)

    # Fit StandardScaler STRICTLY on real training data only
    scaler = StandardScaler()
    X_train_real_scaled = scaler.fit_transform(X_train_real)
    X_synth_scaled = scaler.transform(X_synth) if len(X_synth) > 0 else np.empty((0, len(feature_cols)))
    
    # Combine scaled real and scaled synthetic features strictly for training
    X_train_scaled = np.vstack([X_train_real_scaled, X_synth_scaled])
    y_train_augmented = np.concatenate([y_train_real, y_synth])

    if smoke_test:
        print("Running in SMOKE-TEST mode (1,000 train samples, 500 val samples)...")
        # Stratified slice for smoke test
        idx_0 = np.where(y_train_augmented == 0)[0][:500]
        idx_1 = np.where(y_train_augmented == 1)[0][:500]
        smoke_idx = np.concatenate([idx_0, idx_1])
        X_train_scaled = X_train_scaled[smoke_idx]
        y_train_augmented = y_train_augmented[smoke_idx]
        X_val_scaled = scaler.transform(X_val[:500])
        y_val = y_val[:500]
    else:
        # Transform validation data using the real-train fitted scaler
        X_val_scaled = scaler.transform(X_val)

    # Assert no NaNs or Infs
    assert not np.isnan(X_train_scaled).any(), "NaN found in X_train_scaled!"
    assert not np.isinf(X_train_scaled).any(), "Inf found in X_train_scaled!"
    assert not np.isnan(X_val_scaled).any(), "NaN found in X_val_scaled!"
    assert not np.isinf(X_val_scaled).any(), "Inf found in X_val_scaled!"

    print(f"Augmented Train Shape: {X_train_scaled.shape} (Class 0: {np.sum(y_train_augmented==0)}, Class 1: {np.sum(y_train_augmented==1)})")
    print(f"Validation Shape:      {X_val_scaled.shape} (Class 0: {np.sum(y_val==0)}, Class 1: {np.sum(y_val==1)})")

    return X_train_scaled, y_train_augmented, X_val_scaled, y_val, feature_cols, scaler


def train_single_model(model, model_name, X_tr, y_tr, X_va, y_va, epochs=10, batch_size=128, lr=0.001, device='cpu'):
    print(f"\n--- Training {model_name} ({epochs} epochs, lr={lr}) ---")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Tensor shapes for CNN/FMRCNN: (B, 1, 16), for GRU: (B, 16, 1)
    if "GRU" in model_name:
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(-1)  # (B, 16, 1)
        X_va_t = torch.tensor(X_va, dtype=torch.float32).unsqueeze(-1)
    else:
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1)   # (B, 1, 16)
        X_va_t = torch.tensor(X_va, dtype=torch.float32).unsqueeze(1)

    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(-1)
    y_va_t = torch.tensor(y_va, dtype=torch.float32).unsqueeze(-1)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    history = []

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0

    start_time = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, latent = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        mean_loss = float(np.mean(losses))

        # Validation check (Used strictly for model selection via checkpoint tracking)
        model.eval()
        with torch.no_grad():
            val_logits, val_latent = model(X_va_t.to(device))
            val_loss = criterion(val_logits, y_va_t.to(device)).item()
            val_preds = (torch.sigmoid(val_logits) >= 0.5).cpu().numpy().flatten()
            val_acc = float(np.mean(val_preds == y_va))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep

        history.append({
            'epoch': ep,
            'train_loss': mean_loss,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        if ep % max(1, epochs // 5) == 0 or ep == epochs:
            print(f"Epoch {ep:2d}/{epochs:2d} | Train Loss: {mean_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4%}")

    elapsed = time.time() - start_time
    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[Model Selection] Restored best {model_name} checkpoint from epoch {best_epoch} (val_loss: {best_val_loss:.4f})")

    print(f"{model_name} training finished in {elapsed:.2f}s.")
    return model, history, best_epoch, best_val_loss


def extract_deep_features(model, model_name, X_data, batch_size=256, device='cpu'):
    model.eval()
    if "GRU" in model_name:
        X_t = torch.tensor(X_data, dtype=torch.float32).unsqueeze(-1)
    else:
        X_t = torch.tensor(X_data, dtype=torch.float32).unsqueeze(1)

    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
    features = []
    with torch.no_grad():
        for batch in loader:
            bx = batch[0].to(device)
            latent = model.extract_features(bx)
            features.append(latent.cpu().numpy())

    features = np.vstack(features)
    assert not np.isnan(features).any(), f"NaN found in extracted features from {model_name}!"
    assert not np.isinf(features).any(), f"Inf found in extracted features from {model_name}!"
    return features


def run_pipeline(smoke_test=False):
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(project_dir, "results")
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    prefix = "smoke_" if smoke_test else ""
    print("=" * 70)
    print(f"STAGE C: DEEP FEATURE EXTRACTION PIPELINE (CNN + GRU + FMRCNN)")
    print(f"Mode: {'SMOKE TEST' if smoke_test else 'FULL TRAINING'}")
    print("=" * 70)

    # 1. Load data
    X_train, y_train, X_val, y_val, feature_cols, scaler = load_and_prepare_data(
        smoke_test=smoke_test, project_dir=project_dir
    )

    epochs = 2 if smoke_test else 15
    batch_size = 64 if smoke_test else 128
    device = 'cpu'

    # 2. Instantiate models
    cnn = CNNExtractor(in_features=16, latent_dim=512)
    gru = GRUExtractor(in_features=16, hidden_dim=128, latent_dim=256)
    fmrcnn = FMRCNNExtractor(in_features=16, conv_filters=128, gru_units=128, latent_dim=256)

    # Verify initial tensor shapes
    dummy_cnn_in = torch.randn(4, 1, 16)
    dummy_gru_in = torch.randn(4, 16, 1)
    with torch.no_grad():
        out_cnn = cnn.extract_features(dummy_cnn_in)
        out_gru = gru.extract_features(dummy_gru_in)
        out_fmrcnn = fmrcnn.extract_features(dummy_cnn_in)
    
    print("\n[Shape Verification on Dummy Batch (B=4)]:")
    print(f"  CNN feature shape:    {list(out_cnn.shape)} (Expected: [4, 512])")
    print(f"  GRU feature shape:    {list(out_gru.shape)} (Expected: [4, 256])")
    print(f"  FMRCNN feature shape: {list(out_fmrcnn.shape)} (Expected: [4, 256])")
    assert out_cnn.shape == (4, 512), f"CNN shape mismatch: {out_cnn.shape}"
    assert out_gru.shape == (4, 256), f"GRU shape mismatch: {out_gru.shape}"
    assert out_fmrcnn.shape == (4, 256), f"FMRCNN shape mismatch: {out_fmrcnn.shape}"

    # 3. Train models
    cnn, cnn_hist, cnn_best_ep, cnn_best_val_loss = train_single_model(
        cnn, "CNN_Extractor", X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, lr=0.001, device=device
    )
    gru, gru_hist, gru_best_ep, gru_best_val_loss = train_single_model(
        gru, "GRU_Extractor", X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, lr=0.001, device=device
    )
    fmrcnn, fmrcnn_hist, fmrcnn_best_ep, fmrcnn_best_val_loss = train_single_model(
        fmrcnn, "FMRCNN_Extractor", X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, lr=0.001, device=device
    )

    # 4. Extract latent features
    print("\n[4/5] Extracting deep latent features using best-checkpoint models...")
    F_train_cnn = extract_deep_features(cnn, "CNN", X_train, device=device)
    F_val_cnn = extract_deep_features(cnn, "CNN", X_val, device=device)

    F_train_gru = extract_deep_features(gru, "GRU", X_train, device=device)
    F_val_gru = extract_deep_features(gru, "GRU", X_val, device=device)

    F_train_fmrcnn = extract_deep_features(fmrcnn, "FMRCNN", X_train, device=device)
    F_val_fmrcnn = extract_deep_features(fmrcnn, "FMRCNN", X_val, device=device)

    # 5. Fuse features
    F_train_fused = np.hstack([F_train_cnn, F_train_gru, F_train_fmrcnn])
    F_val_fused = np.hstack([F_val_cnn, F_val_gru, F_val_fmrcnn])

    print(f"\n[Feature Fusion Complete]:")
    print(f"  F_train_fused shape: {F_train_fused.shape} (Expected: [N_train, 1024])")
    print(f"  F_val_fused shape:   {F_val_fused.shape} (Expected: [N_val, 1024])")
    assert F_train_fused.shape[1] == 1024, "Fused feature dimension must be 1024!"
    assert F_val_fused.shape[1] == 1024, "Fused feature dimension must be 1024!"

    # Save checkpoints & artifacts
    torch.save(cnn.state_dict(), os.path.join(models_dir, f"{prefix}cnn_extractor.pth"))
    torch.save(gru.state_dict(), os.path.join(models_dir, f"{prefix}gru_extractor.pth"))
    torch.save(fmrcnn.state_dict(), os.path.join(models_dir, f"{prefix}fmrcnn_extractor.pth"))
    joblib.dump(scaler, os.path.join(models_dir, f"{prefix}deep_models_scaler.joblib"))

    # Save feature tensors
    np.save(os.path.join(models_dir, f"{prefix}F_train_fused.npy"), F_train_fused)
    np.save(os.path.join(models_dir, f"{prefix}F_val_fused.npy"), F_val_fused)
    np.save(os.path.join(models_dir, f"{prefix}y_train.npy"), y_train)
    np.save(os.path.join(models_dir, f"{prefix}y_val.npy"), y_val)

    # Save detailed training history
    history_report = {
        "CNN": cnn_hist,
        "GRU": gru_hist,
        "FMRCNN": fmrcnn_hist
    }
    with open(os.path.join(results_dir, f"{prefix}deep_models_training_history.json"), "w") as f:
        json.dump(history_report, f, indent=4)

    pipeline_report = {
        "smoke_test": smoke_test,
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val),
        "input_features": 16,
        "cnn_latent_dim": 512,
        "gru_latent_dim": 256,
        "fmrcnn_latent_dim": 256,
        "fused_feature_dim": 1024,
        "model_selection_best_epochs": {
            "CNN": cnn_best_ep,
            "GRU": gru_best_ep,
            "FMRCNN": fmrcnn_best_ep
        },
        "model_selection_best_val_loss": {
            "CNN": float(cnn_best_val_loss),
            "GRU": float(gru_best_val_loss),
            "FMRCNN": float(fmrcnn_best_val_loss)
        },
        "cnn_final_train_loss": float(cnn_hist[-1]['train_loss']),
        "cnn_final_val_acc": float(cnn_hist[-1]['val_acc']),
        "gru_final_train_loss": float(gru_hist[-1]['train_loss']),
        "gru_final_val_acc": float(gru_hist[-1]['val_acc']),
        "fmrcnn_final_train_loss": float(fmrcnn_hist[-1]['train_loss']),
        "fmrcnn_final_val_acc": float(fmrcnn_hist[-1]['val_acc']),
        "nan_inf_check_passed": True,
        "data_usage_notice": (
            "Validation data was used strictly for model selection (checkpoint selection "
            "via minimum validation loss) and validation benchmarking. Validation metrics "
            "must NOT be described as unbiased test performance. Test set remains strictly quarantined."
        )
    }

    report_path = os.path.join(results_dir, f"{prefix}deep_models_report.json")
    with open(report_path, "w") as f:
        json.dump(pipeline_report, f, indent=4)

    print(f"\nPipeline step completed successfully. Saved summary to: {report_path}")
    return pipeline_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Hybrid Feature Extraction")
    parser.add_argument("--smoke-test", action="store_true", help="Run small smoke test")
    parser.add_argument("--full-train", action="store_true", help="Run full training")
    args = parser.parse_args()

    run_pipeline(smoke_test=args.smoke_test or not args.full_train)
