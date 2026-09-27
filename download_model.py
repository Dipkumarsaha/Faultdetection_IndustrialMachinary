#!/usr/bin/env python3
"""
Model Weights Downloader & Checkpoint Generator.
Enables conference reviewers to either:
  1) Fetch pre-trained weights from remote repository mirrors (HuggingFace / Zenodo / GitHub Releases).
  2) Generate local calibrated checkpoint weights for instantaneous zero-setup offline evaluation.
"""

import os
import sys
import argparse
import urllib.request

# Default remote repository mirrors (placeholder for camera-ready release)
REMOTE_MIRRORS = {
    "dataset1_cnn_bigru_fedavg": "https://github.com/anonymous-research-artifact/pdm-fl-models/releases/download/v1.0.0/d1_cnn_bigru_fedavg.weights.h5",
    "dataset2_cnn_bigru_fedavg": "https://github.com/anonymous-research-artifact/pdm-fl-models/releases/download/v1.0.0/d2_cnn_bigru_fedavg.weights.h5",
}

def generate_local_calibrated_weights(dataset_id=1, model_name="CNN-BiGRU", output_path=None):
    """Generates a calibrated local weight checkpoint to allow immediate offline verification."""
    print(f"[*] Training calibrated reference weights locally on Dataset-{dataset_id} ({model_name})...")
    try:
        from evaluate import load_and_preprocess_dataset, build_model, train_federated, init_libraries, check_dependencies
        check_dependencies()
        init_libraries()
        from sklearn.model_selection import train_test_split
        from imblearn.over_sampling import SMOTE
    except ImportError as e:
        print("\n" + "=" * 80)
        print("ENVIRONMENT NOTICE: MISSING DEPENDENCY")
        print("=" * 80)
        print(f"Cannot import: {e.name if hasattr(e, 'name') else e}")
        print("Please install the required dependencies with:")
        print("    pip install imbalanced-learn scikit-learn tensorflow")
        print("or:")
        print("    pip install -r requirements.txt")
        print("=" * 80 + "\n")
        sys.exit(1)

    X_raw, y_raw, preprocessor = load_and_preprocess_dataset(dataset_id=dataset_id)
    X_tr_raw, _, y_train, _ = train_test_split(X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=42)
    X_train = preprocessor.fit_transform(X_tr_raw)
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = train_federated(
        X_train_res, y_train_res, model_name=model_name,
        input_dim=X_train.shape[1], algo='FedAvg',
        num_clients=3, rounds=2, local_epochs=2, alpha=0.5
    )
    if output_path is None:
        output_path = f"checkpoints/dataset{dataset_id}_{model_name.lower().replace('-', '_')}_fedavg.weights.h5"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    model.save_weights(output_path)
    print(f"[✓] Calibrated weights successfully generated and saved to: {output_path}")
    return output_path

def download_remote_weights(model_key, output_path):
    """Downloads pre-trained weights from remote artifact mirror."""
    if model_key not in REMOTE_MIRRORS:
        print(f"[!] Unknown model key '{model_key}'. Available: {list(REMOTE_MIRRORS.keys())}")
        sys.exit(1)

    url = REMOTE_MIRRORS[model_key]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"[*] Fetching pre-trained checkpoint from {url} ...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"[✓] Download complete: {output_path}")
    except Exception as e:
        print(f"[!] Remote download failed ({e}). Falling back to local offline calibration.")
        d_id = 1 if "dataset1" in model_key else 2
        generate_local_calibrated_weights(dataset_id=d_id, output_path=output_path)

def main():
    parser = argparse.ArgumentParser(description="Download or generate pre-trained model weights for offline evaluation.")
    parser.add_argument("--model", type=str, default="CNN-BiGRU", help="Model architecture")
    parser.add_argument("--dataset", type=int, default=1, choices=[1, 2], help="Dataset ID")
    parser.add_argument("--remote", action="store_true", help="Attempt remote download from artifact mirror")
    parser.add_argument("--output", type=str, default=None, help="Custom target checkpoint path")
    args = parser.parse_args()

    default_path = args.output or f"checkpoints/dataset{args.dataset}_{args.model.lower().replace('-', '_')}_fedavg.weights.h5"

    if args.remote:
        key = f"dataset{args.dataset}_{args.model.lower().replace('-', '_')}_fedavg"
        download_remote_weights(key, default_path)
    else:
        generate_local_calibrated_weights(dataset_id=args.dataset, model_name=args.model, output_path=default_path)

if __name__ == "__main__":
    main()
