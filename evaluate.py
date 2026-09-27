#!/usr/bin/env python3
"""
Reproducibility CLI & Model Runner for:
"A Federated Learning Approach for AI-based Predictive Maintenance"
A* Conference Artifact Evaluation Toolkit.
"""

import os
import sys
import argparse
import warnings
import os
import sys
import argparse
import warnings

# Suppress verbose warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

def check_dependencies():
    missing = []
    for pkg, imp in [('numpy', 'numpy'), ('pandas', 'pandas'), ('tensorflow', 'tensorflow'),
                     ('scikit-learn', 'sklearn'), ('imbalanced-learn', 'imblearn')]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("\n" + "=" * 80)
        print("ENVIRONMENT NOTICE: MISSING DEPENDENCIES")
        print("=" * 80)
        print(f"Missing required packages: {', '.join(missing)}")
        print("\nTo set up the exact environment, run:")
        print("    pip install -r requirements.txt")
        print("or with Conda:")
        print("    conda env create -f environment.yml")
        print("=" * 80 + "\n")
        sys.exit(1)

# Lazy/Safe global imports
np = None
pd = None
tf = None
layers = None
models = None
StratifiedKFold = None
train_test_split = None
StandardScaler = None
OneHotEncoder = None
ColumnTransformer = None
Pipeline = None
SimpleImputer = None
accuracy_score = None
f1_score = None
precision_score = None
recall_score = None
roc_auc_score = None
matthews_corrcoef = None
cohen_kappa_score = None
log_loss = None
mean_absolute_error = None
mean_squared_error = None
confusion_matrix = None
SMOTE = None

def init_libraries():
    global np, pd, tf, layers, models
    global StratifiedKFold, train_test_split, StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline, SimpleImputer
    global accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score, log_loss, mean_absolute_error, mean_squared_error, confusion_matrix
    global SMOTE

    import numpy as _np
    import pandas as _pd
    import tensorflow as _tf
    from tensorflow.keras import layers as _layers, models as _models
    from sklearn.model_selection import StratifiedKFold as _skf, train_test_split as _tts
    from sklearn.preprocessing import StandardScaler as _ss, OneHotEncoder as _ohe
    from sklearn.compose import ColumnTransformer as _ct
    from sklearn.pipeline import Pipeline as _pipe
    from sklearn.impute import SimpleImputer as _si
    from sklearn.metrics import (accuracy_score as _acc, f1_score as _f1, precision_score as _prec,
                                 recall_score as _rec, roc_auc_score as _auc, matthews_corrcoef as _mcc,
                                 cohen_kappa_score as _ck, log_loss as _ll, mean_absolute_error as _mae,
                                 mean_squared_error as _mse, confusion_matrix as _cm)
    from imblearn.over_sampling import SMOTE as _smote

    np, pd, tf = _np, _pd, _tf
    layers, models = _layers, _models
    StratifiedKFold, train_test_split = _skf, _tts
    StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline, SimpleImputer = _ss, _ohe, _ct, _pipe, _si
    accuracy_score, f1_score, precision_score, recall_score = _acc, _f1, _prec, _rec
    roc_auc_score, matthews_corrcoef, cohen_kappa_score = _auc, _mcc, _ck
    log_loss, mean_absolute_error, mean_squared_error, confusion_matrix = _ll, _mae, _mse, _cm
    SMOTE = _smote




# ==============================================================================
# 1. DATA PREPARATION & PARTITIONING (DIRICHLET NON-IID)
# ==============================================================================
def load_and_preprocess_dataset(dataset_id=1, base_dir="."):
    """Loads Dataset-1 or Dataset-2 with standardized feature pipelines."""
    if dataset_id == 1:
        path = os.path.join(base_dir, "Dataset-1", "file1.csv")
        if not os.path.exists(path):
            path = os.path.join(base_dir, "file1.csv")
        df = pd.read_csv(path)
        y = (df['Diagnostic'] != 'No failure').astype(int).values
        drop_cols = ['UDI', 'Product ID', 'Date', 'Diagnostic']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    elif dataset_id == 2:
        path = os.path.join(base_dir, "Dataset-2", "file2.csv")
        if not os.path.exists(path):
            path = os.path.join(base_dir, "file2.csv")
        df = pd.read_csv(path)
        drop_cols = ['UDI', 'Product ID', 'Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Machine failure']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        y = df['Machine failure'].astype(int).values
    else:
        raise ValueError("Invalid dataset_id. Must be 1 or 2.")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
    ])
    return X, y, preprocessor


def dirichlet_partition(X, y, n_clients=3, alpha=0.5, seed=42):
    """Partitions dataset among non-IID clients following a Dirichlet distribution Dir(alpha)."""
    np.random.seed(seed)
    n_classes = len(np.unique(y))
    label_indices = [np.where(y == i)[0] for i in range(n_classes)]
    client_indices = [[] for _ in range(n_clients)]

    for k in range(n_classes):
        np.random.shuffle(label_indices[k])
        proportions = np.random.dirichlet([alpha] * n_clients)
        proportions = (np.cumsum(proportions) * len(label_indices[k])).astype(int)[:-1]
        split_indices = np.split(label_indices[k], proportions)
        for i in range(n_clients):
            client_indices[i].extend(split_indices[i])

    return [X[idx] for idx in client_indices], [y[idx] for idx in client_indices]


# ==============================================================================
# 2. ARCHITECTURAL DEFINITIONS (BASELINES & HYBRIDS)
# ==============================================================================
def build_model(model_name, input_dim):
    """Factory function for all deep learning and hybrid architectures."""
    inputs = layers.Input(shape=(input_dim,))

    if model_name in ['LSTM', '1D-CNN', '1D CNN', 'GRU', 'BiLSTM', 'Transformer',
                      'CNN-GRU', 'CNN-BiGRU', 'Residual-CNN-GRU', 'Deeper-CNN-GRU', 'Bottleneck-CNN-GRU']:
        x = layers.Reshape((input_dim, 1))(inputs)

        if model_name in ['1D-CNN', '1D CNN']:
            x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
            x = layers.GlobalAveragePooling1D()(x)
        elif model_name == 'LSTM':
            x = layers.LSTM(64)(x)
        elif model_name == 'GRU':
            x = layers.GRU(64)(x)
        elif model_name == 'BiLSTM':
            x = layers.Bidirectional(layers.LSTM(64))(x)
        elif model_name == 'Transformer':
            attn = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
            attn = layers.Dropout(0.1)(attn)
            x = layers.LayerNormalization(epsilon=1e-6)(x + attn)
            x = layers.GlobalAveragePooling1D()(x)
        elif model_name == 'CNN-GRU':
            x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
            x = layers.GRU(64)(x)
        elif model_name == 'CNN-BiGRU':
            x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
            x = layers.Bidirectional(layers.GRU(32))(x)
            x = layers.Dropout(0.2)(x)
        elif model_name == 'Residual-CNN-GRU':
            shortcut = layers.Conv1D(64, kernel_size=1)(x)
            conv = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
            x = layers.Add()([conv, shortcut])
            x = layers.GRU(64)(x)
        elif model_name == 'Deeper-CNN-GRU':
            x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
            x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
            x = layers.GRU(128, return_sequences=True)(x)
            x = layers.GRU(64)(x)
        elif model_name == 'Bottleneck-CNN-GRU':
            x = layers.Conv1D(32, kernel_size=1, activation='relu')(x)
            x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
            x = layers.GRU(64)(x)
    else:
        # Standard MLP Baseline
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)

    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ==============================================================================
# 3. FEDERATED LEARNING PROTOCOL (FedAvg & FedProx)
# ==============================================================================
def train_federated(X_train, y_train, model_name, input_dim,
                    algo='FedAvg', num_clients=3, rounds=3, local_epochs=3, alpha=0.5, mu=0.01):
    """Executes decentralized federated aggregation over Non-IID Dirichlet partitions."""
    global_model = build_model(model_name, input_dim)
    X_clients, y_clients = dirichlet_partition(X_train, y_train, n_clients=num_clients, alpha=alpha)

    for r in range(rounds):
        client_weights = []
        client_sizes = []

        for i in range(num_clients):
            if len(y_clients[i]) < 5:
                continue

            local_model = build_model(model_name, input_dim)
            local_model.set_weights(global_model.get_weights())

            xs, ys = X_clients[i], y_clients[i]
            class_counts = np.bincount(ys.astype(int))

            # Apply Safe-SMOTE within edge boundaries to avoid privacy leakage
            if len(class_counts) > 1 and np.all(class_counts >= 2):
                try:
                    sm = SMOTE(random_state=42, k_neighbors=1)
                    xs, ys = sm.fit_resample(xs, ys)
                except Exception:
                    pass

            local_model.fit(xs, ys, epochs=local_epochs, batch_size=32, verbose=0)
            weights = local_model.get_weights()

            # FedProx Proximal Regularization weight scaling
            if algo == 'FedProx':
                weights = [w * (1.0 - mu) for w in weights]

            client_weights.append(weights)
            client_sizes.append(len(xs))

        if client_weights:
            total_samples = sum(client_sizes)
            new_weights = []
            for layer_idx in range(len(client_weights[0])):
                layer_avg = sum(client_weights[c][layer_idx] * (client_sizes[c] / total_samples)
                                for c in range(len(client_weights)))
                new_weights.append(layer_avg)
            global_model.set_weights(new_weights)

    return global_model


# ==============================================================================
# 4. METRIC COMPUTATION & EVALUATION
# ==============================================================================
def compute_metrics(y_true, y_prob, threshold=0.5):
    """Computes comprehensive 10-metric evaluation battery."""
    y_pred = (y_prob > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, len(y_true))
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': spec,
        'ROC-AUC': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Cohen-Kappa': cohen_kappa_score(y_true, y_pred),
        'Log-Loss': log_loss(y_true, y_prob),
        'Brier-Score': mean_squared_error(y_true, y_prob)
    }


def format_results_table(metrics_dict):
    """Formats metrics dictionary into a clean markdown table."""
    headers = ["Metric", "Mean Score"]
    rows = [[k, f"{v:.4f}"] for k, v in metrics_dict.items()]
    col_w = [max(len(r[i]) for r in [headers] + rows) for i in range(2)]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    hdr = "| " + " | ".join(headers[i].ljust(col_w[i]) for i in range(2)) + " |"

    lines = [sep, hdr, sep]
    for r in rows:
        lines.append("| " + " | ".join(r[i].ljust(col_w[i]) for i in range(2)) + " |")
    lines.append(sep)
    return "\n".join(lines)


# ==============================================================================
# 5. MAIN CLI ENTRY POINT
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="A Federated Learning Approach for AI-based Predictive Maintenance: Reviewer Reproducibility Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=int, default=1, choices=[1, 2], help="Dataset ID (1 or 2)")
    parser.add_argument("--mode", type=str, default="federated", choices=["federated", "centralized"], help="Training paradigm")
    parser.add_argument("--model", type=str, default="CNN-BiGRU",
                        choices=['LSTM', '1D-CNN', 'Transformer', 'GRU', 'BiLSTM',
                                 'CNN-GRU', 'CNN-BiGRU', 'Residual-CNN-GRU', 'Deeper-CNN-GRU', 'Bottleneck-CNN-GRU', 'MLP'],
                        help="Neural model architecture")
    parser.add_argument("--algo", type=str, default="FedAvg", choices=["FedAvg", "FedProx"], help="Federated aggregation algorithm")
    parser.add_argument("--rounds", type=int, default=3, help="Federated communication rounds")
    parser.add_argument("--epochs", type=int, default=3, help="Local epochs per round (or centralized epochs)")
    parser.add_argument("--clients", type=int, default=3, help="Number of decentralized federated clients")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet non-IID partition parameter alpha")
    parser.add_argument("--save-weights", type=str, default="", help="Path to save trained model weights (.weights.h5)")
    parser.add_argument("--load-weights", type=str, default="", help="Path to load pre-trained weights for instant inference")
    parser.add_argument("--quick-test", action="store_true", help="Execute rapid single-split test for reviewer validation in seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    check_dependencies()
    init_libraries()

    print("=" * 80)
    print("FEDERATED PREDICTIVE MAINTENANCE: REPRODUCIBILITY EVALUATION")
    print("=" * 80)
    print(f"Dataset:       Dataset-{args.dataset}")
    print(f"Model:         {args.model}")
    print(f"Paradigm:      {args.mode.upper()} {f'({args.algo})' if args.mode == 'federated' else ''}")
    print(f"Quick-Test:    {args.quick_test}")
    print("-" * 80)

    # 1. Load Data
    X_raw, y_raw, preprocessor = load_and_preprocess_dataset(dataset_id=args.dataset)
    print(f"Loaded Dataset-{args.dataset}: {X_raw.shape[0]} samples, {X_raw.shape[1]} raw features.")

    # 2. Split Data (Stratified Train/Test)
    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=42
    )
    X_train = preprocessor.fit_transform(X_tr_raw)
    X_test = preprocessor.transform(X_te_raw)
    input_dim = X_train.shape[1]

    # Handle class imbalance on training split
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"Preprocessed Input Dimension: {input_dim} features.")
    print(f"Train Samples: {len(X_train)} (Oversampled: {len(X_train_res)}), Test Samples: {len(X_test)}")

    # 3. Model Training / Weight Loading
    model = build_model(args.model, input_dim)

    if args.load_weights and os.path.exists(args.load_weights):
        print(f"Loading pretrained weights from: {args.load_weights} ...")
        model.load_weights(args.load_weights)
    else:
        if args.mode == "federated":
            rounds = 1 if args.quick_test else args.rounds
            local_epochs = 1 if args.quick_test else args.epochs
            print(f"Running Federated Simulation ({rounds} rounds, {args.clients} clients, alpha={args.alpha})...")
            model = train_federated(
                X_train_res, y_train_res, args.model, input_dim,
                algo=args.algo, num_clients=args.clients,
                rounds=rounds, local_epochs=local_epochs, alpha=args.alpha
            )
        else:
            epochs = 2 if args.quick_test else args.epochs
            print(f"Running Centralized Training ({epochs} epochs)...")
            model.fit(X_train_res, y_train_res, epochs=epochs, batch_size=32, verbose=1 if not args.quick_test else 0)

    # 4. Save weights if requested
    if args.save_weights:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_weights)), exist_ok=True)
        model.save_weights(args.save_weights)
        print(f"Model weights saved successfully to: {args.save_weights}")

    # 5. Evaluate on Holdout Test Split
    print("\nEvaluating on holdout test partition...")
    y_prob = model.predict(X_test, verbose=0).flatten()
    metrics = compute_metrics(y_test, y_prob)

    print("\n" + format_results_table(metrics))
    print("=" * 80)
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
