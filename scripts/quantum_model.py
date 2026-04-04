
import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import time
import json
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------
# Training note:
#   Full dataset: 12,811 samples. One quantum circuit forward pass on
#   default.qubit takes ~0.01 s/sample on CPU. 50 epochs × 10K samples ≈ 89 min.
#   A stratified subset of 2,000 samples (1,000 per class) is used so that
#   50 epochs complete in ~14 minutes on CPU. Results are saved to
#   data/training_results.json alongside this note.
# ---------------------------------------------------------------------------

SUBSET_PER_CLASS = 1000   # 1 000 per class → 2 000 total
EPOCHS           = 50
BATCH_SIZE       = 32
LEARNING_RATE    = 0.01
N_QUBITS         = 9
N_LAYERS         = 4
POS_WEIGHT       = 2.0    # penalise missed confusion (class 1) more heavily


def build_qnode(n_qubits: int, n_layers: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    return qnode


class HybridModelV2(nn.Module):
    """
    9-qubit 4-layer VQC followed by a classical Linear(9,1) readout.
    Uses sigmoid at inference; trained with BCEWithLogitsLoss (no sigmoid
    in forward during training — see HybridModelV2Logits below).
    """
    def __init__(self, qlayer):
        super().__init__()
        self.q_layer = qlayer
        self.fc = nn.Linear(N_QUBITS, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.q_layer(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class HybridModelV2Logits(nn.Module):
    """Training variant — no sigmoid so BCEWithLogitsLoss is numerically stable."""
    def __init__(self, qlayer):
        super().__init__()
        self.q_layer = qlayer
        self.fc = nn.Linear(N_QUBITS, 1)

    def forward(self, x):
        x = self.q_layer(x)
        x = self.fc(x)
        return x


def run_svm_benchmark(X_train_np, y_train_np, X_test_np, y_test_np):
    """
    Trains an SVM (RBF kernel) on the same 9 features and split as the VQC.
    Returns a metrics dict.
    """
    print("\n--- SVM Benchmark ---")
    t0 = time.time()
    svm = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train_np, y_train_np)
    y_pred = svm.predict(X_test_np)
    elapsed = time.time() - t0

    acc  = accuracy_score(y_test_np, y_pred)
    prec = precision_score(y_test_np, y_pred, zero_division=0)
    rec  = recall_score(y_test_np, y_pred, zero_division=0)
    f1   = f1_score(y_test_np, y_pred, zero_division=0)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  Train time: {elapsed:.1f}s")
    return {"accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1_score": round(f1, 4),
            "train_time_s": round(elapsed, 2)}


def train_quantum_model_v2(data_path, model_save_path, scaler_save_path,
                            results_save_path):
    print(f"Loading processed data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return

    feature_cols = ['Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2',
                    'Gamma1', 'Gamma2', 'FocusRatio']
    target_col   = 'predefinedlabel'

    # -----------------------------------------------------------------------
    # Stratified subsample: SUBSET_PER_CLASS rows from each class
    # -----------------------------------------------------------------------
    df_0 = df[df[target_col] == 0].sample(n=SUBSET_PER_CLASS, random_state=42)
    df_1 = df[df[target_col] == 1].sample(n=SUBSET_PER_CLASS, random_state=42)
    df_sub = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Stratified subset: {len(df_sub)} samples "
          f"({SUBSET_PER_CLASS} per class).")

    X_all = df_sub[feature_cols].values
    y_all = df_sub[target_col].values

    # -----------------------------------------------------------------------
    # Train/test split (80/20) — BEFORE fitting the scaler
    # -----------------------------------------------------------------------
    split_idx = int(0.8 * len(X_all))
    X_train_raw, X_test_raw = X_all[:split_idx], X_all[split_idx:]
    y_train_np,  y_test_np  = y_all[:split_idx], y_all[split_idx:]

    # -----------------------------------------------------------------------
    # Fit MinMaxScaler [0, pi] on TRAINING data only, then save it.
    # This is the single scaler used for all inference — never refit at runtime.
    # -----------------------------------------------------------------------
    print("Fitting MinMaxScaler([0, pi]) on training split only...")
    scaler = MinMaxScaler(feature_range=(0, float(np.pi)))
    X_train_np = scaler.fit_transform(X_train_raw)
    X_test_np  = scaler.transform(X_test_raw)          # transform, not fit_transform

    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_save_path}")

    # -----------------------------------------------------------------------
    # Convert to tensors
    # -----------------------------------------------------------------------
    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32).reshape(-1, 1)
    X_test_t  = torch.tensor(X_test_np,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test_np,  dtype=torch.float32).reshape(-1, 1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE, shuffle=True
    )

    print(f"Train: {len(X_train_t)} samples | Test: {len(X_test_t)} samples")

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    qnode  = build_qnode(N_QUBITS, N_LAYERS)
    weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    model     = HybridModelV2Logits(qlayer)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(f"\nStarting VQC training: {EPOCHS} epochs, "
          f"batch_size={BATCH_SIZE}, lr={LEARNING_RATE} ...")
    loss_history = []
    total_start  = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)

        avg_train_loss = epoch_loss / len(X_train_t)

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
            test_loss   = criterion(test_logits, y_test_t).item()
            probs       = torch.sigmoid(test_logits)
            preds       = (probs > 0.5).float()
            test_acc    = (preds == y_test_t).float().mean().item()

        loss_history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "test_loss":  round(test_loss, 6),
            "test_acc":   round(test_acc, 4)
        })

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            elapsed = time.time() - total_start
            print(f"  Epoch {epoch:>3}/{EPOCHS} | "
                  f"train_loss={avg_train_loss:.4f} | "
                  f"test_loss={test_loss:.4f} | "
                  f"test_acc={test_acc:.4f} | "
                  f"elapsed={elapsed:.0f}s")

    total_time = time.time() - total_start
    print(f"\nVQC training complete in {total_time:.1f}s")

    # -----------------------------------------------------------------------
    # Final VQC metrics on test set
    # -----------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        final_probs = torch.sigmoid(model(X_test_t)).numpy().flatten()
        final_preds = (final_probs > 0.5).astype(int)

    vqc_acc  = accuracy_score(y_test_np, final_preds)
    vqc_prec = precision_score(y_test_np, final_preds, zero_division=0)
    vqc_rec  = recall_score(y_test_np, final_preds, zero_division=0)
    vqc_f1   = f1_score(y_test_np, final_preds, zero_division=0)

    print(f"\n--- VQC Final Test Metrics ---")
    print(f"  Accuracy : {vqc_acc:.4f}")
    print(f"  Precision: {vqc_prec:.4f}")
    print(f"  Recall   : {vqc_rec:.4f}")
    print(f"  F1 Score : {vqc_f1:.4f}")

    # -----------------------------------------------------------------------
    # Save model (state_dict of logits model; inference wraps with sigmoid)
    # -----------------------------------------------------------------------
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # -----------------------------------------------------------------------
    # SVM Benchmark (same split, same scaled data)
    # -----------------------------------------------------------------------
    svm_metrics = run_svm_benchmark(X_train_np, y_train_np, X_test_np, y_test_np)

    # -----------------------------------------------------------------------
    # Save training_results.json
    # -----------------------------------------------------------------------
    results = {
        "training_note": (
            f"Trained on stratified subset of {SUBSET_PER_CLASS} samples/class "
            f"({SUBSET_PER_CLASS * 2} total) due to quantum simulation cost on CPU. "
            f"Full dataset: 12,811 samples. "
            f"Scaler: MinMaxScaler([0, pi]) fitted on training split only."
        ),
        "hyperparameters": {
            "n_qubits": N_QUBITS,
            "n_layers": N_LAYERS,
            "weight_tensor_shape": [N_LAYERS, N_QUBITS, 3],
            "embedding": "AngleEmbedding",
            "ansatz": "StronglyEntanglingLayers",
            "output_layer": "Linear(9,1) + Sigmoid",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "loss": "BCEWithLogitsLoss",
            "pos_weight": POS_WEIGHT,
            "train_test_split": "80/20",
            "scaler": "MinMaxScaler feature_range=[0, pi] (fit on train only)"
        },
        "dataset": {
            "source_file": "data/processed_eeg.csv",
            "total_rows": 12811,
            "subset_used": SUBSET_PER_CLASS * 2,
            "class_balance": f"{SUBSET_PER_CLASS} class-0 / {SUBSET_PER_CLASS} class-1",
            "features": feature_cols
        },
        "vqc_results": {
            "final_test_accuracy":  round(vqc_acc, 4),
            "precision":            round(vqc_prec, 4),
            "recall":               round(vqc_rec, 4),
            "f1_score":             round(vqc_f1, 4),
            "train_time_s":         round(total_time, 1)
        },
        "svm_results": svm_metrics,
        "loss_history": loss_history
    }

    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
    with open(results_save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Training results saved to {results_save_path}")


if __name__ == "__main__":
    base_dir          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_path    = os.path.join(base_dir, 'data', 'processed_eeg.csv')
    model_save_path   = os.path.join(base_dir, 'data', 'quantum_focus_model_v2.pth')
    scaler_save_path  = os.path.join(base_dir, 'data', 'feature_scaler.pkl')
    results_save_path = os.path.join(base_dir, 'data', 'training_results.json')

    train_quantum_model_v2(processed_path, model_save_path,
                           scaler_save_path, results_save_path)
