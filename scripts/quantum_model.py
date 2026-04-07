"""
quantum_model.py  —  VQC training experiments for the QENCS project.

Three tasks:
  Task 1  Adam vs QNG comparison  (2 000-sample subset, 50 epochs each)
  Task 2  Full-dataset training   (all available samples, Adam, overnight)
  Task 3  MLP baseline            (same split/scaler as VQC, 50 epochs)

Results are written to data/training_results.json with keys:
  adam_results, qng_results, full_dataset_results, mlp_results
  (legacy keys vqc_results / svm_results / loss_history are also kept)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json
import os
import pickle
import random
import time

# Keep PennyLane/Matplotlib caches inside the writable sandbox.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np                          # plain numpy for speed elsewhere
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp          # PennyLane-aware numpy (for QNG)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

# ---------------------------------------------------------------------------
# Global hyper-parameters
# ---------------------------------------------------------------------------
SUBSET_PER_CLASS = 1000       # 1 000 per class → 2 000 total  (Tasks 1 & 3)
EPOCHS           = 50
BATCH_SIZE       = 32
LEARNING_RATE    = 0.01
N_QUBITS         = 9
N_LAYERS         = 4
POS_WEIGHT       = 2.0        # penalise missed confusion (class 1) more
SEED             = 42

# QNG-specific
QNG_APPROX = "block-diag"     # fastest correct approximation
QNG_LAM    = 1e-4


def set_global_seeds(seed: int = SEED):
    """Keep all experiment runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    pnp.random.seed(seed)
    torch.manual_seed(seed)


def rounded_float(value, digits=4):
    """Convert numpy/autograd scalars to JSON-safe Python floats."""
    return float(round(float(value), digits))


# ===========================================================================
# Shared circuit / model builders
# ===========================================================================

def build_qnode_torch(n_qubits: int, n_layers: int):
    """QNode wrapped for PyTorch TorchLayer (used by Adam run)."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    return qnode


def build_qnode_native(n_qubits: int, n_layers: int):
    """QNode for PennyLane-native optimisers (QNG)."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    return circuit


# ---------------------------------------------------------------------------
# PyTorch model variants
# ---------------------------------------------------------------------------

class HybridModelV2Logits(nn.Module):
    """VQC + linear readout (no sigmoid — BCEWithLogitsLoss handles it)."""
    def __init__(self, qlayer):
        super().__init__()
        self.q_layer = qlayer
        self.fc      = nn.Linear(N_QUBITS, 1)

    def forward(self, x):
        x = self.q_layer(x)
        return self.fc(x)


class MLPBaseline(nn.Module):
    """3-layer MLP: Linear(9,18) → ReLU → Linear(18,18) → ReLU → Linear(18,1)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9,  18), nn.ReLU(),
            nn.Linear(18, 18), nn.ReLU(),
            nn.Linear(18, 1),
        )

    def forward(self, x):
        return self.net(x)


# ===========================================================================
# Data preparation helper
# ===========================================================================

def prepare_data(df, feature_cols, target_col, subset_per_class=None,
                 random_state=42):
    """
    Stratified subsample (if subset_per_class is given), 80/20 split,
    fit MinMaxScaler([0, π]) on training split only.

    Returns: X_train_np, X_test_np, y_train_np, y_test_np, scaler
    """
    if subset_per_class is not None:
        n0 = min(subset_per_class, (df[target_col] == 0).sum())
        n1 = min(subset_per_class, (df[target_col] == 1).sum())
        df_0 = df[df[target_col] == 0].sample(n=n0, random_state=random_state)
        df_1 = df[df[target_col] == 1].sample(n=n1, random_state=random_state)
        df = pd.concat([df_0, df_1]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        print(f"  Subset: {len(df)} samples ({n0} class-0, {n1} class-1)")
    else:
        print(f"  Full dataset: {len(df)} samples")

    X_all = df[feature_cols].values
    y_all = df[target_col].astype(int).values

    X_train_raw, X_test_raw, y_train_np, y_test_np = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=random_state,
        stratify=y_all,
        shuffle=True,
    )

    scaler = MinMaxScaler(feature_range=(0, float(np.pi)))
    X_train_np = scaler.fit_transform(X_train_raw)
    X_test_np  = scaler.transform(X_test_raw)

    return X_train_np, X_test_np, y_train_np, y_test_np, scaler


# ===========================================================================
# Task 1a — Adam training (PyTorch)
# ===========================================================================

def run_adam_training(X_train_np, X_test_np, y_train_np, y_test_np,
                      epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                      label="Adam"):
    print(f"\n{'='*60}")
    print(f"  VQC training — {label} optimiser")
    print(f"  train={len(X_train_np)}  test={len(X_test_np)}  "
          f"epochs={epochs}  batch={batch_size}  lr={lr}")
    print(f"{'='*60}")

    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32).reshape(-1, 1)
    X_test_t  = torch.tensor(X_test_np,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test_np,  dtype=torch.float32).reshape(-1, 1)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    qnode  = build_qnode_torch(N_QUBITS, N_LAYERS)
    qlayer = qml.qnn.TorchLayer(qnode, {"weights": (N_LAYERS, N_QUBITS, 3)})
    model     = HybridModelV2Logits(qlayer)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in loader:
            optimizer.zero_grad()
            out  = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(Xb)

        avg_train_loss = epoch_loss / len(X_train_t)

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
            test_loss   = criterion(test_logits, y_test_t).item()
            probs       = torch.sigmoid(test_logits)
            preds       = (probs > 0.5).float()
            test_acc    = (preds == y_test_t).float().mean().item()

        loss_history.append({
            "epoch": epoch,
            "train_loss": rounded_float(avg_train_loss, 6),
            "test_loss":  rounded_float(test_loss, 6),
            "test_acc":   rounded_float(test_acc, 4),
        })

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            elapsed = time.time() - t0
            eta     = (elapsed / epoch) * (epochs - epoch)
            print(f"  [{label}] Ep {epoch:>3}/{epochs} | "
                  f"train={avg_train_loss:.4f} test={test_loss:.4f} "
                  f"acc={test_acc:.4f} | {elapsed:.0f}s  ETA {eta:.0f}s")

    total_time = time.time() - t0

    # Final metrics
    model.eval()
    with torch.no_grad():
        final_probs = torch.sigmoid(model(X_test_t)).numpy().flatten()
    final_preds = (final_probs > 0.5).astype(int)

    acc  = accuracy_score(y_test_np, final_preds)
    prec = precision_score(y_test_np, final_preds, zero_division=0)
    rec  = recall_score(y_test_np, final_preds, zero_division=0)
    f1   = f1_score(y_test_np, final_preds, zero_division=0)

    print(f"\n  [{label}] Accuracy={acc:.4f}  Precision={prec:.4f}  "
          f"Recall={rec:.4f}  F1={f1:.4f}  time={total_time:.1f}s")

    return {
        "optimiser": "Adam",
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "accuracy":     rounded_float(acc,  4),
        "precision":    rounded_float(prec, 4),
        "recall":       rounded_float(rec,  4),
        "f1_score":     rounded_float(f1,   4),
        "train_time_s": rounded_float(total_time, 1),
        "loss_history": loss_history,
    }, model


# ===========================================================================
# Task 1b — QNG training (PennyLane native)
# ===========================================================================

def _bce_logits_pl(logit, y, pos_weight=POS_WEIGHT):
    """Binary cross-entropy with logits + pos_weight, using PL numpy."""
    # Numerically stable form: max(x,0) - x*y + log(1+exp(-|x|))
    # Weighted: pos_weight * y * log σ(x) + (1-y) * log(1-σ(x))
    # = -(pw * y * log(1/(1+e^-x)) + (1-y) * log(e^-x/(1+e^-x)))
    eps = 1e-8
    sig = 1.0 / (1.0 + pnp.exp(-logit))
    loss = -(pos_weight * y * pnp.log(sig + eps) +
             (1.0 - y) * pnp.log(1.0 - sig + eps))
    return loss


def run_qng_training(X_train_np, X_test_np, y_train_np, y_test_np,
                     epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """
    Quantum Natural Gradient (QNG) training for the quantum weights, with a
    classical SGD update for the linear readout.

    Architecture identical to the Adam run:
      AngleEmbedding → StronglyEntanglingLayers(4) → Linear(9,1) readout.
    """
    print(f"\n{'='*60}")
    print(f"  VQC training — QNGOptimizer (approx={QNG_APPROX})")
    print(f"  train={len(X_train_np)}  test={len(X_test_np)}  "
          f"epochs={epochs}  batch={batch_size}  lr={lr}")
    print(f"  NOTE: QNG is slower than Adam per epoch (metric-tensor cost).")
    print(f"{'='*60}")

    circuit = build_qnode_native(N_QUBITS, N_LAYERS)
    mt_fn = qml.metric_tensor(circuit, approx=QNG_APPROX)
    qng_opt = qml.QNGOptimizer(stepsize=lr, approx=QNG_APPROX, lam=QNG_LAM)

    # Initialise parameters — same seed as would be used for Adam re-run
    pnp.random.seed(SEED)
    q_weights = pnp.random.uniform(-pnp.pi, pnp.pi,
                                   (N_LAYERS, N_QUBITS, 3), requires_grad=True)
    w_fc = pnp.array(np.random.default_rng(SEED).standard_normal(N_QUBITS) * 0.1,
                     requires_grad=True)
    b_fc = pnp.array(0.0, requires_grad=True)

    loss_history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        idx        = np.random.permutation(len(X_train_np))
        epoch_loss = 0.0
        n_seen     = 0

        for batch_num, start in enumerate(range(0, len(X_train_np), batch_size), start=1):
            batch_idx = idx[start : start + batch_size]
            X_b = X_train_np[batch_idx]
            y_b = y_train_np[batch_idx]

            # ----------------------------------------------------------
            # Step 1 — quantum gradient  ∂L/∂q_weights
            # Freeze classical params so autograd only tracks q_weights.
            # ----------------------------------------------------------
            w_fc_frozen = pnp.array(np.array(w_fc), requires_grad=False)
            b_fc_frozen = float(b_fc)

            def cost_quantum(q_w):
                total = pnp.array(0.0)
                for i in range(len(X_b)):
                    x_i   = pnp.array(X_b[i], requires_grad=False)
                    q_out = pnp.stack(circuit(x_i, q_w))
                    logit = pnp.dot(q_out, w_fc_frozen) + b_fc_frozen
                    total = total + _bce_logits_pl(logit, float(y_b[i]))
                return total / len(X_b)

            x_rep = pnp.array(X_b[0], requires_grad=False)
            q_grad_fn = lambda q_w: qml.grad(cost_quantum)(q_w)
            q_metric_fn = lambda q_w: mt_fn(x_rep, q_w)

            q_weights, batch_loss_val = qng_opt.step_and_cost(
                cost_quantum,
                q_weights,
                grad_fn=q_grad_fn,
                metric_tensor_fn=q_metric_fn,
                recompute_tensor=(batch_num == 1),
            )

            # ----------------------------------------------------------
            # Step 4 — classical SGD update for linear readout
            # Freeze updated quantum weights; differentiate w.r.t. w_fc, b_fc.
            # ----------------------------------------------------------
            q_w_frozen = pnp.array(np.array(q_weights), requires_grad=False)

            def cost_classical(w_f, b_f):
                total = pnp.array(0.0)
                for i in range(len(X_b)):
                    x_i   = pnp.array(X_b[i], requires_grad=False)
                    q_out = pnp.stack(circuit(x_i, q_w_frozen))
                    logit = pnp.dot(q_out, w_f) + b_f
                    total = total + _bce_logits_pl(logit, float(y_b[i]))
                return total / len(X_b)

            w_grad, b_grad = qml.grad(cost_classical, argnum=[0, 1])(w_fc, b_fc)
            w_fc = pnp.array(np.array(w_fc) - lr * np.array(w_grad), requires_grad=True)
            b_fc = pnp.array(float(b_fc)   - lr * float(b_grad),     requires_grad=True)

            epoch_loss += float(batch_loss_val) * len(batch_idx)
            n_seen     += len(batch_idx)

        avg_train_loss = epoch_loss / n_seen

        # ------------------------------------------------------------------
        # Evaluation on test set (plain numpy — no grad tracking needed)
        # ------------------------------------------------------------------
        q_w_eval   = pnp.array(np.array(q_weights), requires_grad=False)
        w_fc_eval  = np.array(w_fc)
        b_fc_eval  = float(b_fc)

        test_logits_list = []
        for i in range(len(X_test_np)):
            x_i   = pnp.array(X_test_np[i], requires_grad=False)
            q_out = np.array(list(circuit(x_i, q_w_eval)))
            test_logits_list.append(np.dot(q_out, w_fc_eval) + b_fc_eval)

        test_logits_np = np.array(test_logits_list)
        test_probs     = 1.0 / (1.0 + np.exp(-test_logits_np))

        eps       = 1e-8
        test_loss = -np.mean(
            y_test_np * np.log(test_probs + eps) +
            (1 - y_test_np) * np.log(1 - test_probs + eps)
        )
        test_acc  = ((test_probs > 0.5).astype(float) == y_test_np).mean()

        loss_history.append({
            "epoch":      epoch,
            "train_loss": rounded_float(avg_train_loss, 6),
            "test_loss":  rounded_float(test_loss, 6),
            "test_acc":   rounded_float(test_acc, 4),
        })

        elapsed = time.time() - t0
        eta     = (elapsed / epoch) * (epochs - epoch)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  [QNG] Ep {epoch:>3}/{epochs} | "
                  f"train={avg_train_loss:.4f} test={test_loss:.4f} "
                  f"acc={test_acc:.4f} | {elapsed:.0f}s  ETA {eta:.0f}s")

    total_time = time.time() - t0

    # Final metrics
    final_preds_int = (test_probs > 0.5).astype(int)
    acc  = accuracy_score(y_test_np, final_preds_int)
    prec = precision_score(y_test_np, final_preds_int, zero_division=0)
    rec  = recall_score(y_test_np, final_preds_int, zero_division=0)
    f1   = f1_score(y_test_np, final_preds_int, zero_division=0)

    print(f"\n  [QNG] Accuracy={acc:.4f}  Precision={prec:.4f}  "
          f"Recall={rec:.4f}  F1={f1:.4f}  time={total_time:.1f}s")

    return {
        "optimiser":     f"QNGOptimizer(approx={QNG_APPROX})",
        "learning_rate": lr,
        "epochs":        epochs,
        "batch_size":    batch_size,
        "accuracy":      rounded_float(acc,  4),
        "precision":     rounded_float(prec, 4),
        "recall":        rounded_float(rec,  4),
        "f1_score":      rounded_float(f1,   4),
        "train_time_s":  rounded_float(total_time, 1),
        "loss_history":  loss_history,
    }


# ===========================================================================
# Task 3 — MLP baseline (PyTorch)
# ===========================================================================

def run_mlp_training(X_train_np, X_test_np, y_train_np, y_test_np,
                     epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    print(f"\n{'='*60}")
    print(f"  MLP baseline  Linear(9,18)→ReLU→Linear(18,18)→ReLU→Linear(18,1)")
    print(f"  train={len(X_train_np)}  test={len(X_test_np)}  "
          f"epochs={epochs}  batch={batch_size}  lr={lr}")
    print(f"{'='*60}")

    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32).reshape(-1, 1)
    X_test_t  = torch.tensor(X_test_np,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test_np,  dtype=torch.float32).reshape(-1, 1)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    model     = MLPBaseline()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in loader:
            optimizer.zero_grad()
            out  = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(Xb)

        avg_train_loss = epoch_loss / len(X_train_t)

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
            test_loss   = criterion(test_logits, y_test_t).item()
            probs       = torch.sigmoid(test_logits)
            preds       = (probs > 0.5).float()
            test_acc    = (preds == y_test_t).float().mean().item()

        loss_history.append({
            "epoch":      epoch,
            "train_loss": rounded_float(avg_train_loss, 6),
            "test_loss":  rounded_float(test_loss, 6),
            "test_acc":   rounded_float(test_acc, 4),
        })

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            elapsed = time.time() - t0
            print(f"  [MLP] Ep {epoch:>3}/{epochs} | "
                  f"train={avg_train_loss:.4f} test={test_loss:.4f} "
                  f"acc={test_acc:.4f} | {elapsed:.0f}s")

    total_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        final_probs = torch.sigmoid(model(X_test_t)).numpy().flatten()
    final_preds = (final_probs > 0.5).astype(int)

    acc  = accuracy_score(y_test_np, final_preds)
    prec = precision_score(y_test_np, final_preds, zero_division=0)
    rec  = recall_score(y_test_np, final_preds, zero_division=0)
    f1   = f1_score(y_test_np, final_preds, zero_division=0)

    print(f"\n  [MLP] Accuracy={acc:.4f}  Precision={prec:.4f}  "
          f"Recall={rec:.4f}  F1={f1:.4f}  time={total_time:.1f}s")

    return {
        "architecture":  "Linear(9,18)→ReLU→Linear(18,18)→ReLU→Linear(18,1)",
        "optimiser":     "Adam",
        "learning_rate": lr,
        "epochs":        epochs,
        "batch_size":    batch_size,
        "accuracy":      rounded_float(acc,  4),
        "precision":     rounded_float(prec, 4),
        "recall":        rounded_float(rec,  4),
        "f1_score":      rounded_float(f1,   4),
        "train_time_s":  rounded_float(total_time, 1),
        "loss_history":  loss_history,
    }


# ===========================================================================
# SVM benchmark
# ===========================================================================

def run_svm_benchmark(X_train_np, y_train_np, X_test_np, y_test_np):
    print("\n--- SVM Benchmark (RBF, C=1, gamma=scale) ---")
    t0 = time.time()
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train_np, y_train_np)
    y_pred   = svm.predict(X_test_np)
    elapsed  = time.time() - t0
    acc  = accuracy_score(y_test_np, y_pred)
    prec = precision_score(y_test_np, y_pred, zero_division=0)
    rec  = recall_score(y_test_np, y_pred, zero_division=0)
    f1   = f1_score(y_test_np, y_pred, zero_division=0)
    print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  "
          f"Recall={rec:.4f}  F1={f1:.4f}  time={elapsed:.2f}s")
    return {"accuracy": rounded_float(acc, 4), "precision": rounded_float(prec, 4),
            "recall": rounded_float(rec, 4), "f1_score": rounded_float(f1, 4),
            "train_time_s": rounded_float(elapsed, 2)}


# ===========================================================================
# Comparison tables
# ===========================================================================

def print_comparison_tables(results: dict):
    adam = results.get("adam_results", {})
    qng  = results.get("qng_results",  {})
    full = results.get("full_dataset_results", {})
    mlp  = results.get("mlp_results",  {})
    svm  = results.get("svm_results",  {})

    W = 62

    # ------------------------------------------------------------------
    # Table 1: Adam vs QNG  (2 000-sample subset)
    # ------------------------------------------------------------------
    print("\n" + "=" * W)
    print("  TABLE 1  Adam vs QNG  (2 000-sample subset, 50 epochs)")
    print("=" * W)
    print(f"  {'Metric':<18} {'Adam':>10} {'QNG':>10} {'Delta':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10}")
    for key, label in [("accuracy",  "Accuracy"),
                        ("precision", "Precision"),
                        ("recall",    "Recall"),
                        ("f1_score",  "F1 Score")]:
        a = adam.get(key, float("nan"))
        q = qng.get(key,  float("nan"))
        d = q - a if isinstance(q, float) and isinstance(a, float) else float("nan")
        sign = "+" if d >= 0 else ""
        print(f"  {label:<18} {a:>10.4f} {q:>10.4f} {sign+f'{d:.4f}':>10}")
    for key, label in [("train_time_s", "Train time (s)")]:
        a = adam.get(key, float("nan"))
        q = qng.get(key,  float("nan"))
        print(f"  {label:<18} {a:>10.1f} {q:>10.1f}")

    # Convergence observation
    if adam.get("loss_history") and qng.get("loss_history"):
        a_hist = adam["loss_history"]
        q_hist = qng["loss_history"]
        # Find epoch where train_loss first drops below 0.60
        def first_below(hist, thr):
            for h in hist:
                if h["train_loss"] < thr:
                    return h["epoch"]
            return ">50"
        a_thresh = first_below(a_hist, 0.60)
        q_thresh = first_below(q_hist, 0.60)
        print(f"\n  First epoch train_loss < 0.60:  Adam={a_thresh}  QNG={q_thresh}")
        a_final = a_hist[-1]["train_loss"]
        q_final = q_hist[-1]["train_loss"]
        print(f"  Final epoch train_loss:          Adam={a_final:.4f}  QNG={q_final:.4f}")
        if isinstance(q_thresh, int) and isinstance(a_thresh, int):
            if q_thresh < a_thresh:
                print("  → QNG converged FASTER than Adam.")
            elif q_thresh > a_thresh:
                print("  → Adam converged FASTER than QNG.")
            else:
                print("  → Similar convergence speed.")
        print(f"\n  Did QNG make a meaningful difference?")
        delta_f1 = qng.get("f1_score", 0) - adam.get("f1_score", 0)
        if abs(delta_f1) < 0.01:
            print(f"  No significant difference in final F1 (Δ={delta_f1:+.4f}).")
            print(f"  QNG is {qng.get('train_time_s',0)/max(adam.get('train_time_s',1),1):.1f}× "
                  f"slower — Adam is preferred for this scale.")
        elif delta_f1 > 0:
            print(f"  QNG improved F1 by {delta_f1:+.4f} — the metric-tensor "
                  f"preconditioner helped navigate the loss landscape.")
        else:
            print(f"  Adam outperformed QNG by {-delta_f1:.4f} F1 — likely because "
                  f"mini-batch metric-tensor approximation introduced noise.")

    # ------------------------------------------------------------------
    # Table 2: 2 000-sample vs full dataset  (both Adam)
    # ------------------------------------------------------------------
    print("\n" + "=" * W)
    print("  TABLE 2  2 000-sample vs Full-dataset  (Adam)")
    print("=" * W)
    subset_vqc = {k: adam.get(k) for k in ("accuracy", "precision", "recall",
                                             "f1_score", "train_time_s")}
    full_vqc   = {k: full.get(k) for k in ("accuracy", "precision", "recall",
                                             "f1_score", "train_time_s")}
    print(f"  {'Metric':<18} {'2k-subset':>12} {'Full dataset':>14} {'Delta':>10}")
    print(f"  {'-'*18} {'-'*12} {'-'*14} {'-'*10}")
    for key, label in [("accuracy",  "Accuracy"),
                        ("precision", "Precision"),
                        ("recall",    "Recall"),
                        ("f1_score",  "F1 Score")]:
        s = subset_vqc.get(key, float("nan"))
        f = full_vqc.get(key,   float("nan"))
        d = f - s if isinstance(f, float) and isinstance(s, float) else float("nan")
        sign = "+" if d >= 0 else ""
        print(f"  {label:<18} {s:>12.4f} {f:>14.4f} {sign+f'{d:.4f}':>10}")
    ts = subset_vqc.get("train_time_s", float("nan"))
    tf = full_vqc.get("train_time_s",   float("nan"))
    print(f"  {'Train time (s)':<18} {ts:>12.1f} {tf:>14.1f}")

    print(f"\n  Did the full dataset change results meaningfully?")
    delta_f1_full = (full_vqc.get("f1_score", 0) or 0) - (subset_vqc.get("f1_score", 0) or 0)
    if abs(delta_f1_full) < 0.02:
        print(f"  Marginal gain (Δ F1={delta_f1_full:+.4f}). The 2k subset is likely "
              f"sufficient for this circuit depth — additional samples do not "
              f"reduce VQC variance meaningfully on CPU simulation.")
    elif delta_f1_full > 0:
        print(f"  Full dataset improved F1 by {delta_f1_full:+.4f}. More data "
              f"helped the model generalise.")
    else:
        print(f"  Surprising: full dataset slightly worse (Δ={delta_f1_full:+.4f}). "
              f"Possible over-fitting or learning-rate needs tuning for larger N.")

    # ------------------------------------------------------------------
    # Table 3: VQC (Adam) vs SVM vs MLP
    # ------------------------------------------------------------------
    print("\n" + "=" * W)
    print("  TABLE 3  VQC (Adam) vs SVM vs MLP  (2 000-sample subset)")
    print("=" * W)
    print(f"  {'Metric':<18} {'VQC (Adam)':>12} {'SVM':>10} {'MLP':>10}")
    print(f"  {'-'*18} {'-'*12} {'-'*10} {'-'*10}")
    for key, label in [("accuracy",  "Accuracy"),
                        ("precision", "Precision"),
                        ("recall",    "Recall"),
                        ("f1_score",  "F1 Score"),
                        ("train_time_s", "Train time (s)")]:
        a = adam.get(key, float("nan"))
        s = svm.get(key,  float("nan"))
        m = mlp.get(key,  float("nan"))
        if key == "train_time_s":
            print(f"  {label:<18} {a:>12.1f} {s:>10.1f} {m:>10.1f}")
        else:
            print(f"  {label:<18} {a:>12.4f} {s:>10.4f} {m:>10.4f}")

    print(f"\n  Takeaway:")
    models = {"VQC": adam.get("f1_score", 0) or 0,
              "SVM": svm.get("f1_score",  0) or 0,
              "MLP": mlp.get("f1_score",  0) or 0}
    best   = max(models, key=models.get)
    worst  = min(models, key=models.get)
    print(f"  Best F1: {best} ({models[best]:.4f})   "
          f"Worst F1: {worst} ({models[worst]:.4f})")
    vqc_f1 = models["VQC"]
    mlp_f1 = models["MLP"]
    if mlp_f1 > vqc_f1 + 0.02:
        print(f"  MLP outperforms VQC by {mlp_f1-vqc_f1:.4f} F1 on CPU simulation. "
              f"The classical baseline is competitive — quantum advantage would "
              f"require hardware execution or deeper circuits.")
    elif vqc_f1 > mlp_f1 + 0.02:
        print(f"  VQC outperforms MLP by {vqc_f1-mlp_f1:.4f} F1. The quantum "
              f"feature map provides a meaningful inductive bias for this dataset.")
    else:
        print(f"  VQC and MLP perform comparably (Δ F1 < 0.02). This is expected "
              f"for classical simulation with limited circuit depth.")
    print("=" * W)


# ===========================================================================
# Main orchestration
# ===========================================================================

def main():
    base_dir          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_path    = os.path.join(base_dir, 'data', 'processed_eeg.csv')
    model_save_path   = os.path.join(base_dir, 'data', 'quantum_focus_model_v2.pth')
    scaler_save_path  = os.path.join(base_dir, 'data', 'feature_scaler.pkl')
    results_save_path = os.path.join(base_dir, 'data', 'training_results.json')

    feature_cols = ['Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2',
                    'Gamma1', 'Gamma2', 'FocusRatio']
    target_col   = 'predefinedlabel'

    set_global_seeds(SEED)

    print(f"Loading processed data from {processed_path} ...")
    try:
        df = pd.read_csv(processed_path)
    except FileNotFoundError:
        print(f"ERROR: {processed_path} not found.")
        return

    print(f"  Dataset: {len(df)} rows  |  "
          f"class distribution:\n  {df[target_col].value_counts().to_dict()}")

    # -----------------------------------------------------------------------
    # SHARED data prep for Tasks 1 & 3  (2 000-sample subset)
    # -----------------------------------------------------------------------
    print("\n[Preparing 2 000-sample subset ...]")
    X_tr2k, X_te2k, y_tr2k, y_te2k, scaler_2k = prepare_data(
        df, feature_cols, target_col, subset_per_class=SUBSET_PER_CLASS
    )

    # Save scaler
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler_2k, f)
    print(f"  Scaler saved → {scaler_save_path}")

    # -----------------------------------------------------------------------
    # TASK 1a — Adam on 2 000 samples
    # -----------------------------------------------------------------------
    adam_results, adam_model = run_adam_training(
        X_tr2k, X_te2k, y_tr2k, y_te2k, epochs=EPOCHS
    )

    # Save the trained Adam model
    torch.save(adam_model.state_dict(), model_save_path)
    print(f"  Model saved → {model_save_path}")

    # SVM on the same 2k split (for Table 3)
    svm_results = run_svm_benchmark(X_tr2k, y_tr2k, X_te2k, y_te2k)

    # -----------------------------------------------------------------------
    # TASK 1b — QNG on 2 000 samples
    # -----------------------------------------------------------------------
    qng_results = run_qng_training(
        X_tr2k, X_te2k, y_tr2k, y_te2k, epochs=EPOCHS
    )

    # -----------------------------------------------------------------------
    # TASK 3 — MLP baseline on 2 000 samples  (same split/scaler)
    # -----------------------------------------------------------------------
    mlp_results = run_mlp_training(
        X_tr2k, X_te2k, y_tr2k, y_te2k, epochs=EPOCHS
    )

    # -----------------------------------------------------------------------
    # TASK 2 — Full-dataset Adam (overnight)
    # -----------------------------------------------------------------------
    print("\n[Preparing FULL dataset ...]")
    X_tr_full, X_te_full, y_tr_full, y_te_full, _ = prepare_data(
        df, feature_cols, target_col, subset_per_class=None
    )
    full_adam_results, _ = run_adam_training(
        X_tr_full, X_te_full, y_tr_full, y_te_full,
        epochs=EPOCHS, label="Adam-full"
    )
    svm_full = run_svm_benchmark(X_tr_full, y_tr_full, X_te_full, y_te_full)

    # Annotate full-dataset result with extra metadata
    full_dataset_results = {
        **full_adam_results,
        "dataset_size":    len(df),
        "train_samples":   len(X_tr_full),
        "test_samples":    len(X_te_full),
        "svm_comparison":  svm_full,
    }

    # -----------------------------------------------------------------------
    # Assemble and save training_results.json
    # -----------------------------------------------------------------------
    results = {
        # -- New structured keys -----------------------------------------
        "adam_results": {
            **adam_results,
            "dataset":      f"stratified subset {SUBSET_PER_CLASS}/class ({SUBSET_PER_CLASS*2} total)",
            "train_samples": len(X_tr2k),
            "test_samples":  len(X_te2k),
        },
        "qng_results": {
            **qng_results,
            "dataset":      f"stratified subset {SUBSET_PER_CLASS}/class ({SUBSET_PER_CLASS*2} total)",
            "train_samples": len(X_tr2k),
            "test_samples":  len(X_te2k),
        },
        "full_dataset_results": full_dataset_results,
        "mlp_results": {
            **mlp_results,
            "dataset":      f"stratified subset {SUBSET_PER_CLASS}/class ({SUBSET_PER_CLASS*2} total)",
            "train_samples": len(X_tr2k),
            "test_samples":  len(X_te2k),
        },
        # -- Legacy keys (backward-compat with web app / research viewer) -
        "training_note": (
            f"Multi-experiment run: Adam vs QNG (2k subset), full-dataset Adam, MLP baseline. "
            f"Scaler: MinMaxScaler([0,π]) fit on training split only."
        ),
        "hyperparameters": {
            "n_qubits": N_QUBITS, "n_layers": N_LAYERS,
            "weight_tensor_shape": [N_LAYERS, N_QUBITS, 3],
            "embedding": "AngleEmbedding", "ansatz": "StronglyEntanglingLayers",
            "output_layer": "Linear(9,1) + Sigmoid",
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "optimizer": "Adam (primary) / QNGOptimizer (comparison)",
            "learning_rate": LEARNING_RATE,
            "loss": "BCEWithLogitsLoss", "pos_weight": POS_WEIGHT,
            "train_test_split": "80/20",
            "scaler": "MinMaxScaler feature_range=[0, pi] (fit on train only)",
        },
        "dataset": {
            "source_file": "data/processed_eeg.csv",
            "total_rows":  len(df),
            "subset_used": SUBSET_PER_CLASS * 2,
            "class_balance": f"{SUBSET_PER_CLASS} class-0 / {SUBSET_PER_CLASS} class-1",
            "features": feature_cols,
        },
        # Legacy single-result keys (point to Adam 2k run)
        "vqc_results": {
            "final_test_accuracy": adam_results["accuracy"],
            "precision":           adam_results["precision"],
            "recall":              adam_results["recall"],
            "f1_score":            adam_results["f1_score"],
            "train_time_s":        adam_results["train_time_s"],
        },
        "svm_results":   svm_results,
        "loss_history":  adam_results["loss_history"],
    }

    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
    with open(results_save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nAll results saved → {results_save_path}")

    # -----------------------------------------------------------------------
    # Print comparison tables
    # -----------------------------------------------------------------------
    print_comparison_tables(results)


if __name__ == "__main__":
    main()
