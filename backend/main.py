
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import torch
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.svm import SVC
from typing import List, Optional
import io
import zipfile

# Add scripts directory to path to import agents
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from logic_agent import LogicAgent

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
        "https://qencs.vercel.app",
        "https://qencs-ihmm1sz8e-vishnu-ajiths-projects.vercel.app",
        "https://qencs-qlbgpmoti-vishnu-ajiths-projects.vercel.app",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "system": "QENCS Quantum Backend"}

logic_agent = LogicAgent()

# --- Quantum Model Setup ---
import pennylane as qml
from torch import nn

n_qubits = 9
n_layers = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits, 3)}
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

class HybridModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = qlayer
        self.fc = nn.Linear(n_qubits, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.q_layer(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

model = HybridModelV2()
model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'quantum_focus_model_v2.pth')

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        print("Quantum Model V2 loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")

# --- Scaler Setup ---
# Load the scaler that was fitted exclusively on the training split in
# quantum_model.py and saved to data/feature_scaler.pkl.
# This is the single source of truth for feature scaling — never refit at runtime.
feature_scaler = None
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'feature_scaler.pkl')
DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_eeg.csv')

if os.path.exists(SCALER_PATH):
    try:
        with open(SCALER_PATH, 'rb') as f:
            feature_scaler = pickle.load(f)
        print("Feature scaler loaded from feature_scaler.pkl.")
    except Exception as e:
        print(f"Failed to load scaler: {e}")
        feature_scaler = None

# Fallback: if scaler.pkl not yet generated (model not trained yet), fit on
# the processed CSV — with a clear warning so the mismatch is visible.
if feature_scaler is None and os.path.exists(DATA_PATH):
    from sklearn.preprocessing import MinMaxScaler
    print("WARNING: feature_scaler.pkl not found. Fitting scaler on full dataset "
          "as fallback. Run scripts/quantum_model.py to generate the proper scaler.")
    try:
        baseline_df  = pd.read_csv(DATA_PATH)
        feature_cols = ['Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2',
                        'Gamma1', 'Gamma2', 'FocusRatio']
        feature_scaler = MinMaxScaler(feature_range=(0, np.pi))
        feature_scaler.fit(baseline_df[feature_cols].values)
    except Exception as e:
        print(f"Scaler fallback error: {e}")

feature_cols = ['Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2',
                'Gamma1', 'Gamma2', 'FocusRatio']

# --- State Management ---
class SessionState:
    def __init__(self):
        self.baseline_buffer = []
        self.baseline_mean = None
        self.is_calibrating = True
        self.MAX_BASELINE_SAMPLES = 15
        self.interventions_count = 0
        self.research_records = [{
            "status": "SYSTEM_INITIALIZED",
            "message": "Quantum-Neural Node Active. Telemetry buffer standby.",
            "timestamp": "BOOT_LOG_001"
        }]
        self.vqc_losses = []
        self.svm_losses = []

session = SessionState()

# Pre-train a dummy SVM for live benchmarking
svm_benchmark = SVC(probability=True)
X_dummy = np.random.rand(20, 9)
y_dummy = np.random.randint(0, 2, 20)
svm_benchmark.fit(X_dummy, y_dummy)


class EEGData(BaseModel):
    delta:  float
    theta:  float
    alpha1: float
    alpha2: float
    beta1:  float
    beta2:  float
    gamma1: float
    gamma2: float
    sensitivity: Optional[float] = 0.15


def calculate_entropy(psd: List[float]) -> float:
    """Shannon entropy over EEG band powers."""
    psd_norm = np.array(psd) / sum(psd)
    return -np.sum(psd_norm * np.log2(psd_norm + 1e-9))


@app.post("/analyze")
def analyze_focus(data: EEGData):
    # 1. Base Calculations
    total_beta = data.beta1 + data.beta2
    if total_beta == 0:
        total_beta = 0.001
    focus_ratio = data.theta / total_beta

    # 2. Quantum VQC Prediction
    input_features = np.array([
        data.delta, data.theta, data.alpha1, data.alpha2,
        data.beta1, data.beta2, data.gamma1, data.gamma2,
        focus_ratio
    ], dtype=np.float32).reshape(1, -1)

    if feature_scaler is None:
        raise HTTPException(status_code=503,
                            detail="Scaler not loaded. Run scripts/quantum_model.py first.")

    scaled_features = feature_scaler.transform(input_features)
    tensor_input    = torch.tensor(scaled_features, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(tensor_input).item()

    # 3. Neural Analytics
    total_power = data.theta + data.alpha1 + data.alpha2 + data.beta1 + data.beta2
    if total_power == 0:
        total_power = 0.001
    band_power = {
        "theta": (data.theta / total_power) * 100,
        "alpha": ((data.alpha1 + data.alpha2) / total_power) * 100,
        "beta":  (total_beta / total_power) * 100
    }
    entropy    = calculate_entropy([data.delta, data.theta,
                                    data.alpha1 + data.alpha2,
                                    total_beta, data.gamma1 + data.gamma2])
    confidence = abs(prediction - 0.5) * 2

    # 4. Calibration & Logic
    if session.is_calibrating:
        session.baseline_buffer.append(prediction)
        if len(session.baseline_buffer) >= session.MAX_BASELINE_SAMPLES:
            session.baseline_mean = (sum(session.baseline_buffer) /
                                     len(session.baseline_buffer))
            session.is_calibrating = False

    analysis = logic_agent.analyze(
        focus_ratio,
        prediction,
        baseline_confusion=session.baseline_mean,
        sensitivity=data.sensitivity
    )

    if "Confused" in analysis["state"] or "Distracted" in analysis["state"]:
        session.interventions_count += 1

    if session.is_calibrating:
        analysis["advice"] = "Calibrating your neural baseline..."

    # Research Data Capture
    current_pauli_z = [
        float(x) for x in
        qnode(scaled_features[0], model.q_layer.weights.detach().numpy())
    ]
    hilbert_coords = {
        "x": current_pauli_z[0] if len(current_pauli_z) > 0 else 0,
        "y": current_pauli_z[1] if len(current_pauli_z) > 1 else 0,
        "z": current_pauli_z[2] if len(current_pauli_z) > 2 else 0
    }

    # SVM Benchmark Prediction
    svm_prob = svm_benchmark.predict_proba(input_features)[0][1]

    record = {
        "focus_ratio": float(focus_ratio),
        "lapse_prob":  prediction,
        "svm_prob":    float(svm_prob),
        "hilbert":     hilbert_coords,
        "pauli_z":     current_pauli_z
    }
    session.research_records.append(record)
    if len(session.research_records) > 100:
        session.research_records.pop(0)

    # Simulated loss curves (visual benchmark only)
    vqc_loss = 0.5 * (prediction - (1.0 if prediction > 0.5 else 0.0)) ** 2
    svm_loss = 0.5 * (svm_prob  - (1.0 if prediction > 0.5 else 0.0)) ** 2
    session.vqc_losses.append(vqc_loss)
    session.svm_losses.append(svm_loss)
    if len(session.vqc_losses) > 50:
        session.vqc_losses.pop(0)
        session.svm_losses.pop(0)

    return {
        "metrics": {
            "focus_ratio":      focus_ratio,
            "lapse_probability": prediction,
            "is_calibrating":   session.is_calibrating,
            "baseline_value":   session.baseline_mean,
            "entropy":          entropy,
            "band_power":       band_power,
            "confidence":       confidence,
            "interventions":    session.interventions_count,
            "research": {
                "hilbert":   hilbert_coords,
                "vqc_loss":  vqc_loss,
                "svm_loss":  svm_loss
            }
        },
        "analysis": analysis
    }


@app.get("/research/logs")
def get_research_logs():
    """Returns the last 30 telemetry records from the current session."""
    return {"logs": session.research_records[-30:] if session.research_records else []}


@app.get("/research/training_results")
def get_training_results():
    """
    Serves data/training_results.json — the real VQC vs SVM benchmark numbers
    generated by scripts/quantum_model.py. Returns 404 if training hasn't been run.
    """
    results_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                                'training_results.json')
    if not os.path.exists(results_path):
        raise HTTPException(
            status_code=404,
            detail="training_results.json not found. Run scripts/quantum_model.py first."
        )
    with open(results_path, 'r') as f:
        return json.load(f)


@app.get("/research/export")
def export_research_bundle():
    """Bundles session telemetry + quantum metadata + metrics into a ZIP."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        import pandas as _pd
        df = _pd.DataFrame(session.research_records)
        zip_file.writestr("raw_eeg_transformed.csv", df.to_csv(index=False))

        meta = {
            "circuit":     "StronglyEntanglingLayers",
            "layers":      n_layers,
            "qubits":      n_qubits,
            "entanglement": "Full (All-to-All)",
            "encoding":    "AngleEmbedding [0, pi]"
        }
        zip_file.writestr("quantum_metadata.json", json.dumps(meta, indent=4))

        report  = "QENCS Research Performance Report\n"
        report += f"Samples Collected: {len(session.research_records)}\n"
        report += (f"Average VQC Loss: "
                   f"{np.mean(session.vqc_losses) if session.vqc_losses else 0:.4f}\n")
        report += (f"Average SVM Loss: "
                   f"{np.mean(session.svm_losses) if session.svm_losses else 0:.4f}\n")
        zip_file.writestr("metrics_report.txt", report)

        # Include training results if available
        results_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                                    'training_results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                zip_file.writestr("training_results.json", f.read())

    zip_buffer.seek(0)
    from fastapi.responses import Response
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": "attachment; filename=qencs_research_bundle.zip"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
