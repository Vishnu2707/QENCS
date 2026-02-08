
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional

# Add scripts directory to path to import agents
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from logic_agent import LogicAgent

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logic_agent = LogicAgent()

# --- Quantum Model Setup V2 ---
import pennylane as qml
from torch import nn

n_qubits = 9
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

n_layers = 4
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
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Quantum Model V2 Loaded Successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")

# Scaler Setup
feature_scaler = MinMaxScaler(feature_range=(0, np.pi))
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_eeg.csv')
if os.path.exists(DATA_PATH):
    try:
        baseline_df = pd.read_csv(DATA_PATH)
        feature_cols = ['Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2', 'FocusRatio']
        feature_scaler.fit(baseline_df[feature_cols].values)
    except Exception as e:
        print(f"Scaler fit error: {e}")

# --- State Management ---
class SessionState:
    def __init__(self):
        self.baseline_buffer = []
        self.baseline_mean = None
        self.is_calibrating = True
        self.MAX_BASELINE_SAMPLES = 15
        self.interventions_count = 0

session = SessionState()

class EEGData(BaseModel):
    delta: float
    theta: float
    alpha1: float
    alpha2: float
    beta1: float
    beta2: float
    gamma1: float
    gamma2: float
    sensitivity: Optional[float] = 0.15

def calculate_entropy(psd: List[float]) -> float:
    """Calculates Shannon Entropy for EEG bands."""
    psd_norm = np.array(psd) / sum(psd)
    return -np.sum(psd_norm * np.log2(psd_norm + 1e-9))

@app.post("/analyze")
def analyze_focus(data: EEGData):
    # 1. Base Calculations
    total_beta = data.beta1 + data.beta2
    if total_beta == 0: total_beta = 0.001
    focus_ratio = data.theta / total_beta

    # 2. Quantum VQC Prediction
    input_features = np.array([
        data.delta, data.theta, data.alpha1, data.alpha2, 
        data.beta1, data.beta2, data.gamma1, data.gamma2, 
        focus_ratio
    ], dtype=np.float32).reshape(1, -1)
    
    scaled_features = feature_scaler.transform(input_features)
    tensor_input = torch.tensor(scaled_features, dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(tensor_input).item()
    
    # 3. Phase 3: Neural Analytics
    # Band Power Distribution
    total_power = data.theta + data.alpha1 + data.alpha2 + data.beta1 + data.beta2
    if total_power == 0: total_power = 0.001
    band_power = {
        "theta": (data.theta / total_power) * 100,
        "alpha": ((data.alpha1 + data.alpha2) / total_power) * 100,
        "beta": (total_beta / total_power) * 100
    }
    
    # Entropy
    entropy = calculate_entropy([data.delta, data.theta, data.alpha1 + data.alpha2, total_beta, data.gamma1 + data.gamma2])
    
    # Quantum Confidence (distance from boundary 0.5)
    confidence = abs(prediction - 0.5) * 2

    # 4. Calibration & Logic
    if session.is_calibrating:
        session.baseline_buffer.append(prediction)
        if len(session.baseline_buffer) >= session.MAX_BASELINE_SAMPLES:
            session.baseline_mean = sum(session.baseline_buffer) / len(session.baseline_buffer)
            session.is_calibrating = False

    analysis = logic_agent.analyze(
        focus_ratio, 
        prediction, 
        baseline_confusion=session.baseline_mean,
        sensitivity=data.sensitivity
    )
    
    # Track interventions (whenever state is not Focused & Clear)
    if "Confused" in analysis["state"] or "Distracted" in analysis["state"]:
        session.interventions_count += 1

    if session.is_calibrating:
        analysis["advice"] = "Calibrating your neural baseline..."

    return {
        "metrics": {
            "focus_ratio": focus_ratio,
            "lapse_probability": prediction,
            "is_calibrating": session.is_calibrating,
            "baseline_value": session.baseline_mean,
            "entropy": entropy,
            "band_power": band_power,
            "confidence": confidence,
            "interventions": session.interventions_count
        },
        "analysis": analysis
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
