
import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import time
from sklearn.preprocessing import MinMaxScaler

def train_quantum_model_v2(data_path, model_save_path):
    print(f"Loading processed data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return

    # Prepare Data
    feature_cols = ['Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2', 'FocusRatio']
    target_col = 'predefinedlabel'

    X = df[feature_cols].values
    y = df[target_col].values

    # 1. Quantum Feature Scaling: Min-Max to [0, pi]
    # This is crucial for Angle Encoding to avoid periodicity issues and ensure distinct mappings.
    print("Applying Min-Max scaling [0, pi] for Quantum Feature Map...")
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X)

    # Convert to Tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    # Split Data (Simple 80/20 split)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]

    print(f"Training with {len(X_train)} samples, Testing with {len(X_test)} samples.")

    # --- Quantum Model Definition V2 ---
    n_qubits = len(feature_cols) 
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        # StronglyEntanglingLayers provides more complexity than BasicEntangler
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    n_layers = 4 # Increased capacity
    # StronglyEntanglingLayers weight shape: (n_layers, n_qubits, 3)
    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    class HybridModelV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_layer = qlayer
            self.fc = nn.Linear(n_qubits, 1)
            # Use Sigmoid for probability output if not using BCEWithLogitsLoss
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.q_layer(x)
            x = self.fc(x)
            x = self.sigmoid(x)
            return x

    model = HybridModelV2()
    
    # --- Training Loop with Weighted Loss ---
    # Weight the positive class (Confusion = 1) higher to improve sensitivity
    # Ratio: 2.0 to penalize misses on confusion more.
    pos_weight = torch.tensor([2.0])
    # BCELoss doesn't take pos_weight directly like BCEWithLogitsLoss, 
    # so we'll use a manually weighted BCE if needed, or stick to a simpler method.
    # Actually, let's use BCEWithLogitsLoss for better stability if we remove self.sigmoid
    
    class HybridModelV2Logits(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_layer = qlayer
            self.fc = nn.Linear(n_qubits, 1)

        def forward(self, x):
            x = self.q_layer(x)
            x = self.fc(x)
            return x

    model_logits = HybridModelV2Logits()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model_logits.parameters(), lr=0.01)
    epochs = 5

    print("Starting training V2...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model_logits.train()
        optimizer.zero_grad()
        outputs = model_logits(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        model_logits.eval()
        with torch.no_grad():
            test_outputs = model_logits(X_test)
            test_loss = criterion(test_outputs, y_test)
            probs = torch.sigmoid(test_outputs)
            predicted = (probs > 0.5).float()
            accuracy = (predicted == y_test).float().mean()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Test Acc: {accuracy.item():.4f}")

    print(f"Training finished in {time.time() - start_time:.2f}s")

    # Save Model (We'll save the state dict of the logits version but can use it in the sigmoid version for inference)
    torch.save(model_logits.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_path = os.path.join(base_dir, 'data', 'processed_eeg.csv')
    model_save_path = os.path.join(base_dir, 'data', 'quantum_focus_model_v2.pth')
    
    train_quantum_model_v2(processed_data_path, model_save_path)
