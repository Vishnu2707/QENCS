# QENCS Technical Whitepaper & Manual

## 1. The Quantum Advantage: Variational Quantum Classifiers (VQC)

### why 4 Layers?
The QENCS system utilizes a 4-layer Variational Quantum Classifier (VQC) built with PennyLane. Increasing the depth to 4 `StronglyEntanglingLayers` allows the model to map input features into a higher-dimensional Hilbert space, providing significantly more "expressive power" than a shallow 2-layer model. 

### Quantum Entanglement & Non-Linearity
Standard classical models (like Logistic Regression or shallow SVMs) often struggle to find correlations between high-frequency EEG bands (Beta) and low-frequency bands (Theta) when they are subtly interleaved. 
- **Entanglement Capability**: By using `StronglyEntanglingLayers`, we induce quantum entanglement between the 9 qubits representing different EEG features. This allows the model to capture **multi-qubit correlations**.
- **Non-Linear Mapping**: Quantum circuits are inherently non-linear. The combination of Angle Encoding (rotations) and entangling gates creates a complex decision boundary that can detect "ADHD-lapse" patterns that are invisible to linear statistical methods.

## 2. System Flow: Data-to-Advice

The QENCS pipeline is an automated loop:
1.  **Preprocessing (Data Agent)**: Raw EEG (Delta, Theta, Alpha, Beta, Gamma) is normalized. A `FocusRatio` (Theta/Beta) is calculated.
2.  **Quantum Inference (Quantum Agent)**: Features are Min-Max scaled to $[0, \pi]$ and fed into the 4-layer VQC. The model outputs a `Lapse Probability`.
3.  **Baseline Comparison (Logic Agent)**: The current probability is compared against the user's previously calculated personal baseline.
4.  **Coaching Trigger**: If the probability exceeds the baseline by the user-defined sensitivity (10%, 15%, or 20%), a coaching intervención is triggered.

## 3. Developer Guide

### Starting the System
**Backend (FastAPI):**
```bash
cd backend
python3 main.py
```
**Frontend (Next.js):**
```bash
cd web-app
npm run dev
```

### Swapping the Live Stream Simulator for Real EEG
The current system uses a mock generator in `use-eeg-data.ts`. To connect a real device (e.g., Muse, OpenBCI):
1.  **LSL (Lab Streaming Layer)**: Install `pylsl` in the backend environment.
2.  **LSL Resolver**: In `backend/main.py`, replace the mock input with an LSL inlet that listens for the EEG stream.
3.  **Frequency Match**: Ensure the LSL sampling rate matches the 2s analysis window of the Logic Agent.

## 4. Environment Configuration

### Vercel (Frontend)
- `NEXT_PUBLIC_BACKEND_URL`: `https://your-backend-on-render.com`

### Render (Backend)
- `PORT`: `8000`
- `PYTHON_VERSION`: `3.9`
