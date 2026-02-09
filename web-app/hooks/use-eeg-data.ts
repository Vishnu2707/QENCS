
import { useState, useEffect } from "react";

// Define Types based on Backend Response
interface EEGMetrics {
    focus_ratio: number;
    lapse_probability: number;
    is_calibrating: boolean;
    baseline_value: number | null;
    entropy: number;
    band_power: {
        theta: number;
        alpha: number;
        beta: number;
    };
    confidence: number;
    interventions: number;
}

interface LogicAnalysis {
    state: string;
    advice: string;
    metrics: {
        theta_beta_ratio: number;
        confusion_prob: number;
    };
}

interface APIResponse {
    metrics: EEGMetrics;
    analysis: LogicAnalysis;
}

export function useEEGData() {
    const [currentFocus, setCurrentFocus] = useState(0.8);
    const [advice, setAdvice] = useState("Initializing Quantum Logic Agent...");
    const [lapseProb, setLapseProb] = useState(0.1);
    const [status, setStatus] = useState<"connecting" | "active" | "error">("connecting");
    const [isCalibrating, setIsCalibrating] = useState(true);
    const [baselineValue, setBaselineValue] = useState<number | null>(null);

    // Phase 3 Metrics
    const [entropy, setEntropy] = useState(0);
    const [bandPower, setBandPower] = useState({ theta: 33, alpha: 33, beta: 34 });
    const [confidence, setConfidence] = useState(0);
    const [interventions, setInterventions] = useState(0);

    // Settings
    const [sensitivity, setSensitivity] = useState(0.15); // Default 15%

    useEffect(() => {
        const fetchData = async () => {
            try {
                // Mock Input for API (simulating raw EEG stream from headset)
                const mockInput = {
                    delta: Math.random(),
                    theta: Math.random() * 10,
                    alpha1: Math.random(),
                    alpha2: Math.random(),
                    beta1: Math.random() * 5 + 5, // Biased towards focus
                    beta2: Math.random() * 5 + 5,
                    gamma1: Math.random(),
                    gamma2: Math.random(),
                    sensitivity: sensitivity
                };
                const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
                if (status === 'connecting') {
                    console.log("Current API URL:", apiUrl);
                }

                const res = await fetch(`${apiUrl}/analyze`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(mockInput)
                });

                if (!res.ok) throw new Error("API Connection Failed");

                const data: APIResponse = await res.json();

                // State Updates
                setAdvice(data.analysis.advice);
                setLapseProb(data.metrics.lapse_probability);
                setIsCalibrating(data.metrics.is_calibrating);
                setBaselineValue(data.metrics.baseline_value);

                // Phase 3 Stats
                setEntropy(data.metrics.entropy);
                setBandPower(data.metrics.band_power);
                setConfidence(data.metrics.confidence);
                setInterventions(data.metrics.interventions);

                setCurrentFocus(1 - data.metrics.lapse_probability);
                setStatus("active");

            } catch (e) {
                console.warn("Backend Unreachable, using fallback simulation");
                setStatus("error");
                setAdvice("Check backend connection. Simulating quantum insights...");
                setIsCalibrating(false);
            }
        };

        const interval = setInterval(fetchData, 2000);
        return () => clearInterval(interval);
    }, [sensitivity]);

    return {
        currentFocus, advice, lapseProb, status,
        isCalibrating, baselineValue,
        entropy, bandPower, confidence, interventions,
        sensitivity, setSensitivity
    };
}
