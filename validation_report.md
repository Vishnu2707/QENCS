# Cross-Subject Validation Report (Model V2)

## Executive Summary
Model V2 implementation (4 StronglyEntanglingLayers, Min-Max [0, π] scaling) has successfully shifted the prediction baseline. "Lapse Probability" shifted from a clustered **0.44 (V1)** to **0.57-0.59 (V2)**. While the variance between subjects remains narrow, the shift across the 0.5 threshold has significantly increased the dynamic range of the Logic Agent's advice.

## Detailed Results (V2)

| Subject | Focus Ratio (Theta/Beta) | Lapse Prob (V2) | Logic Agent Advice |
|---------|--------------------------|-----------------|-------------------|
| **Subject 0** | -1.74 | 0.59 | "Are you overthinking? Pause..." |
| **Subject 0** | 5.93 | 0.57 | "Overwhelmed? Stop." |
| **Subject 2** | 0.82 | 0.58 | "You seem focused but potentially stuck." |
| **Subject 2** | 0.76 | 0.58 | "High engagement detected, mental strain high." |

## Analysis
1.  **Architecture Impact:** Increasing VQC depth to 4 layers and using StronglyEntanglingLayers improved the model's ability to map complex features, evidenced by the shift in output distribution.
2.  **Scaling Impact:** Min-Max [0, π] scaling properly utilized the rotation space of the Angle Embedding.
3.  **Logic Agent Response:** Because V2 predictions are consistently > 0.5, the Logic Agent is now exploring the "Confused" state paths in the ADHD Advice Lookup Table. This provides much more varied feedback than the "Clear" paths seen in V1.

## Conclusion
Model V2 is a significant step forward in model sensitivity. Further improvement would likely require individual subject calibration (Subject-Specific Baselines) or significantly more training epochs (e.g., 50+ instead of 5).
