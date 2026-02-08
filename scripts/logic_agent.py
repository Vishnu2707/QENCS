
class LogicAgent:
    def __init__(self):
        # ADHD Advice Lookup Table
        # Key: (FocusState, ConfusionState)
        # FocusState: "Focused" (Theta/Beta < Threshold) or "Distracted" (Theta/Beta > Threshold)
        # ConfusionState: "Clear" (Model Prob < 0.5) or "Confused" (Model Prob > 0.5)
        self.advice_table = {
            ("Focused", "Clear"): [
                "You're in the Flow! Keep maintaining this rhythm.",
                "Great focus. Remember to blink and hydrate.",
                "Optimal state detected. Push through your current task."
            ],
            ("Focused", "Confused"): [
                "You seem focused but potentially stuck. Try breaking the problem down.",
                "High engagement detected, but mental strain is high. Take a deep breath.",
                "Are you overthinking? Pause for 10 seconds and reset."
            ],
            ("Distracted", "Clear"): [
                "Mind wandering? Let's bring it back. What's the very next step?",
                "Daydreaming detected. Stand up and stretch for 1 minute.",
                "Focus drift. Try the 5-4-3-2-1 grounding technique."
            ],
            ("Distracted", "Confused"): [
                "Overwhelmed? Stop. Write down your immediate goal.",
                "High distraction and confusion. Time for a 5-minute break.",
                "Sensory overload detected. Try reducing background noise or dimming lights."
            ]
        }
        
        # Thresholds
        self.focus_ratio_threshold = 1.5 # Normal Beta/Theta is often ~1/2. High Ratio = Distracted? 
        # Actually standard: Theta/Beta ratio. 
        # High Theta (drowsy/daydream) / Low Beta (active) = High Ratio -> Distracted/ADHD.
        # Low Theta / High Beta = Low Ratio -> Focused.
        # Let's set a heuristic threshold. 
        self.theta_beta_threshold = 2.0 

    def analyze(self, theta_beta_ratio, confusion_probability, baseline_confusion=None, sensitivity=0.15):
        """
        Determines the state and returns appropriate advice.
        """
        # Determine Focus State
        if theta_beta_ratio > self.theta_beta_threshold:
            focus_state = "Distracted"
        else:
            focus_state = "Focused"

        # Determine Confusion State (from model prediction)
        # Assuming model outputs probability of "Lapse/Confusion"
        # Phase 2/3: Dynamic Baseline. If baseline exists, use baseline + sensitivity as threshold.
        # Otherwise fallback to 0.5.
        threshold = 0.5
        if baseline_confusion is not None:
            threshold = baseline_confusion + sensitivity
        
        if confusion_probability > threshold:
            confusion_state = "Confused"
        else:
            confusion_state = "Clear"

        # Lookup Advice
        key = (focus_state, confusion_state)
        possible_advice = self.advice_table.get(key, ["Stay mindful."])
        
        # For now, just return the first one, or random if we imported random
        import random
        selected_advice = random.choice(possible_advice)

        return {
            "state": f"{focus_state} & {confusion_state}",
            "advice": selected_advice,
            "metrics": {
                "theta_beta_ratio": theta_beta_ratio,
                "confusion_prob": confusion_probability
            }
        }

if __name__ == "__main__":
    # Test
    agent = LogicAgent()
    print(agent.analyze(1.2, 0.2)) # Focused, Clear
    print(agent.analyze(2.5, 0.8)) # Distracted, Confused
