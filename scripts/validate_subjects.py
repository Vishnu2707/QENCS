
import pandas as pd
import requests
import time
import json
import os

# Configuration
API_URL = "http://localhost:8000/analyze"
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_eeg.csv')
REPORT_PATH = os.path.join(os.path.dirname(__file__), '..', 'validation_report.md')

def run_validation():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Select Test Subjects
    # Subject 0 = Expected Stable/Flow 
    # Subject 2 = Expected Confusion/Distraction
    
    subject_0 = df[df['SubjectID'] == 0.0].head(5)
    subject_2 = df[df['SubjectID'] == 2.0].head(5)
    
    results = []

    print("\nStarting Validation Loop...")
    
    # Test Subject 0
    print("Testing Subject 0 (Stable)...")
    for _, row in subject_0.iterrows():
        payload = {
            "delta": row['Delta'],
            "theta": row['Theta'],
            "alpha1": row['Alpha1'],
            "alpha2": row['Alpha2'],
            "beta1": row['Beta1'],
            "beta2": row['Beta2'],
            "gamma1": row['Gamma1'],
            "gamma2": row['Gamma2']
        }
        
        try:
            res = requests.post(API_URL, json=payload)
            if res.status_code == 200:
                data = res.json()
                results.append({
                    "Subject": "Subject 0",
                    "Focus Ratio": data['metrics']['focus_ratio'],
                    "Lapse Prob": data['metrics']['lapse_probability'],
                    "Advice": data['analysis']['advice']
                })
            else:
                print(f"API Error: {res.status_code}")
        except Exception as e:
            print(f"Request Failed: {e}")

    # Test Subject 2
    print("Testing Subject 2 (High Confusion)...")
    for _, row in subject_2.iterrows():
        payload = {
            "delta": row['Delta'],
            "theta": row['Theta'],
            "alpha1": row['Alpha1'],
            "alpha2": row['Alpha2'],
            "beta1": row['Beta1'],
            "beta2": row['Beta2'],
            "gamma1": row['Gamma1'],
            "gamma2": row['Gamma2']
        }
        
        try:
            res = requests.post(API_URL, json=payload)
            if res.status_code == 200:
                data = res.json()
                results.append({
                    "Subject": "Subject 2",
                    "Focus Ratio": data['metrics']['focus_ratio'],
                    "Lapse Prob": data['metrics']['lapse_probability'],
                    "Advice": data['analysis']['advice']
                })
            else:
                print(f"API Error: {res.status_code}")
        except Exception as e:
            print(f"Request Failed: {e}")

    # Generate Report
    print(f"\nGenerating Report at {REPORT_PATH}...")
    with open(REPORT_PATH, "w") as f:
        f.write("# Cross-Subject Validation Report\n\n")
        f.write("Validation of Quantum Model and Logic Agent response to different Subject profiles.\n\n")
        
        f.write("## Test Subjects\n")
        f.write("- **Subject 0:** Expected Stable/Flow state.\n")
        f.write("- **Subject 2:** Expected Confusion/Distraction state.\n\n")
        
        f.write("## Results\n\n")
        f.write("| Subject | Focus Ratio | Lapse Prob | Logic Agent Advice |\n")
        f.write("|---------|-------------|------------|-------------------|\n")
        
        for r in results:
            advice_short = r['Advice'].split('.')[0] + "."
            f.write(f"| {r['Subject']} | {r['Focus Ratio']:.2f} | {r['Lapse Prob']:.2f} | {advice_short} |\n")
            
        f.write("\n## Conclusion\n")
        f.write("Check if Subject 2 triggered 'ADHD Overload' tips (e.g., 'Overwhelmed?', 'High distraction') compared to Subject 0's 'Flow' tips.\n")

    print("Validation Complete.")

if __name__ == "__main__":
    run_validation()
