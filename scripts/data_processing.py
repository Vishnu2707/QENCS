
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def process_eeg_data(input_path, output_path):
    """
    Loads EEG data, calculates FocusRatio, normalizes features, and saves processed data.
    """
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    # Check for required columns
    required_columns = ['Theta', 'Beta1', 'Beta2', 'Delta', 'Alpha1', 'Alpha2', 'Gamma1', 'Gamma2', 'predefinedlabel']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing column '{col}' in input CSV.")
            return

    # 1. Handle Missing/Infinite Values
    print("Handling missing and infinite values...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True) 

    # 2. Calculate Focus Ratio (Theta / Beta)
    # Beta is commonly defined as the sum of Beta1 and Beta2 in this context if not specified otherwise,
    # or sometimes just Beta1. The plan specified (Beta1 + Beta2).
    print("Calculating FocusRatio = Theta / (Beta1 + Beta2)...")
    # Avoid division by zero
    beta_sum = df['Beta1'] + df['Beta2']
    df['FocusRatio'] = df['Theta'] / beta_sum.replace(0, np.nan)
    
    # Drop rows where FocusRatio calculation failed (if any)
    df.dropna(subset=['FocusRatio'], inplace=True)

    # 3. Normalize Features
    # We want to normalize the frequency bands and the calculated ratio
    feature_columns = ['Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2', 'FocusRatio']
    
    print("Normalizing features...")
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    # 4. Save Processed Data
    # Keep relevant columns + SubjectID/VideoID if needed for splitting later
    output_columns = ['SubjectID', 'VideoID'] + feature_columns + ['predefinedlabel', 'user-definedlabeln']
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving processed data to {output_path}...")
    df[output_columns].to_csv(output_path, index=False)
    print("Data processing complete.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(base_dir, 'data', 'EEG_data.csv')
    output_csv = os.path.join(base_dir, 'data', 'processed_eeg.csv')
    
    process_eeg_data(input_csv, output_csv)
