
import pandas as pd
import numpy as np
import os

def process_eeg_data(input_path, output_path):
    """
    Loads raw EEG band-power data, computes FocusRatio, drops invalid rows,
    and saves cleaned (unscaled) data.

    Scaling decision: StandardScaler has been intentionally removed.
    The MinMaxScaler([0, pi]) required for AngleEmbedding is fitted exclusively
    on the training split inside quantum_model.py and saved to
    data/feature_scaler.pkl so that inference reuses the identical transform.
    Applying a second scaler here would double-transform the data.
    """
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    # Check for required columns
    required_columns = ['Theta', 'Beta1', 'Beta2', 'Delta', 'Alpha1', 'Alpha2',
                        'Gamma1', 'Gamma2', 'predefinedlabel']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing column '{col}' in input CSV.")
            return

    # 1. Handle Missing/Infinite Values
    print("Handling missing and infinite values...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 2. Calculate FocusRatio = Theta / (Beta1 + Beta2)
    print("Calculating FocusRatio = Theta / (Beta1 + Beta2)...")
    beta_sum = df['Beta1'] + df['Beta2']
    df['FocusRatio'] = df['Theta'] / beta_sum.replace(0, np.nan)
    df.dropna(subset=['FocusRatio'], inplace=True)

    # 3. No normalization here.
    # Raw band-power values are preserved so that quantum_model.py can fit
    # a single MinMaxScaler([0, pi]) on the training split only and persist
    # it to data/feature_scaler.pkl for use at inference time.

    # 4. Save Processed Data
    feature_columns = ['Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2',
                       'Gamma1', 'Gamma2', 'FocusRatio']
    output_columns = (['SubjectID', 'VideoID'] + feature_columns +
                      ['predefinedlabel', 'user-definedlabeln'])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving processed data to {output_path}...")
    df[output_columns].to_csv(output_path, index=False)
    print(f"Data processing complete. {len(df)} rows saved.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(base_dir, 'data', 'EEG_data.csv')
    output_csv = os.path.join(base_dir, 'data', 'processed_eeg.csv')
    process_eeg_data(input_csv, output_csv)
