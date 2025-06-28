import pandas as pd
import numpy as np
import joblib
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

def interpolate_spectrum_to_target(df, target_wavenumbers):
    """
    Interpolates the given spectrum (df with 'wavenumber' and 'absorbance') to match target_wavenumbers.
    Returns an array of interpolated absorbance values.
    """
    f_interp = interp1d(df['wavenumber'], df['absorbance'], kind='linear', fill_value="extrapolate")
    absorbance_interp = f_interp(target_wavenumbers)
    return absorbance_interp

def predict_drug_proportions(csv_path):
    # Load model and scaler
    scaler = joblib.load("spectra_scaler.pkl")
    model = joblib.load("drug_regression_model.pkl")

    # Load input spectrum
    df = pd.read_csv(csv_path)

    # Load target wavenumbers (from any training file)
    reference_df = pd.read_csv("GeneratedMixtures/mix_000.csv")
    target_wavenumbers = reference_df['wavenumber'].values

    # Interpolate to 1000 features
    absorbance_interp = interpolate_spectrum_to_target(df, target_wavenumbers)
    absorbance_interp = absorbance_interp.reshape(1, -1)

    # Scale
    absorbance_scaled = scaler.transform(absorbance_interp)

    # Predict
    prediction = model.predict(absorbance_scaled)[0]
    heroin_pct, morphine_pct = prediction

    print(f"\n🔍 Prediction for: {csv_path}")
    print(f"Estimated Heroin proportion   : {heroin_pct:.4f}")
    print(f"Estimated Morphine proportion : {morphine_pct:.4f}\n")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    predict_drug_proportions("GeneratedMixtures/mix_1020.csv")
