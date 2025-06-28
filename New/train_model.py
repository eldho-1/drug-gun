import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# -----------------------------
# 1. Load Labels
# -----------------------------
labels_df = pd.read_csv("GeneratedMixtures/labels.csv")

all_spectra = []
all_labels = []

for idx, row in labels_df.iterrows():
    spectrum_path = row['spectrum_file'].replace("\\", "/")
    full_path = os.path.join("GeneratedMixtures", os.path.basename(spectrum_path))

    # Load spectrum file
    df = pd.read_csv(full_path)

    # Extract absorbance (ignore wavenumber)
    absorbance = df['absorbance'].values
    all_spectra.append(absorbance)

    # Get label [heroin, morphine]
    label = [row['heroin'], row['morphine']]
    all_labels.append(label)

# Convert to NumPy arrays
X = np.array(all_spectra)
y = np.array(all_labels)

print(f"Loaded {X.shape[0]} samples with {X.shape[1]} spectral features each.")

# -----------------------------
# 2. Standardize Spectral Features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "spectra_scaler.pkl")

# -----------------------------
# 3. Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# -----------------------------
# 4. Train Multi-Output Regressor
# -----------------------------
base_model = RandomForestRegressor(n_estimators=100, random_state=42)
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate the Model
# -----------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# -----------------------------
# 6. Save the Trained Model
# -----------------------------
joblib.dump(model, "drug_regression_model.pkl")
print("\n✅ Model and scaler saved successfully.")
