import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# ========= Setup =========
DRUGS = {
    "heroin": "Training Data/heroin.csv",
    "morphine": "Training Data/morphine.csv"
}

ADULTERANTS = {
    "sucrose": "Training Data/sucrose.csv",
    "citric_acid": "Training Data/citric.csv"
}

OUTPUT_FOLDER = "GeneratedMixtures"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

N_SAMPLES_PER_PAIR = 500  # Total 4 pairs × 125 = 500 samples
N_POINTS = 1000

# ========= Baseline Correction =========
def baseline_correct(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = spdiags([np.ones(L), -2*np.ones(L), np.ones(L)], [0, -1, -2], L, L)
    w = np.ones(L)
    for _ in range(niter):
        W = spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return y - z

# ========= Load and Preprocess Spectrum =========
def load_and_process(filepath, target_wn):
    df = pd.read_csv(filepath)
    wn = df['wavenumber'].values
    absorb = df['absorbance'].values
    interp = interp1d(wn, absorb, kind='linear', fill_value='extrapolate')
    resampled = interp(target_wn)
    corrected = baseline_correct(resampled)
    return corrected

# ========= Determine Common Wavenumber Range =========
all_wns = []
for f in list(DRUGS.values()) + list(ADULTERANTS.values()):
    df = pd.read_csv(f)
    all_wns.append(df['wavenumber'].values)

wn_min = max([wn.min() for wn in all_wns])
wn_max = min([wn.max() for wn in all_wns])
common_wavenumbers = np.linspace(wn_min, wn_max, N_POINTS)

# ========= Generate Mixtures =========
labels = []
counter = 0

for drug_name, drug_file in DRUGS.items():
    for ad_name, ad_file in ADULTERANTS.items():
        for i in range(N_SAMPLES_PER_PAIR):
            # Random drug weight between 0.05 and 0.95
            w_drug = np.round(np.random.uniform(0.05, 0.95), 2)
            w_ad = np.round(1.0 - w_drug, 2)

            # Load and mix
            drug_spec = load_and_process(drug_file, common_wavenumbers)
            ad_spec = load_and_process(ad_file, common_wavenumbers)
            mix = w_drug * drug_spec + w_ad * ad_spec

            # Save mixture
            mix_df = pd.DataFrame({
                "wavenumber": common_wavenumbers,
                "absorbance": mix
            })
            mix_filename = f"mix_{counter:03d}.csv"
            mix_df.to_csv(os.path.join(OUTPUT_FOLDER, mix_filename), index=False)

            # Save label: only drug weights
            label_row = {
                "spectrum_file": os.path.join(OUTPUT_FOLDER, mix_filename),
                "heroin": w_drug if drug_name == "heroin" else 0.0,
                "morphine": w_drug if drug_name == "morphine" else 0.0
            }
            labels.append(label_row)
            counter += 1

print(f"✅ Generated {counter} mixtures.")

# ========= Save Labels =========
labels_df = pd.DataFrame(labels)
labels_df.to_csv(os.path.join(OUTPUT_FOLDER, "labels.csv"), index=False)
print(f"📄 Saved labels to {os.path.join(OUTPUT_FOLDER, 'labels.csv')}")
