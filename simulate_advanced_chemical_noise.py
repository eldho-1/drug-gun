
import pandas as pd
import numpy as np
import os

def add_advanced_chemical_noise(input_csv, output_csv=None,
                                 noise_level=0.002,
                                 peak_intensity=(0.002, 0.01),
                                 num_peaks=5,
                                 apply_shift=True,
                                 apply_dropout=True,
                                 apply_clipping=True,
                                 apply_warp=True,
                                 apply_environment=True):
    # Load pure spectrum
    df = pd.read_csv(input_csv)
    if 'wavenumber' not in df.columns or 'absorbance' not in df.columns:
        raise ValueError("CSV must contain 'wavenumber' and 'absorbance' columns.")

    noisy_df = df.copy()

    # Gaussian noise
    np.random.seed(42)
    gaussian_noise = np.random.normal(0, noise_level, len(df))
    noisy_df['absorbance'] += gaussian_noise

    # Baseline drift
    baseline_drift = 0.002 * np.sin(np.linspace(0, np.pi, len(df)))
    noisy_df['absorbance'] += baseline_drift

    # Multiplicative scatter
    scatter = np.linspace(0.95, 1.05, len(df))
    noisy_df['absorbance'] *= scatter

    # Impurity peaks (spikes)
    spike_indices = np.random.choice(len(df), size=num_peaks, replace=False)
    for i in spike_indices:
        noisy_df.at[i, 'absorbance'] += np.random.uniform(*peak_intensity)

    # Spectral shift
    #if apply_shift:
       # shift = np.random.choice([-2, -1, 0, 1, 2])
        #noisy_df['absorbance'] = np.roll(noisy_df['absorbance'], shift)

    #Flat dropout region
    if apply_dropout:
        drop_start = np.random.randint(20, len(df) - 30)
        noisy_df.loc[drop_start:drop_start + 10, 'absorbance'] = 0

    # Absorbance clipping
    if apply_clipping:
        noisy_df['absorbance'] = np.clip(noisy_df['absorbance'], None, 1.5)

    # Polynomial warping
    if apply_warp:
        x = np.linspace(-1, 1, len(df))
        poly_curve = 0.001 * (x**3 - 0.5 * x**2)
        noisy_df['absorbance'] += poly_curve

    # Environmental CO₂ peak at ~2350 cm⁻¹
    if apply_environment:
        env_index = np.argmin(np.abs(df['wavenumber'] - 2350))
        if 0 <= env_index < len(df):
            noisy_df.at[env_index, 'absorbance'] += 0.005

    # Output file name
    if not output_csv:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_noisy.csv"

    noisy_df.to_csv(output_csv, index=False)
    print(f"✅ Noisy spectrum saved to: {output_csv}")


