import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def add_realistic_environmental_noise(df):
    df = df.copy()
    
    # 1. Gaussian sensor noise (variable strength)
    sensor_noise_std = np.random.uniform(0.0005, 0.003)
    df['absorbance'] += np.random.normal(loc=0, scale=sensor_noise_std, size=len(df))
    
    # 2. Water vapor interference
    if np.random.rand() < 0.9:  # Usually present outdoors
        intensity = np.random.uniform(0.0008, 0.003)
        wn = df['wavenumber'].values
        peak1 = intensity * np.exp(-0.5 * ((wn - 3400) / 60) ** 2)
        peak2 = intensity * np.exp(-0.5 * ((wn - 1600) / 40) ** 2)
        df['absorbance'] += peak1 + peak2

    # 3. Dust attenuation
    if np.random.rand() < 0.8:  # Often present
        min_factor = np.random.uniform(0.80, 0.95)
        max_factor = np.random.uniform(min_factor, 1.0)
        attenuation = np.random.uniform(min_factor, max_factor, size=len(df))
        df['absorbance'] *= attenuation

    # 4. Baseline drift (e.g., due to power or angle fluctuation)
    if np.random.rand() < 0.6:  # Happens sometimes
        drift_amplitude = np.random.uniform(0.0005, 0.002)
        drift = drift_amplitude * np.sin(np.linspace(0, np.pi * 3, len(df)))
        df['absorbance'] += drift

    # 5. Clipping to keep values in valid range
    df['absorbance'] = np.clip(df['absorbance'], a_min=0, a_max=None)
    
    return df


