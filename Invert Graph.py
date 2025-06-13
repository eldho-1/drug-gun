import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load your CSV file
df = pd.read_csv("compress_sucrose.csv")

# Rename columns properly
df.columns = ['wavelength', 'absorbance', 'drug_type']

# Invert absorbance (if needed — this step depends on your context)
df['inverted_absorbance'] = 1 - df['absorbance']

# Normalize the inverted absorbance to 0–1 range
scaler = MinMaxScaler()
df['normalized_absorbance'] = scaler.fit_transform(df[['inverted_absorbance']])

# Plot the original vs corrected absorbance
plt.figure(figsize=(10, 6))
plt.plot(df['wavelength'], df['absorbance'], label='Original', alpha=0.5)
plt.plot(df['wavelength'], df['normalized_absorbance'], label='Corrected & Normalized', linewidth=2)
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Absorbance')
plt.title(f'Corrected IR Spectrum - {df["drug_type"].iloc[0]}')
plt.legend()
plt.gca().invert_xaxis()  # IR spectra usually display wavenumber decreasing
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Save the corrected spectrum
df[['wavelength', 'normalized_absorbance', 'drug_type']].to_csv("correct_methadone.csv", index=False)
