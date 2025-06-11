import numpy as np
import pandas as pd

# JCAMP-DX parameters
firstx = 550.0  # First wavenumber
deltax = 4.0    # Wavenumber step
yfactor = 0.000003300  # Multiply Y-values by this

# Extract all Y-values (ignore metadata lines)
y_values = []
with open("methadone.jdx") as f:
    for line in f:
        if line.startswith(("##", "$$")): continue
        parts = line.strip().split()
        if not parts: continue
        
        # Skip wavenumber labels (e.g., "550.0")
        for val in parts[1:]:  
            try:
                y_values.append(float(val) * yfactor)
            except ValueError:
                continue

# Generate wavenumbers
wavenumbers = np.arange(firstx, firstx + len(y_values)*deltax, deltax)

# Create DataFrame
df = pd.DataFrame({
    "wavelength": wavenumbers,
    "absorbance": y_values,
    "drug_type": "Methadone"
})

# Save to CSV
df.to_csv("methadone.csv", index=False)
print("Saved to methadone.csv")