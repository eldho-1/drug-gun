# temp_simulator.py

import numpy as np

BASE_TEMP = 25  # Reference temperature in °C

def adjust_absorbance(absorbance, temperature):
    """
    Adjusts absorbance values based on temperature.

    Parameters:
        absorbance (pd.Series): Original absorbance values.
        temperature (float): New temperature in Celsius.

    Returns:
        pd.Series: Modified absorbance values.
    """
    scale_factor = 1 + 0.0006 * (temperature - BASE_TEMP)
    thermal_noise = np.random.normal(0, 0.0005 * abs(temperature - BASE_TEMP), size=len(absorbance))
    return absorbance * scale_factor + thermal_noise
