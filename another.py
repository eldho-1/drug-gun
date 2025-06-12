import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from scipy.signal import savgol_filter
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# 1. Load and preprocess data (baseline correction, smoothing, etc.)
def preprocess_spectrum(df):
    df = df.copy()
    # Baseline correction (subtract 5th percentile)
    baseline = df['absorbance'].quantile(0.05)
    df['absorbance_corrected'] = df['absorbance'] - baseline
    
    # Normalization (0 to 1)
    min_abs = df['absorbance_corrected'].min()
    max_abs = df['absorbance_corrected'].max()
    df['absorbance_normalized'] = (df['absorbance_corrected'] - min_abs) / (max_abs - min_abs)
    
    # Smoothing (Savitzky-Golay filter)
    df['absorbance_smoothed'] = savgol_filter(df['absorbance_normalized'], window_length=11, polyorder=2)
    
    # First derivative (highlights peaks)
    df['absorbance_derivative'] = np.gradient(df['absorbance_smoothed'])
    
    return df

# 2. Train the model
def train_model(cocaine_file, heroin_file, methadone_file):
    # Load and label data
    cocaine_df = pd.read_csv(cocaine_file).assign(drug_type='cocaine')
    heroin_df = pd.read_csv(heroin_file).assign(drug_type='heroin')
    methadone_df = pd.read_csv(methadone_file).assign(drug_type='methadone')
    
    # Combine and preprocess
    combined_df = pd.concat([cocaine_df, heroin_df, methadone_df])
    processed_df = preprocess_spectrum(combined_df)
    
    # Features (X) = Wavelength + Absorbance | Label (y) = Drug Type
    X = processed_df[['wavelength', 'absorbance_derivative']].values
    y = processed_df['drug_type'].values
    
    # Encode labels (cocaine → 0, heroin → 1, methadone → 2)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train-test split (stratified to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Model pipeline (SMOTE + Scaler + Classifier)
    model = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),  # Only applied on training data
        ('classifier', RandomForestClassifier(n_estimators=100))  # Default best params
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Test Accuracy:", model.score(X_test, y_test))
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save model and label encoder
    joblib.dump(model, 'drug_classifier.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("Model saved as 'drug_classifier.pkl'")
    
    return model, le

# 3. Predict new spectra (FIXED)
def predict_new_spectrum(new_csv_file, model, le):
    # Load new data (ensure it has 'wavelength' and 'absorbance' columns)
    new_df = pd.read_csv(new_csv_file)
    
    # Preprocess EXACTLY like training data
    processed_df = preprocess_spectrum(new_df)
    X_new = processed_df[['wavelength', 'absorbance_derivative']].values
    
    # Predict probabilities for each class (across the entire spectrum)
    proba = model.predict_proba(X_new)
    
    # Aggregate predictions (mean probability across all wavelengths)
    mean_proba = np.mean(proba, axis=0)
    
    # Get the class with highest mean probability
    predicted_class_idx = np.argmax(mean_proba)
    predicted_class = le.inverse_transform([predicted_class_idx])[0]
    
    print(f"\nPredicted Drug: {predicted_class}")
    print("Confidence Scores:")
    for drug, score in zip(le.classes_, mean_proba):
        print(f"{drug}: {score:.2f}")
    
    return predicted_class

# Example usage
if __name__ == "__main__":
    # Step 1: Train model
    model, le = train_model(
        cocaine_file="cocaine.csv",
        heroin_file="heroin.csv",
        methadone_file="methadone.csv"
    )
    
    # Step 2: Predict a new spectrum
    predict_new_spectrum("methadone .csv", model, le)
