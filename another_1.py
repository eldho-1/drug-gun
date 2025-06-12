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

# Drug information database
DRUG_INFO = {
    'cocaine': {
        'class': 'Stimulant',
        'effects': 'Euphoria, increased energy, mental alertness',
        'risks': 'Heart attack, stroke, seizures, addiction'
    },
    'heroin': {
        'class': 'Opioid',
        'effects': 'Euphoria, pain relief, drowsiness',
        'risks': 'Respiratory depression, overdose, addiction'
    },
    'methadone': {
        'class': 'Synthetic opioid',
        'effects': 'Pain relief, prevention of opioid withdrawal symptoms',
        'risks': 'Respiratory depression, heart problems, addiction'
    }
}

# 1. Load and preprocess spectrum (same as before)
def preprocess_spectrum(df):
    df = df.copy()
    # Baseline correction (subtract 5th percentile)
    baseline = df['absorbance'].quantile( 0.05)
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

# 2. Train the model with drug and non-drug samples
def train_model(drug_files, non_drug_files):
    # Load and label drug data
    drug_dfs = []
    for drug_type, file_path in drug_files.items():
        df = pd.read_csv(file_path).assign(drug_type=drug_type, is_drug=True)
        drug_dfs.append(df)
    
    # Load and label non-drug data
    non_drug_dfs = []
    for i, file_path in enumerate(non_drug_files):
        df = pd.read_csv(file_path).assign(drug_type=f'non_drug_{i}', is_drug=False)
        non_drug_dfs.append(df)
    
    # Combine all data
    combined_df = pd.concat(drug_dfs + non_drug_dfs)
    processed_df = preprocess_spectrum(combined_df)
    
    # Features (X) = Wavelength + Absorbance | Labels (y) = is_drug (binary) and drug_type (multiclass)
    X = processed_df[['wavelength', 'absorbance_derivative']].values
    y_is_drug = processed_df['is_drug'].values
    y_drug_type = processed_df['drug_type'].values
    
    # Encode labels
    le_drug_type = LabelEncoder()
    y_drug_type_encoded = le_drug_type.fit_transform(y_drug_type)
    
    # Train-test split (stratified to keep class balance)
    X_train, X_test, y_train_is_drug, y_test_is_drug, y_train_drug, y_test_drug = train_test_split(
        X, y_is_drug, y_drug_type_encoded, test_size=0.2, random_state=42, stratify=y_is_drug
    )
    
    # Binary classifier (drug vs non-drug)
    binary_model = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100))
    ])
    binary_model.fit(X_train, y_train_is_drug)
    
    # Multiclass classifier (only for drug samples)
    drug_samples = processed_df[processed_df['is_drug']]
    X_drug = drug_samples[['wavelength', 'absorbance_derivative']].values
    y_drug = le_drug_type.transform(drug_samples['drug_type'])
    
    X_train_drug, X_test_drug, y_train_drug, y_test_drug = train_test_split(
        X_drug, y_drug, test_size=0.2, random_state=42, stratify=y_drug
    )
    
    multiclass_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100))
    ])
    multiclass_model.fit(X_train_drug, y_train_drug)
    
    # Evaluate models
    print("\nBinary Classification (Drug vs Non-Drug):")
    print("Test Accuracy:", binary_model.score(X_test, y_test_is_drug))
    print(classification_report(y_test_is_drug, binary_model.predict(X_test), target_names=['Non-Drug', 'Drug']))
    
    print("\nMulticlass Classification (Drug Types):")
    print("Test Accuracy:", multiclass_model.score(X_test_drug, y_test_drug))
    print(classification_report(y_test_drug, multiclass_model.predict(X_test_drug), target_names=le_drug_type.classes_))
    
    # Save models and label encoder
    joblib.dump(binary_model, 'drug_detection.pkl')
    joblib.dump(multiclass_model, 'drug_classification.pkl')
    joblib.dump(le_drug_type, 'drug_label_encoder.pkl')
    print("\nModels saved successfully.")
    
    return binary_model, multiclass_model, le_drug_type

# 3. Predict new spectrum with drug information
def predict_new_spectrum(new_csv_file, binary_model, multiclass_model, le_drug_type):
    # Load and preprocess new data
    new_df = pd.read_csv(new_csv_file)
    processed_df = preprocess_spectrum(new_df)
    X_new = processed_df[['wavelength', 'absorbance_derivative']].values
    
    # First check if it's a drug
    is_drug_proba = binary_model.predict_proba(X_new)[:, 1]  # Probability of being a drug
    mean_is_drug = np.mean(is_drug_proba)
    
    if mean_is_drug > 0.5:  # Threshold can be adjusted
        print("\nDrug detected!")
        print(f"Confidence: {mean_is_drug:.2%}")
        
        # Classify drug type
        drug_proba = multiclass_model.predict_proba(X_new)
        mean_drug_proba = np.mean(drug_proba, axis=0)
        predicted_idx = np.argmax(mean_drug_proba)
        predicted_drug = le_drug_type.inverse_transform([predicted_idx])[0]
        
        # Get drug information
        drug_data = DRUG_INFO.get(predicted_drug, {})
        
        print(f"\nPredicted Drug: {predicted_drug}")
        print(f"Class: {drug_data.get('class', 'Unknown')}")
        print(f"Effects: {drug_data.get('effects', 'Unknown')}")
        print(f"Risks: {drug_data.get('risks', 'Unknown')}")
        print("\nConfidence Scores for Drug Types:")
        for drug, score in zip(le_drug_type.classes_, mean_drug_proba):
            if drug in DRUG_INFO:  # Only show actual drugs, not non-drug classes
                print(f"{drug}: {score:.2f}")
        
        return {'is_drug': True, 'drug_type': predicted_drug, 'info': drug_data}
    else:
        print("\nNo drug detected.")
        print(f"Confidence: {(1 - mean_is_drug):.2%}")
        return {'is_drug': False}

# Example usage
if __name__ == "__main__":
    # Define your data files
    drug_files = {
        'cocaine': "cocaine.csv",
        'heroin': "heroin.csv",
        'methadone': "methadone.csv"
    }
    non_drug_files = [
        "sugar.csv",
        "flour.csv",
        "salt.csv"
    ]
    
    # Train models
    binary_model, multiclass_model, le_drug_type = train_model(drug_files, non_drug_files)
    
    # Test prediction
    test_file = "methadone.csv"  # Replace with your test file
    result = predict_new_spectrum(test_file, binary_model, multiclass_model, le_drug_type)