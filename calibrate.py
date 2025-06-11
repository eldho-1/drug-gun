import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import savgol_filter
from imblearn.over_sampling import SMOTE
from collections import Counter

# 1. Load and Preprocess Data
def load_and_preprocess(filepaths):
    dfs = []
    for filepath, drug_type in filepaths.items():
        df = pd.read_csv(filepath)
        df['drug_type'] = drug_type  # Add drug type label
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Basic preprocessing
    combined_df.dropna(inplace=True)
    
    # Baseline correction
    combined_df['absorbance_corrected'] = combined_df.groupby('drug_type')['absorbance'].transform(
        lambda x: x - x.quantile(0.05))
    
    # Normalization (per spectrum)
    combined_df['absorbance_normalized'] = combined_df.groupby('drug_type')['absorbance_corrected'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    
    # Smoothing
    window_size = 11
    poly_order = 2
    combined_df['absorbance_smoothed'] = combined_df.groupby('drug_type')['absorbance_normalized'].transform(
        lambda x: savgol_filter(x, window_size, poly_order))
    
    # First derivative
    combined_df['absorbance_derivative'] = combined_df.groupby('drug_type')['absorbance_smoothed'].transform(
        lambda x: np.gradient(x))
    
    # Standardization
    scaler = StandardScaler()
    combined_df['absorbance_standardized'] = combined_df.groupby('drug_type')['absorbance_derivative'].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())
    
    return combined_df

# 2. Feature Selection
def select_features(df):
    # Define characteristic peaks for each drug type
    common_peaks = [
        (600, 800),    # Common region 1
        (900, 1100),   # Common region 2
        (1200, 1400),  # Common region 3
        (1600, 1800)   # Common region 4
    ]
    
    # Drug-specific peaks (example - adjust based on your actual data)
    drug_peaks = {
        'cocaine': [(700, 720), (1260, 1280)],
        'heroin': [(750, 770), (1300, 1320)],
        'fentanyl': [(680, 700), (1240, 1260)]
    }
    
    selected_data = pd.DataFrame()
    
    # Select common peaks
    for region in common_peaks:
        region_data = df[(df['wavelength'] >= region[0]) & (df['wavelength'] <= region[1])]
        selected_data = pd.concat([selected_data, region_data])
    
    # Select drug-specific peaks
    for drug, peaks in drug_peaks.items():
        for region in peaks:
            region_data = df[(df['wavelength'] >= region[0]) & 
                           (df['wavelength'] <= region[1]) & 
                           (df['drug_type'] == drug)]
            selected_data = pd.concat([selected_data, region_data])
    
    return selected_data.drop_duplicates()

# 3. Data Augmentation
def augment_data(df, n_augment=3):
    augmented_dfs = []
    drugs = df['drug_type'].unique()
    
    for drug in drugs:
        drug_df = df[df['drug_type'] == drug]
        wavelengths = drug_df['wavelength'].unique()
        
        for _ in range(n_augment):
            noise_factor = np.random.uniform(0.005, 0.02)
            shift_factor = np.random.uniform(0.1, 0.5)
            scale_factor = np.random.uniform(0.9, 1.1)
            
            augmented_abs = drug_df['absorbance_standardized'].values * scale_factor + \
                          noise_factor * np.random.normal(size=len(drug_df))
            
            augmented_df = drug_df.copy()
            augmented_df['absorbance_standardized'] = augmented_abs
            augmented_df['wavelength'] = drug_df['wavelength'] + np.random.uniform(-shift_factor, shift_factor)
            
            augmented_dfs.append(augmented_df)
    
    return pd.concat([df] + augmented_dfs)

# 4. Modeling Pipeline
def build_model(X_train, y_train):
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('classifier', SVC(probability=True))
    ])
    
    # Parameters for grid search
    params = [
        {
            'pca__n_components': [1, 2],
            'classifier': [SVC()],
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto']
        },
        {
            'pca__n_components': [10, 20, 30],
            'classifier': [RandomForestClassifier()],
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20]
        }
    ]
    
    # Use stratified k-fold for imbalanced data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid = GridSearchCV(pipeline, params, cv=cv, n_jobs=-1, scoring='f1_weighted')
    grid.fit(X_train, y_train)
    
    return grid

# Main Execution
if __name__ == "__main__":
    # File paths for each drug type
    filepaths = {
        'cocaine.csv': 'cocaine',
        'heroin.csv': 'heroin',
        'methadone.csv': 'methadone'
    }
    
    # 1. Load and preprocess all data
    df = load_and_preprocess(filepaths)
    
    # 2. Select important features
    selected_data = select_features(df)
    
    # 3. Augment data to balance classes
    augmented_data = augment_data(selected_data, n_augment=3)
    
    # 4. Prepare for modeling
    # Create feature matrix (wavelength + absorbance)
    X = augmented_data[['wavelength', 'absorbance_standardized']].values
    y = augmented_data['drug_type'].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Check class distribution
    print("\nClass distribution:", Counter(y))
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y_encoded)
    print("Class distribution after SMOTE:", Counter(y_res))
    
    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
    
    # 6. Build and train model
    model = build_model(X_train, y_train)
    
    # 7. Evaluation
    print("\nBest Parameters:", model.best_params_)
    print("Best CV Score:", model.best_score_)
    print("Training Accuracy:", model.score(X_train, y_train))
    print("Test Accuracy:", model.score(X_test, y_test))
    
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    
    # Feature importance (for RandomForest)
    if isinstance(model.best_estimator_.named_steps['classifier'], RandomForestClassifier):
        importances = model.best_estimator_.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title("Feature Importances")
        plt.bar(range(X_train.shape[1]), importances[indices], align="center")
        plt.xticks(range(X_train.shape[1]), indices)
        plt.xlim([-1, X_train.shape[1]])
        plt.show()

    # Save the model for later use
    import joblib
    joblib.dump(model.best_estimator_, 'drug_classifier.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("\nModel saved as 'drug_classifier.pkl'")