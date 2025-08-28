import os
import pandas as pd
import joblib
import sys

# Path Setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.data_processing import process_file_to_features
from src.model import create_model
from imblearn.over_sampling import SMOTE


def train_model():
    """Main function to train the final model."""

    # Build full paths from the project root
    raw_data_path = os.path.join(PROJECT_ROOT, 'data', 'raw')
    models_dir = os.path.join(PROJECT_ROOT, 'models')

    # Data aggregation
    all_files = [f for f in os.listdir(raw_data_path) if f.endswith('.txt')]
    list_of_dfs = []

    print("Processing all data files...")
    for file in all_files:
        file_path = os.path.join(raw_data_path, file)
        processed_df = process_file_to_features(file_path)
        list_of_dfs.append(processed_df)
    master_df = pd.concat(list_of_dfs, ignore_index=True)
    print("Data aggregation complete.")

    X = master_df.drop('Label', axis=1)
    y = master_df['Label']

    print("\nApplying SMOTE to the full dataset...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("SMOTE complete.")

    model = create_model()

    print("\nTraining the final RandomForest model...")
    model.fit(X_resampled, y_resampled)
    print("Model training complete.")

    os.makedirs(models_dir, exist_ok=True)
    model_filename = os.path.join(models_dir, 'fog_detection_random_forest.joblib')
    joblib.dump(model, model_filename)
    print(f"\nFinal model saved to: {model_filename}")


if __name__ == '__main__':
    train_model()