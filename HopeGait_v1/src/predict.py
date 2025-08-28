import pandas as pd
import numpy as np
import joblib
import os
import sys


# --- Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.data_processing import process_file_to_features


def predict_continuous(input_file_path):
    """
    Loads a trained model and performs near-continuous prediction with smoothing
    to identify and report robust, continuous FoG episodes.
    """
    # Load model & config
    model_path = os.path.join(PROJECT_ROOT, 'models', 'fog_detection_random_forest.joblib')
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found. Please run 'python src/train.py' first.")
        return

    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        return

    # Process data with high-frequency window
    print(f"\nProcessing input file for continuous detection: {input_file_path}")
    STEP_SIZE = 8
    feature_df = process_file_to_features(input_file_path, window_size=128, step_size=STEP_SIZE)

    if feature_df.empty:
        print("Could not process the input file.")
        return

    X_new = feature_df.drop('Label', axis=1, errors='ignore')
    predictions = model.predict(X_new)

    # Smooth predictions to find FoG episodes ---
    print("\n--- Continuous Detection Results (with Smoothing) ---")

    FOG_CONFIRM_BUFFER = 4
    WALK_CONFIRM_BUFFER = 8

    in_fog_episode = False
    episodes = []
    fog_counter = 0
    walk_counter = 0
    episode_start_index = -1

    for i, pred in enumerate(predictions):
        if pred == 2:
            walk_counter = 0
            fog_counter += 1
            if fog_counter >= FOG_CONFIRM_BUFFER and not in_fog_episode:
                in_fog_episode = True
                episode_start_index = i - fog_counter
        else:
            fog_counter = 0
            if in_fog_episode:
                walk_counter += 1
                if walk_counter >= WALK_CONFIRM_BUFFER:
                    in_fog_episode = False
                    episodes.append((episode_start_index, i - walk_counter))
                    episode_start_index = -1

    if in_fog_episode:
        episodes.append((episode_start_index, len(predictions) - 1))

    # Report found episodes
    if episodes:
        print(f"✅ Found {len(episodes)} continuous FoG episode(s):")
        raw_df_time = pd.read_csv(input_file_path, sep=' ', header=None, usecols=[0], names=['Time'])

        for i, (start_idx, end_idx) in enumerate(episodes):
            start_time_ms = raw_df_time.iloc[start_idx * STEP_SIZE]['Time']
            end_time_ms = raw_df_time.iloc[end_idx * STEP_SIZE + 128]['Time']
            duration_s = end_time_s - start_time_s

            # Print results in seconds with 2 decimal places
            print(
                f"  - Episode {i + 1}: From {start_time_s:.2f} s to {end_time_s:.2f} s (Duration: {duration_s:.2f} seconds)")
    else:
        print("✅ No continuous FoG episodes detected.")


if __name__ == '__main__':
    raw_data_path = os.path.join(PROJECT_ROOT, 'data', 'raw')
    try:
        available_files = sorted([f for f in os.listdir(raw_data_path) if f.endswith('.txt')])
        if not available_files: sys.exit(f"Error: No .txt files found in '{raw_data_path}'")
    except FileNotFoundError:
        sys.exit(f"Error: Data directory not found at '{raw_data_path}'")

    print("\nPlease select a file to predict:")
    for i, filename in enumerate(available_files): print(f"  [{i + 1}] {filename}")

    while True:
        try:
            choice_str = input(f"\nEnter a number (1-{len(available_files)}): ")
            choice_int = int(choice_str)
            if 1 <= choice_int <= len(available_files):
                break
            else:
                print(f"Invalid number.")
        except ValueError:
            print("Invalid input.")

    selected_filename = available_files[choice_int - 1]
    file_to_predict = os.path.join(raw_data_path, selected_filename)

    predict_continuous(file_to_predict)