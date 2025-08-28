import pandas as pd
import numpy as np
from scipy import stats


def process_file_to_features(file_path, window_size=128, step_size=64):
    """
    Processes a raw data file into a feature DataFrame using a sliding window.
    Includes time-domain, energy, and frequency-domain features.

    Args:
        file_path (str): The path to the raw data file.
        window_size (int): The number of samples in each window.
        step_size (int): The number of samples to slide the window forward.

    Returns:
        pandas.DataFrame: A DataFrame where each row is a window and each column is a feature.
    """
    # Data loading
    column_names = [
        'Time',
        'Trunk_Accel_X', 'Trunk_Accel_Y', 'Trunk_Accel_Z',
        'Thigh_Accel_X', 'Thigh_Accel_Y', 'Thigh_Accel_Z',
        'Ankle_Accel_X', 'Ankle_Accel_Y', 'Ankle_Accel_Z',
        'Label'
    ]
    try:
        df = pd.read_csv(file_path, sep=' ', header=None, names=column_names)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return pd.DataFrame()

    df = df[df['Label'] != 0].reset_index(drop=True)
    sensor_cols = [col for col in df.columns if 'Accel' in col]

    # windowing & feature extraction
    features_list = []
    sampling_rate = 64  # Hz, from dataset documentation

    for i in range(0, len(df) - window_size, step_size):
        window = df.iloc[i: i + window_size]
        window_features = {}

        for col in sensor_cols:
            signal = window[col].values  # Use .values for numpy operations

            # Time-domain features
            window_features[f'{col}_mean'] = np.mean(signal)
            window_features[f'{col}_std'] = np.std(signal)
            window_features[f'{col}_max'] = np.max(signal)
            window_features[f'{col}_min'] = np.min(signal)
            window_features[f'{col}_var'] = np.var(signal)

            # Energy feature
            # Represents the signal's magnitude or "power"
            window_features[f'{col}_energy'] = np.sum(signal ** 2)

            # Frequency-domain feature
            # Calculate FFT and find the dominant frequency
            fft_vals = np.fft.fft(signal)
            fft_power = np.abs(fft_vals) ** 2
            freqs = np.fft.fftfreq(len(signal), d=1.0 / sampling_rate)

            # Find dominant frequency (ignoring the DC component at index 0)
            dominant_freq_idx = np.argmax(fft_power[1:]) + 1
            dominant_freq = freqs[dominant_freq_idx]
            window_features[f'{col}_dominant_freq'] = dominant_freq

        # Label assignment
        window_label = stats.mode(window['Label'], keepdims=True)[0][0]
        window_features['Label'] = window_label

        features_list.append(window_features)

    # Create final dataframe
    feature_df = pd.DataFrame(features_list)
    return feature_df


# Testing block
if __name__ == '__main__':
    sample_file = '../data/raw/S01R01.txt'
    print(f"Processing sample file with advanced features: {sample_file}")

    processed_data = process_file_to_features(sample_file)

    if not processed_data.empty:
        print("\nProcessing complete.")
        # Shape should now have more columns (features)
        print("Shape of the new feature DataFrame:", processed_data.shape)
        print("\nFirst 5 rows of the processed data (first few columns):")
        print(processed_data.iloc[:, :6].head())