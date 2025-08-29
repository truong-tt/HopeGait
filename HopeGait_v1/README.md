# HopeGait_v1: Daphnet FoG Dataset

This project develops a machine learning pipeline to detect Freezing of Gait (FoG) episodes in Parkinson's disease patients using data from wearable IMU sensors.

## 🎯 Project Goal
The primary objective is to build a robust classification model that can accurately identify FoG events from time-series accelerometer data, paving the way for potential real-time intervention systems like soft robotic apparel.

## 💾 Dataset
This project uses the publicly available **Daphnet Freezing of Gait Dataset**.
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait)
- **Characteristics:** Data was collected from 10 Parkinson's patients using 3 IMU sensors (ankle, thigh, trunk) at a 64Hz sampling rate.

## 🛠️ Methodology
The project follows a standard machine learning workflow:
1.  **Data Exploration & Visualization:** Initial analysis of raw sensor signals.
2.  **Feature Engineering:** A sliding window approach was used to extract time-domain (`energy`) and frequency-domain (`dominant_frequency`) features.
3.  **Handling Class Imbalance:** The SMOTE technique was applied to the training data to create a balanced dataset.
4.  **Model Training & Evaluation:** A `RandomForestClassifier` was trained and evaluated, achieving **86% recall** for the FoG class.

## 🚀 How to Use

### 1. Setup
First, clone the repository and install the required dependencies.
```bash
# Clone the repository
git clone https://github.com/truong-tt/HopeGait/tree/main/HopeGait_v1
cd HopeGait_v1

# Install dependencies
pip install -r requirements.txt
```

### 2. Usage Options
There are two ways to use this project:

#### Option A: Explore the Analysis (Jupyter Notebooks)
If you want to understand the project step-by-step, explore the Jupyter Notebooks in the `/notebooks` directory.

#### Option B: Automated Scripts (Train & Predict)
If you want to directly train the model or use it for predictions.

**To train the model:**
```bash
python src/train.py
```

**To make predictions on a new file:**
```bash
python src/predict.py --file data/raw/S09R01.txt
```

## 🔮 Future Work
-   Experiment with more advanced models like XGBoost or deep learning architectures (LSTMs, 1D-CNNs).
-   Incorporate data from more subjects to improve model generalization.
-   Develop a real-time prediction pipeline for live data streams.
