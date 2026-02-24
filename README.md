#  Walking vs Running Activity Classification

##  Overview

This project focuses on **Human Activity Recognition (HAR)** — classifying whether a person is **walking or running** using wearable sensor data (accelerometer + gyroscope). Three deep learning models are built and compared: a Feedforward Neural Network (FNN/MLP), an Artificial Neural Network (ANN), and a 1D Convolutional Neural Network (1D-CNN).

The key emphasis is not just accuracy, but selecting the **most appropriate model** based on the temporal nature of sensor data.

##  Problem Statement

Wearable fitness devices generate continuous streams of motion sensor data. Accurately identifying physical activity in real time supports:
-  Healthcare monitoring
-  Fitness tracking
-  Sports performance analytics

This project builds a binary classifier to distinguish **walking (0)** from **running (1)** using raw accelerometer and gyroscope signals.

##  Dataset

- **File:** `walkrun.csv`
- **Sensor Columns:** `acceleration_x`, `acceleration_y`, `acceleration_z`, `gyro_x`, `gyro_y`, `gyro_z`
- **Target:** `activity` — `0` (Walking) / `1` (Running)
- **Dropped Columns:** `date`, `time`, `username`, `wrist` (non-sensor metadata)
  
##  Tech Stack

| Category | Libraries |
|---|---|
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Signal Processing | `scipy` (stats, signal) |
| Machine Learning | `scikit-learn` |
| Deep Learning | `tensorflow`, `keras` |

##  Project Workflow

### 1.  Data Loading & Inspection
- Loaded CSV and inspected shape, dtypes, null values, and duplicates
- Removed non-sensor columns (date, time, username, wrist) to reduce noise and prevent data leakage

### 2.  Class Distribution Analysis
- Visualized activity balance using countplot
- Both classes were sufficiently represented — no resampling needed

### 3.  Feature Engineering
- **Acceleration Magnitude** computed from 3-axis accelerometer:
  ```
  acc_mag = √(acc_x² + acc_y² + acc_z²)
  ```
  Captures overall movement intensity across all axes

### 4.  Exploratory Data Analysis (EDA)
- **Time-series plots** — raw accelerometer signals over time, revealing frequency and amplitude differences between walking and running
- **KDE distribution plots** — feature distributions by activity class
- **Boxplots** — per-feature spread across walking vs running
- **Violin plots** — density + spread for `acc_mag`, `gyro_x`, `gyro_y`
- **Correlation heatmap** — sensor feature dependencies
- **Pair plots** — feature interaction and class separability
- **Activity-wise statistics** — mean, std, variance grouped by activity

### 5.  Outlier Detection & Removal
- **IQR method** used to detect outliers per sensor column
- Extreme outliers removed using a **3×IQR threshold** (more lenient than standard 1.5× to preserve sensor signal integrity)

### 6.  Sliding Window Feature Extraction (for FNN/MLP)
- **Window size:** 100 samples | **Step size:** 50 samples (50% overlap)
- Per window, per sensor: Mean, Std, Variance, RMS, Signal Entropy, Peak Count
- Result: Rich feature vectors capturing temporal statistics per segment

### 7.  Data Scaling
- `StandardScaler` applied separately on train and test sets
- Ensures uniform feature contribution and faster convergence

### 8.  Model Training

####  FNN / MLP (on extracted features)
```
Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dense(1, sigmoid)
```
- Optimizer: Adam | Loss: Binary Crossentropy | Epochs: 30

####  1D-CNN (on raw windowed sequences)
```
Conv1D(64) → BatchNorm → MaxPool → Dropout →
Conv1D(128) → BatchNorm → MaxPool → Dropout →
Flatten → Dense(64) → Dropout → Dense(2, softmax)
```
- Optimizer: Adam | Loss: Sparse Categorical Crossentropy | Epochs: up to 50 (EarlyStopping)

####  ANN (on flattened raw windows)
```
Dense(256) → BatchNorm → Dropout(0.4) →
Dense(128) → BatchNorm → Dropout(0.4) →
Dense(64) → Dropout(0.3) → Dense(2, softmax)
```
- Optimizer: Adam | Loss: Sparse Categorical Crossentropy | Epochs: up to 50 (EarlyStopping)


##  Model Evaluation & Comparison

All three models achieved **perfect classification** on the test set:

| Model | Accuracy | Precision | Recall | F1-Score | Best For |
|---|---|---|---|---|---|
| FNN / MLP | 1.00 | 1.00 | 1.00 | 1.00 | Quick baseline |
| ANN | 1.00 | 1.00 | 1.00 | 1.00 | Simple tasks |
| **1D-CNN**  | **1.00** | **1.00** | **1.00** | **1.00** | **Time-series data** |

### Best Model: 1D-CNN
```
Classification Report: 1D-CNN
              precision    recall  f1-score   support

     Walking       1.00      1.00      1.00       177
     Running       1.00      1.00      1.00       137

    accuracy                           1.00       314
```

>  Despite identical accuracy, **1D-CNN is the recommended model** because it explicitly learns temporal patterns from raw sensor sequences — making it more robust, generalizable, and appropriate for real-world time-series HAR tasks.

### Model Tradeoff Summary

| Model | Pros | Cons | Verdict |
|---|---|---|---|
| ANN (Flattened) | Simple, fast | Ignores temporal structure | Not ideal |
| FNN / MLP | Easy to train | Same limitation as ANN | Not scalable |
| **1D-CNN** | Learns temporal patterns | Slightly heavier | ✅ Best choice |


##  Key Insights

- **Acceleration magnitude** is the most discriminative feature — running shows significantly higher and more variable magnitudes than walking
- **Temporal modeling matters** — 1D-CNN captures motion rhythms that flat models miss
- **Outlier removal** with 3×IQR preserved signal integrity while removing sensor noise
- **Sliding window segmentation** is critical for transforming raw sensor streams into learnable sequences
- Even simple models achieve perfect accuracy on this dataset — real-world deployment would benefit from 1D-CNN's temporal robustness


##  Repository Structure

```
PRCP-1013-WalkRunClass/
│
├── Data/
│   └── walkrun.csv
│
├── PRCP-1013-WalkRunClass.ipynb    # Main notebook
└── README.md
```

---

### How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/PRCP-1013-WalkRunClass.git
   cd PRCP-1013-WalkRunClass
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow scipy jupyter
   ```

3. Launch the notebook:
   ```bash
   jupyter notebook PRCP-1013-WalkRunClass.ipynb
   ```

4. Update the dataset path in the notebook to match your local setup.

---

##  Contact

**S K Yuvaraja** — yuvarajakannappan@gmail.com
