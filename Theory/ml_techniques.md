# Machine Learning Techniques & Strategies

This document outlines the machine learning methodologies applied in the UrbanSound Classification project.

## 1. Feature Engineering
The project employs two distinct feature extraction strategies to capture the temporal and spectral characteristics of environmental sounds.

### A. Baseline Features
A combination of statistical measures from time and frequency domains:
- **MFCCs (Mel-Frequency Cepstral Coefficients)**: Captures the timbral texture of sound. The first 13 coefficients are used.
- **Spectral Centroid**: Represents the "center of gravity" of the spectrum (brightness).
- **Spectral Rolloff**: The frequency below which a percentage (usually 85%) of accumulated spectral energy lies.
- **Zero-Crossing Rate (ZCR)**: The rate at which the signal changes sign (measure of noisiness).
- **Spectral Contrast**: differences in amplitude between peaks and valleys in the spectrum (good for distinguishing musical/harmonic from noisy sounds).

### B. XGBoost-Optimized Features
An expanded set designed to give tree-based models more granular information:
- **Delta & Delta-Delta MFCCs**: Velocity and acceleration of MFCCs, capturing how timbre changes over time.
- **Mel Spectrogram Statistics**: Mean and Standard Deviation of log-scaled Mel bands.
- **Aggregated Statistics**: Statistical aggregation (mean/std) is performed over the time axis to create a fixed-length vector from variable-length audio clips.

## 2. Preprocessing & Normalization
- **Resampling**: All audio is standardized to **16 kHz** to ensure consistent spectral features.
- **Fixed Duration**: Audio is padded or truncated to exactly **4 seconds**.
- **Amplitutde Normalization**: Signals are normalized to [-1, 1] range to prevent volume differences from affecting the model.
- **Feature Scaling**: `StandardScaler` (Z-score normalization) is applied before training. This is critical for models like SVM, KNN, and MLPs, though less critical (but often helpful) for XGBoost.

## 3. Model Architecture
The project implements a **Model Factory** pattern to support various classifiers.

| Model | Technique | Key Hyperparameters / Rationale |
| :--- | :--- | :--- |
| **XGBoost (Champion)** | Gradient Boosting on Decision Trees | `objective='multi:softprob'`, `max_depth=6`, `n_estimators=400`. Optimized for tabular feature data. |
| **SVM** | Kernel Support Vector Machine | `kernel='rbf'`, `C=10`. Effective for high-dimensional feature spaces. |
| **Random Forest** | Bagging Ensemble | `n_estimators=200`, `max_depth=20`. Robust baseline, handles non-linearities well. |
| **KNN** | K-Nearest Neighbors | `weights='distance'`, `n_neighbors=5`. Instance-based learning, good for local structures. |
| **Logistic Regression** | Linear Model for Classification | `max_iter=1000`. Good baseline for linear separability. |
| **Gradient Boosting** | Boosting Ensemble | `n_estimators=100`, `learning_rate=0.1`. Similar to XGBoost but standard sklearn implementation. |
| **MLP** | Multi-Layer Perceptron | `hidden_layer_sizes=(128,64)`. Feed-forward Neural Network for non-linear mappings. |
| **Gaussian NB** | Naive Bayes | Probabilistic classifier with Gaussian assumption. Fast baseline. |
| **LDA** | Linear Discriminant Analysis | `solver='svd'`. Dimensionality reduction + classification. |
| **QDA** | Quadratic Discriminant Analysis | Assumes different covariance matrices per class. |
| **Decision Tree** | CART | `criterion='gini'`. Simple, interpretable baseline. |

## 4. Training Pipeline
The training process (`src/train.py`) follows a rigorous pipeline to ensure model validity and performance.

### A. Data Preparation & Caching
- **Lazy Loading**: Audio files are processed on-the-fly or loaded from a cache (`data/processed`) to save computation time on subsequent runs.
- **Feature Computation**: If not cached, the `FeatureExtractor` computes the specified feature set (Baseline or XGB) for the audio clip.

### B. Feature Scaling
- **Standardization**: A `StandardScaler` is fitted on the training data to remove the mean and scale to unit variance.
- **Importance**: This is crucial for distance-based models (KNN, SVM) and gradient-based models (MLP) to ensure all features contribute equally.

### C. Validation Strategy (10-Fold Cross-Validation)
Before the final fit, the model undergoes **10-Fold Cross-Validation** on the training set:
- The training data is split into 10 subsets.
- The model is trained on 9 and validated on 1, repeated 10 times.
- **Output**: A mean accuracy score with standard deviation (e.g., `Accuracy: 0.85 (+/- 0.02)`).
- **Goal**: To estimate how well the model generalizes to unseen data and detect overfitting early.

### D. Final Model Persistence
- **Retraining**: After cross-validation, the model is refitted on the **entire** training dataset to maximize learning.
- **Artifact Saving**: Both the trained model and the fitted scaler are saved as `.pkl` files (e.g., `xgboost_xgb_model.pkl`), ensuring the exact same scaling is applied during inference.

## 5. Evaluation Metrics
- **Imbalanced Handling**: The code calculates **weighted** Precision, Recall, and F1-scores to account for potential class imbalances.
- **Confusion Matrix**: Used to visualize misclassifications (e.g., distinguishing between 'children_playing' and 'street_music').
