# Urban Sound Classification — Machine Learning Project

This project was developed as an assignment for the **Machine Learning** course of the [MSc in Artificial Intelligence](https://msc-ai.iit.demokritos.gr/), offered jointly by the University of Piraeus and NCSR "Demokritos".

**Authors:** Kostis Matzorakis, George Manthos, Spiros Batziopoulos

---
## 1. Project Overview

This project performs **automatic classification of urban sound excerpts** into 10 predefined classes using machine learning. The pipeline follows a classical approach: audio → preprocessing → handcrafted features → classifier. The goal is to build a reproducible, interpretable system for environmental sound recognition.

**Key characteristics:**
- **Dataset:** UrbanSound8K (8,732 labeled excerpts, ≤4 seconds each)
- **Task:** Multi-class classification (10 urban sound classes)
- **Approach:** Feature-based ML (no raw waveforms or end-to-end deep learning)
- **Evaluation:** 10-fold cross-validation on predefined folds (UrbanSound8K compliant)

---

## 2. Why This Project?

### 2.1 Motivation

Urban sound classification supports:
- **Noise monitoring:** Identify dominant sources (traffic, construction, etc.)
- **Smart cities:** Automated sound event detection
- **Accessibility:** Alert systems for sensitive groups (e.g., gunshots)
- **Research:** Baseline for comparing handcrafted vs deep learning approaches

### 2.2 Why Feature-Based ML?

We use **handcrafted audio features** plus classical ML instead of end-to-end deep learning because:

1. **Interpretability:** Features (MFCCs, spectral centroid, etc.) map to acoustic properties.
2. **Data efficiency:** Works well with relatively small datasets (UrbanSound8K).
3. **Low resource use:** No GPU; faster iteration.
4. **Reproducibility:** Fixed pipeline, standard libraries, comparable baselines.

---

## 3. Dataset: UrbanSound8K

### 3.1 Description

- **Source:** [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)
- **Size:** 8,732 WAV excerpts
- **Duration:** ≤4 seconds per excerpt
- **Classes:** 10 urban sound categories (air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music)
- **Origin:** Field recordings from [Freesound.org](https://freesound.org)

### 3.2 Metadata Structure

Each file has metadata including:
- **slice_file_name:** `[fsID]-[classID]-[occurrenceID]-[sliceID].wav`
- **fsID:** Freesound recording ID (multiple excerpts can share the same original recording)
- **fold:** Predefined split (1–10)
- **classID / class:** Numeric and textual label

### 3.3 Why Predefined Folds Matter

Excerpts from the same recording (same `fsID`) share:
- Same microphone, environment, background noise
- Similar spectral and temporal properties

If data were randomly shuffled, related excerpts could appear in both train and test, inflating performance (data leakage). The predefined folds keep recordings separated: no recording spans both train and test. Evaluation is therefore on truly unseen sources.

---

## 4. Methodology

### 4.1 Pipeline Overview

Raw Audio (WAV) → Preprocessing → Feature Extraction → Scaling → Classifier → Predictions


### 4.2 Preprocessing

**Purpose:** Standardize inputs so features are comparable across files.

| Step | Operation | Rationale |
|------|-----------|-----------|
| **Resampling** | 16 kHz | Most speech/audio research uses 8–16 kHz. 16 kHz preserves useful content for urban sounds while reducing size. |
| **Peak normalization** | Divide by max amplitude | Removes level differences; avoids models depending on volume. |
| **Length normalization** | Pad or truncate to 4 s | Constant duration needed for frame-based features and consistent feature vectors. |

**Implementation:** `PreProcessor.load_and_preprocess()` in `src/pre_processing.py`.

### 4.3 Feature Extraction

**Purpose:** Convert waveforms into compact numerical vectors that reflect acoustic content.

#### 4.3.1 MFCCs (Mel-Frequency Cepstral Coefficients)

- **Theory:** MFCCs approximate how the ear perceives frequency (mel scale), then apply DCT to decorrelate bands.
- **Why:** Widely used for speech and audio; capture spectral envelope and timbre.
- **Config:** 13 coefficients, FFT 2048, hop 512. Per-clip aggregation: mean and std across frames → 26 values.

#### 4.3.2 Spectral Features

- **Spectral centroid:** Center of mass of spectrum → brightness.
- **Spectral rolloff:** Frequency below which most energy lies.
- **Zero-crossing rate (ZCR):** Sign changes per second → noisiness vs tonal content.
- **Spectral contrast:** Difference between peaks and valleys in sub-bands → harmonic vs noisy structure.

Each is summarized as mean and std across frames.

#### 4.3.3 Feature Sets

| Set | Dimensions | Contents | Use case |
|-----|------------|----------|----------|
| **baseline** | 46 | MFCCs (mean/std) + spectral (centroid, rolloff, ZCR, contrast) | Fast, low-dimensional baseline |
| **xgb** | 100 | MFCCs + Δ + Δ² + spectral + log-mel mean/std | Richer representation for stronger models |

Deltas (first and second derivatives of MFCCs over time) add temporal context; log-mel captures overall spectral energy distribution.

**Implementation:** `FeatureExtractor` in `src/feature_extraction.py`.

### 4.4 Models

Five models are used to cover different paradigms:

| Model | Type | Role | Rationale |
|-------|------|------|-----------|
| **Logistic Regression** | Linear | Baseline | Simple, interpretable, robust linear classifier |
| **Random Forest** | Tree ensemble | Baseline | Handles non-linearity and interactions; no scaling needed (we still scale for consistency) |
| **XGBoost** | Gradient boosting | Main model | Strong on tabular features; common choice for audio ML |
| **SVM** | Kernel method | Non-linear alternative | RBF kernel handles complex boundaries; standard for audio |
| **MLP** | Neural network | Deep baseline | Captures non-linear patterns; different from tree-based methods |

Other models (KNN, GNB, LDA, QDA, Gradient Boosting, Decision Tree) are also included but commented out as redundant, weaker, or numerically unstable for this setup.

### 4.5 Evaluation Protocol

**10-fold cross-validation on predefined folds:**

1. For each fold k = 1…10:
   - Train on folds ≠ k
   - Test on fold k
2. Report mean ± std of accuracy (and optionally precision, recall, F1) over the 10 runs.

This matches the [UrbanSound8K evaluation guidelines](https://urbansounddataset.weebly.com/urbansound8k.html). No shuffling; only fold-based splits.

**Metrics:** Accuracy, weighted precision, weighted recall, weighted F1 (handling class imbalance).

---

## 5. Project Structure



---

## 6. Installation & Usage

### 6.1 Dependencies

pip install -r requirements.txt

Main packages: `librosa, scikit-learn, xgboost, pandas, numpy, matplotlib, seaborn, soundfile, joblib, tqdm.`

### 6.2 Setup
- Download UrbanSound8K from the official site.
- Extract it into data/raw/urbansound8k/.
- Ensure the layout is: folders fold1/ through fold10/, plus UrbanSound8K.csv in the same directory.


### 6.3 Running Experiments
1. Exploration: Run `01_data_exploration.ipynb`
2. Features: Run `02_feature_engineering.ipynb`
3. 10-fold CV (compliant): Run `03_compliant_training.ipynb`
4. All models (single split): Run `04_all_models_training.ipynb`

In `03_compliant_training.ipynb` and `04_all_models_training.ipynb`, set *`FEATURE_SET`* to "baseline" or "xgb" and *`SELECTED_MODELS`* is an array of the models to train such as ["svm"] or ["all"] for all models implemented.

### Notebook comparison: evaluation protocols

We use two training notebooks to compare evaluation strategies:

- **03_compliant_training.ipynb:** Follows the UrbanSound8K creators’ protocol — 10-fold CV on predefined folds, no reshuffling, report mean ± std over the 10 runs. Ensures comparable, publication-style results.

- **04_all_models_training.ipynb:** Uses a standard ML lifecycle with random train/test split (and optional validation). Represents the usual approach with train/validation/test. Results may be inflated due to data leakage (related recordings in both train and test) and are not comparable to the literature.

This setup lets us compare the creators’ recommended evaluation against the conventional random-split approach.

## 7. Reproducibility

- **Random seed:** `RANDOM_STATE = 42`.
- **Feature caching**: Uses `feature_set` and extraction params; clear cache when changing features.
- **Evaluation**: 10-fold CV on predefined folds only (no reshuffling).
- **Config**: Centralized in `config/config.py`.

## 8. References

- [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)
- [Librosa](https://librosa.org/)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)