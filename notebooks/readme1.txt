UrbanSound Classification Project Report

Περιγραφή

Αυτό το project υλοποιεί ένα pipeline για classification ήχων από το UrbanSound8K dataset (περίπου 5000 samples, 10 κλάσεις).  
Ο στόχος είναι να εκπαιδεύσουμε, να αξιολογήσουμε και να αναλύσουμε μοντέλα machine learning για multiclass ήχο κατηγοριοποίηση.

Workflow

1. Προετοιμασία Dataset
10 folders για cross-validation  
Κάθε .wav αρχείο περιέχει label στο όνομα: `[slice_number]-[classID]-[...]`  
Δημιουργία λιστών `X` (features) και `y` (labels)

2. Preprocessing
- Ορισμός σταθερού sample rate για όλα τα αρχεία
- Padding ή trimming για ίδιες διάρκειες
- Feature extraction:
  - MFCCs (mean & std)
  - Chroma features
  - Spectral centroid & bandwidth
  - Zero Crossing Rate
  - RMS energy

3. Label Encoding
- Μετατροπή string labels σε αριθμητικά με `LabelEncoder`

4. Train/Test Split
- Stratified split: 80% train, 20% test
- 5-fold cross-validation για αξιολόγηση

5. Machine Learning Models
Δοκιμάστηκαν τα εξής:
- Logistic Regression
- SVM (RBF kernel)
- Random Forest
- Decision Tree
- KNN
- Perceptron
- Naive Bayes / GaussianNB
- LDA / QDA
- XGBoost

Pipeline για κάθε μοντέλο περιλαμβάνει `StandardScaler` όπου χρειάζεται.

Καλύτερο μοντέλο: XGBoost

6. Feature Selection
- Επιλογή σημαντικότερων features με `feature_importances_` από XGBoost
- Δημιουργία reduced feature set `X_train_fs / X_test_fs`

7. Hyperparameter Tuning (XGBoost)
- Χρήση `RandomizedSearchCV` με:
- `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- Scoring: `f1_macro`
- Εφαρμογή best params στο τελικό μοντέλο `best_xgb`

8. Αξιολόγηση Τελικού Μοντέλου
- Macro F1 score**  
- Per-class accuracy**  
