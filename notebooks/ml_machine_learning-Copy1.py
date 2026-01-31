#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# In[2]:


SAMPLE_RATE = 16000
DURATION = 4.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)


# In[3]:


def load_audio(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y, top_db=20)

    if len(y) > N_SAMPLES:
        y = y[:N_SAMPLES]
    else:
        y = np.pad(y, (0, N_SAMPLES - len(y)))

    return y


# In[4]:


def extract_features(y):
    features = []

    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE)
    features.extend(np.mean(chroma, axis=1))

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE)
    features.append(np.mean(spec_centroid))

    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))

    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))

    return np.array(features)


# In[5]:


ROOT = r"C:\Users\kosti\Desktop\files\ml\assignement"

X = []
y = []

for fold in os.listdir(ROOT):
    fold_path = os.path.join(ROOT, fold)

    for file in os.listdir(fold_path):
        if file.endswith(".wav"):
            label = file.split('-')[1]   # UrbanSound naming
            path = os.path.join(fold_path, file)

            audio = load_audio(path)
            features = extract_features(audio)

            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset:", X.shape, y.shape)


# In[6]:


le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Classes:", le.classes_)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)


# In[8]:


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# In[9]:


from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# In[10]:


models = {
    # Linear
    "LogisticRegression": LogisticRegression(max_iter=500),
    "Perceptron": Perceptron(max_iter=1000),

    # SVM
    "SVM-RBF": SVC(kernel="rbf"),

    # Bayesian / Discriminant
    "GaussianNB": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),

    # Instance-based
    "kNN": KNeighborsClassifier(n_neighbors=7),

    # Trees
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=200),

    # Boosting
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_train)),
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss"
    )
}


# In[11]:


results = {}

for name, model in models.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", model)
    ])

    scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro"
    )

    results[name] = (scores.mean(), scores.std())
    print(f"{name:20s} | F1-macro: {scores.mean():.4f} ± {scores.std():.4f}")


# In[12]:


best_model_name = max(results, key=lambda k: results[k][0])
print("\nBest model:", best_model_name)

best_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", models[best_model_name])
])

best_pipe.fit(X_train, y_train)
y_pred = best_pipe.predict(X_test)

print(classification_report(
    y_test,
    y_pred,
    target_names=le.classes_
))


# In[13]:


xgb_base = XGBClassifier(
    objective="multi:softprob",
    num_class=len(np.unique(y_train)),
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

xgb_base.fit(X_train, y_train)


# In[14]:


importances = xgb_base.feature_importances_

print("Total features:", X_train.shape[1])


# In[15]:


K = 30   # δοκίμασε 20, 30, 40
top_idx = np.argsort(importances)[-K:]

X_train_fs = X_train[:, top_idx]
X_test_fs  = X_test[:, top_idx]


# In[16]:


from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "n_estimators": [200, 300, 400],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}


# In[17]:


xgb_tune = XGBClassifier(
    objective="multi:softprob",
    num_class=len(np.unique(y_train)),
    eval_metric="mlogloss",
    random_state=42
)

search = RandomizedSearchCV(
    estimator=xgb_tune,
    param_distributions=param_dist,
    n_iter=30,
    scoring="f1_macro",
    cv=5,
    verbose=1,
    n_jobs=-1
)

search.fit(X_train_fs, y_train)


# In[18]:


print("Best parameters:")
print(search.best_params_)

best_xgb = search.best_estimator_


# In[19]:


from sklearn.metrics import classification_report, confusion_matrix

y_pred = best_xgb.predict(X_test_fs)

print(classification_report(
    y_test,
    y_pred,
    target_names=le.classes_
))


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("XGBoost Confusion Matrix")
plt.show()


# In[21]:


importance = best_xgb.feature_importances_

top = np.argsort(importance)[-10:]
print("Top features:", top)


# In[22]:


per_class_acc = cm.diagonal() / cm.sum(axis=1)
for cls, acc in zip(le.classes_, per_class_acc):
    print(f"{cls:20s} | Accuracy: {acc:.2f}")


# In[23]:


error_dict = {}
for i, cls_true in enumerate(le.classes_):
    errors = {}
    for j, cls_pred in enumerate(le.classes_):
        if i != j and cm[i,j] > 0:
            errors[cls_pred] = cm[i,j]
    error_dict[cls_true] = errors

error_dict


# In[ ]:




