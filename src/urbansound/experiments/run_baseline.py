import json
from datetime import datetime

import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from urbansound.data.metadata import load_metadata
from urbansound.features.extractors import load_audio_fixed, mfcc_stats
from urbansound.paths import OUTPUTS_DIR, MODELS_DIR


def main() -> None:
    # 1) Load metadata (paths + labels + folds)
    df = load_metadata()

    # 2) Official fold split: train folds 1-9, test fold 10 as kaggle dataset owner suggests
    train_df = df[df["fold"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9])].reset_index(drop=True)
    test_df = df[df["fold"] == 10].reset_index(drop=True)

    # 3) Extract features
    X_train, y_train = [], []
    for fp, cid in zip(train_df["filepath"], train_df["classID"]):
        y, sr = load_audio_fixed(fp)
        X_train.append(mfcc_stats(y, sr))
        y_train.append(int(cid))

    X_test, y_test = [], []
    for fp, cid in zip(test_df["filepath"], test_df["classID"]):
        y, sr = load_audio_fixed(fp)
        X_test.append(mfcc_stats(y, sr))
        y_test.append(int(cid))

    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # 4) Model pipeline (scaler + classifier)
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )

    clf.fit(X_train, y_train)

    # 5) Evaluation
    preds = clf.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    f1m = float(f1_score(y_test, preds, average="macro"))
    cm = confusion_matrix(y_test, preds)
    rep = classification_report(y_test, preds)

    # 6) Save artifacts
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_baseline_mfcc_logreg")
    out_dir = OUTPUTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "run_id": run_id,
        "model": "logreg",
        "features": "mfcc(mean,std) n=13",
        "train_folds": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "test_fold": 10,
        "accuracy": acc,
        "f1_macro": f1m,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "confusion_matrix.json").write_text(json.dumps(cm.tolist()))
    (out_dir / "classification_report.txt").write_text(rep)

    joblib.dump(clf, MODELS_DIR / f"{run_id}.joblib")

    print("Done.")
    print(metrics)


if __name__ == "__main__":
    main()
