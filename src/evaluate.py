import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config.config import Config
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


class Evaluator:

    def __init__(self, feature_set="baseline"):
        self.config = Config()
        self.feature_set = feature_set

    def _tag(self, model_name):
        return f"{model_name}_{self.feature_set}"

    def evaluate_model(self, model, scaler, X_test, y_test):

        X_in = scaler.transform(X_test) if scaler is None else X_test
        Y_pred = model.predict(X_in)

        metrics = {
            "accuracy": accuracy_score(y_test, Y_pred),
            "precision": precision_score(y_test, Y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, Y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, Y_pred, average="weighted", zero_division=0),
        }

        return metrics, Y_pred

    def plot_confusion_matrix(self, Y_test, Y_pred, model_name):
        cm = confusion_matrix(Y_test, Y_pred)
        tag = self._tag(model_name)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.config.CLASS_NAMES,
            yticklabels=self.config.CLASS_NAMES,
        )

        plt.title(f"{tag} Confusion Matrix")
        plt.ylabel(f"True label")
        plt.xlabel(f"Predicted label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        save_path = os.path.join(self.config.FIGURES_DIR, f"confusion_matrix_{tag}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_classification_report(self, Y_test, Y_pred, model_name):
        tag = self._tag(model_name)

        report = classification_report(
            Y_test,
            Y_pred,
            target_names=self.config.CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )

        df = pd.DataFrame(report).transpose()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            df.iloc[:-3, :-1],
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            ax=ax
        )

        plt.title(f"{tag} Classification Report")
        plt.tight_layout()

        save_path = os.path.join(self.config.FIGURES_DIR, f"classification_report_{tag}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def compare_models(self, results_dict):

        comparison_df = pd.DataFrame(results_dict).T
        comparison_df = comparison_df.sort_values("accuracy", ascending=False)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = ["accuracy", "precision", "recall", "f1"]

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            comparison_df[metric].plot(kind="barh", ax=ax, color="skyblue")
            ax.set_title(f"{metric.capitalize()} Comparison ({self.feature_set})")
            ax.set_xlabel("Score")
            ax.set_xlim([0, 1])

            for i, v in enumerate(comparison_df[metric]):
                ax.text(v + 0.01, i, f"{v:.4f}", va="center")

        plt.tight_layout()
        save_path = os.path.join(self.config.FIGURES_DIR, f"comparison_{self.feature_set}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return comparison_df

    def save_metrics_to_csv(self, results_dict, filename=None):
        df = pd.DataFrame(results_dict).T

        if filename is None:
            filename = f"metrics_summary_{self.feature_set}.csv"
        save_path = os.path.join(self.config.MODELS_DIR, filename)
        df.to_csv(save_path)

    def print_metrics(self, model_name, metrics):
        tag = self._tag(model_name)
        print(f"\n{'=' * 50}")
        print(f"Results for {tag}")
        print(f"{'=' * 50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"{'=' * 50}\n")
