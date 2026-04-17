import json
from pathlib import Path

import torch
from torchvision import datasets

from dataset import get_dataloaders
from inference_utils import CLASS_NAMES, load_model


def compute_metrics(y_true, y_pred, num_classes):
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    total = len(y_true)
    correct = sum(confusion[i][i] for i in range(num_classes))
    accuracy = correct / total if total else 0.0

    per_class = {}
    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0
    weighted_f1_sum = 0.0

    for i in range(num_classes):
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(num_classes) if r != i)
        fn = sum(confusion[i][c] for c in range(num_classes) if c != i)
        support = sum(confusion[i])

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        weighted_f1_sum += f1 * support

        per_class[CLASS_NAMES[i]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support,
        }

    macro_precision = macro_p / num_classes if num_classes else 0.0
    macro_recall = macro_r / num_classes if num_classes else 0.0
    macro_f1 = macro_f1 / num_classes if num_classes else 0.0
    weighted_f1 = weighted_f1_sum / total if total else 0.0

    return {
        "num_samples": total,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


@torch.no_grad()
def evaluate(checkpoint_path=None):
    val_dir = Path("plane_classifier/data/val")
    if not val_dir.exists():
        raise FileNotFoundError(
            "Validation folder not found at 'plane_classifier/data/val'. "
            "Run prepare_datset.py first to create train/val splits."
        )

    _, val_loader = get_dataloaders()

    # Keep class names aligned with ImageFolder index mapping.
    val_data = datasets.ImageFolder("plane_classifier/data/val")
    if val_data.classes != CLASS_NAMES:
        raise ValueError(
            f"Class mapping mismatch. Dataset classes={val_data.classes}, expected={CLASS_NAMES}."
        )

    model, loaded_path = load_model(checkpoint_path=checkpoint_path, device="cpu")
    model.eval()

    y_true = []
    y_pred = []

    for images, labels in val_loader:
        outputs = model(images)
        pred = torch.argmax(outputs, dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(pred.tolist())

    metrics = compute_metrics(y_true, y_pred, num_classes=len(CLASS_NAMES))
    metrics["loaded_checkpoint"] = str(loaded_path)
    return metrics


if __name__ == "__main__":
    report = evaluate()

    print("Validation Metrics")
    print("==================")
    print(f"Samples        : {report['num_samples']}")
    print(f"Accuracy       : {report['accuracy']:.4f}")
    print(f"Macro Precision: {report['macro_precision']:.4f}")
    print(f"Macro Recall   : {report['macro_recall']:.4f}")
    print(f"Macro F1       : {report['macro_f1']:.4f}")
    print(f"Weighted F1    : {report['weighted_f1']:.4f}")
    print(f"Checkpoint     : {report['loaded_checkpoint']}")

    print("\nPer-Class Metrics")
    print("-----------------")
    for name, values in report["per_class"].items():
        print(
            f"{name:8s}  P={values['precision']:.4f}  "
            f"R={values['recall']:.4f}  F1={values['f1_score']:.4f}  N={values['support']}"
        )

    print("\nConfusion Matrix (rows=true, cols=pred)")
    for row in report["confusion_matrix"]:
        print(row)

    with open("metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nSaved report to metrics_report.json")
