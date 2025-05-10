import pandas as pd
import matplotlib.pyplot as plt
import os

# Base directory containing all folds
results_base = "OCR_YOLO"
folds = sorted([f for f in os.listdir(results_base) if f.startswith("fold")])

# Metric summary for each fold
fold_metrics = []

for fold in folds:
    csv_path = os.path.join(results_base, fold, "results.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Missing: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    last_epoch = df.iloc[-1]  # take final epoch metrics

    fold_metrics.append({
        "fold": fold,
        "precision": last_epoch["metrics/precision(B)"],
        "recall": last_epoch["metrics/recall(B)"],
        "mAP50": last_epoch["metrics/mAP50(B)"],
        "mAP50-95": last_epoch["metrics/mAP50-95(B)"],
    })

# Convert to DataFrame
metrics_df = pd.DataFrame(fold_metrics)
print(metrics_df)

# Plot
metrics_df.set_index("fold").plot(kind="bar", figsize=(10, 6), rot=0)
plt.title("Cross-Validation Performance")
plt.ylabel("Metric Score")
plt.grid(True)
plt.tight_layout()
plt.show()
