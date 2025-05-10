import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
# # Load the uploaded CSV
# df = pd.read_csv("OCR_YOLO\\2nd version modified dataset\\results.csv")

# # Plot metrics (Precision, Recall, mAP@0.5, mAP@0.5:0.95)
# plt.figure(figsize=(14, 6))
# plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision (B)")
# plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall (B)")
# plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5 (B)")
# # plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95 (B)")

# plt.xlabel("Epoch")
# plt.ylabel("Metric Value")
# plt.title("Precision, Recall, and mAP Metrics Over Epochs")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Your character class names
charclassnames = [ 
    "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "9", "b", "d",
    "h", "v", "t", "ta", "y", "n",
    "s", "sad", "l", "j", "m", "g",
    "e", "wh"
]

# Map class index -> character name
index_to_char = {i: name for i, name in enumerate(charclassnames)}

# Initialize count dictionary
char_counts = {name: 0 for name in charclassnames}

# Set your label directory path
label_dir = "datasets\data_modified"

# Read all label .txt files recursively
for root, dirs, files in os.walk(label_dir):
    for filename in files:
        if filename.endswith(".txt"):
            file_path = os.path.join(root, filename)
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip() == "":
                        continue
                    class_idx = int(line.split()[0])  # class is the first number
                    char_name = index_to_char.get(class_idx)
                    if char_name is not None:
                        char_counts[char_name] += 1


# Keep your original order
characters = list(char_counts.keys())
counts = list(char_counts.values())

# Setup the figure
plt.figure(figsize=(18, 10))
bars = plt.bar(characters, counts, edgecolor='black')
colors = plt.cm.tab20(np.linspace(0.1, 0.9, len(bars)))
for bar, color in zip(bars, colors):
    bar.set_color(color)
# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + max(counts) * 0.01,  # slight offset above the bar
        f'{int(height)}',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

# Add labels and title
plt.xlabel('Characters', fontsize=14, fontweight='bold')
plt.ylabel('Number of Samples', fontsize=14, fontweight='bold')
plt.title('Dataset Distribution of Characters', fontsize=18, fontweight='bold')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Tight layout for better padding
plt.tight_layout()

# Show the plot
plt.show()