from sklearn.model_selection import KFold
import os
import shutil
import random

# Set paths
image_dir = "datasets/kfold/images"
label_dir = "datasets/kfold/labels"

output_dir = "kfold_dataset/folds"
k = 5  # number of folds

# Gather matched image-label pairs
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

image_bases = {os.path.splitext(f)[0] for f in image_files}
label_bases = {os.path.splitext(f)[0] for f in label_files}
common_bases = sorted(image_bases & label_bases)  # only those with both .jpg and .txt

print(f"✅ Found {len(common_bases)} matching image-label pairs.")

# Create file lists
images = [f"{name}.jpg" for name in common_bases]
labels = [f"{name}.txt" for name in common_bases]

# Perform K-Fold split
kf = KFold(n_splits=k, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
    print(f"\n📁 Creating Fold {fold}")
    fold_path = os.path.join(output_dir, f"fold{fold}")
    for subfolder in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(fold_path, subfolder), exist_ok=True)

    for i in train_idx:
        shutil.copy(os.path.join(image_dir, images[i]), os.path.join(fold_path, "images/train", images[i]))
        shutil.copy(os.path.join(label_dir, labels[i]), os.path.join(fold_path, "labels/train", labels[i]))

    for i in val_idx:
        shutil.copy(os.path.join(image_dir, images[i]), os.path.join(fold_path, "images/val", images[i]))
        shutil.copy(os.path.join(label_dir, labels[i]), os.path.join(fold_path, "labels/val", labels[i]))

print("\n✅ K-Fold dataset creation complete.")
