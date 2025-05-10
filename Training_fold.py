import os
import yaml
from ultralytics import YOLO

# Your class names
charclassnames = [ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                   "b", "d", "h", "v", "t", "ta", "y", "n", "s", "sad",
                   "l", "j", "m", "g", "e", "wh" ]
nc = len(charclassnames)
# Base directories
folds_base = "datasets/kfold_dataset/folds"
yaml_output = "datasets/kfold_dataset/yamls"
os.makedirs(yaml_output, exist_ok=True)

# Loop through each fold
for fold_name in sorted(os.listdir(folds_base)):
    fold_path = os.path.join(folds_base, fold_name)
    if not os.path.isdir(fold_path):
        continue

    # Create YAML file for this fold
    yaml_path = os.path.join(yaml_output, f"{fold_name}.yaml")
    data_yaml = {
        "train": os.path.join(*folds_base.split("/")[1:], fold_name,"images/train"),
        "val": os.path.join(*folds_base.split("/")[1:],fold_name, "images/val"),
        "nc": nc,
        "names": charclassnames
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    print(f"\n🚀 Training on {fold_name}...")

    # Load model
    model = YOLO("yolo11s.pt")

    # Train
    model.train(
        data=yaml_path,
        epochs=20,
        imgsz=640,
        batch=8,
        project="OCR_YOLO",
        name=fold_name,
        device="cuda"  # or "cpu" if no GPU
    )
1