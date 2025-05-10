
<p align="center">
  <img src="figs/demo.gif" alt="YOLOv11 Persian LPR Demo" width="600">
</p>

<h1 align="center">Persian License‑Plate Recognition (OCR) with YOLOv11</h1>

<p align="center">
  <a href="https://github.com/ultralytics/ultralytics"><img src="https://img.shields.io/badge/YOLOv11-ultralytics-ff69b4.svg?logo=github"></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg">
  <img src="https://img.shields.io/badge/License-MIT-green.svg">
  <img src="https://img.shields.io/badge/Git LFS-required-critical.svg">
</p>

> Real‑time, end‑to‑end detection **and** recognition of Iranian vehicle license‑plate characters using a single YOLOv11 model.

---

## 📑 Table of Contents

1. [Key Features](#key-features)  
2. [Project Structure](#project-structure)  
3. [Quick Start](#quick-start)  
4. [Dataset](#dataset)  
5. [Training & Validation](#training--validation)  
6. [Results](#results)  
7. [Inference](#inference)  
8. [Deployment](#deployment)  
9. [Benchmarks](#benchmarks)  
10. [Troubleshooting & FAQ](#troubleshooting--faq)  
11. [Contributing](#contributing)  
12. [License](#license)  
13. [Citation](#citation)  
14. [Contact](#contact)  

---

## Key Features

- **One‑shot OCR** – detects every plate **and** individual characters in one forward pass  
- **5‑Fold cross‑validation** ready (see `fold*.yaml`)  
- **Git LFS** integrated to handle large datasets/weights (`*.pt`, `.npy`)  
- Training & inference scripts for **CPU/GPU** and **edge devices**  
- Supports **export to ONNX, TensorRT, Core ML**  
- Extensive **visualizations**: PR curves, confusion matrices, animated demos  
- MIT‑licensed – free for personal & commercial use  

---

## Project Structure

```text
.
├── datasets/             # raw images + labels (not in Git)             ─┐
│   └── README_DATA.md    # download instructions                        ─┘ managed via LFS / external link
├── OCR_YOLO/             # custom YOLOv11 model cfg & anchors
├── runs/                 # training logs, weights, TensorBoard
├── figs/                 # result images, gif demos, PR plots
├── scripts/              # helper utilities (dataset split, download)
├── *.yaml                # data configs (full + five folds)
├── Training.py           # single‑split training
├── Training_fold.py      # k‑fold driver
├── Test.py               # inference CLI
├── requirements.txt
└── README.md             # ← this file
```

Large archives (`My Project.rar`, `.pt` weights) have been **removed from Git history** – see [Removing Large Files](#troubleshooting--faq).

---

## Quick Start

### 1. Clone Repository

```bash
git lfs install
git clone https://github.com/your‑user/iranian‑lpr‑yolov11.git
cd iranian‑lpr‑yolov11
```

### 2. Setup Python Environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Download Dataset

We provide a helper script (needs a free Roboflow API key):

```bash
python scripts/download_dataset.py --dest datasets/
```

Or download manually from **Roboflow Universe** and extract to `datasets/`.

### 4. Train

```bash
# baseline (single split)
python Training.py --model yolo11n.pt --epochs 150 --imgsz 640

# 5‑fold CV
python Training_fold.py --epochs 120
```

Training logs are written to **TensorBoard** (`runs/exp*/`)—open with:

```bash
tensorboard --logdir runs --port 6006
```

---

## Dataset

| Item | Value |
|------|-------|
| Source | Persian Plate Characters (Roboflow) |
| Images | 3 684 (3300 train / 231 val / 153 test) |
| Classes | 26 (0‑9, Persian letters) |
| Annotation | YOLO txt (x<sub>c</sub>, y<sub>c</sub>, w, h) |
| License | CC BY 4.0 |

### Pre‑processing & Augmentation

- HSV/brightness shift  
- Mosaic × 4, MixUp  
- Random perspective (translate ±10%, rotate ±5°)  
- Label‑smoothing = 0.0 (per class)  

Augmentations are configured in `OCR_YOLO/hyp.scratch.yaml`.

---

## Training & Validation

All hyper‑parameters can be overridden via CLI:

```bash
python Training.py --batch 16 --epochs 200 --lr 0.01 --weights yolo11s.pt
```

`Training_fold.py` automatically creates an **aggregated CSV** (`runs/folds_metrics.csv`) for easy comparison.

### YOLOv11 Losses

| Component | Description |
|-----------|-------------|
| **box_loss** | IoU‑aware bounding‑box regression (CIoU) |
| **cls_loss** | BCE / focal loss on class logits |
| **dfl_loss** | Distribution Focal Loss – anchors width/height |

---

## Results

<p align="center">
  <img src="figs/PR_curve_fold0.png" width="450">
  <img src="figs/confusion_matrix.png" width="450">
</p>

| Fold | Precision | Recall | mAP@0.5 | mAP@0.5‑0.95 |
|------|-----------|--------|---------|--------------|
| 0 | 0.982 | 0.996 | 0.994 | 0.752 |
| 1 | 0.985 | 0.997 | 0.996 | 0.748 |
| 2 | 0.980 | 0.995 | 0.993 | 0.736 |
| 3 | 0.984 | 0.996 | 0.995 | 0.740 |
| 4 | 0.981 | 0.995 | 0.992 | 0.737 |
| **Mean** | **0.982** | **0.996** | **0.994** | **0.743** |

Full logs and plots live in `runs/`.

---

## Inference

### Command‑line

```bash
python Test.py --source imgs/demo_plate.jpg --conf 0.55 --save-txt --show
```

Outputs annotated images to `runs/predict/`.

### Python API

```python
from ultralytics import YOLO
model = YOLO("weights/best.pt")
results = model("imgs/demo_plate.jpg", conf=0.5)
for r in results:
    print(r.boxes.cls, r.boxes.xyxy)      # class ids + boxes
    print(r.names)                        # class‑id → label map
```

---

## Deployment

| Target | Tool | Command |
|--------|------|---------|
| **ONNX** | Ultralytics | `yolo export --format onnx` |
| **TensorRT** | `onnx-tensorrt` | see script in `scripts/export_tensorrt.py` |
| **Core ML** | coremltools | `yolo export --format coreml` |
| **Raspberry Pi 5** | NNPACK / TFLite | quantize to INT8, see docs |

---

## Benchmarks

| Model | Size | FPS* | mAP@0.5 | Device |
|-------|------|------|---------|--------|
| YOLOv11‑n (ours) | 7.5 M | **94** | 0.994 | RTX 3070 |
| YOLOv5‑s | 14.2 M | 82 | 0.982 | RTX 3070 |
| LPRNet | 5.4 M | 65 | 0.903 | RTX 3070 |

\* inference on 640×640 batch‑1, PyTorch 2.2

---

## Troubleshooting & FAQ

<details>
<summary><strong>Large files exceed 100 MB – push rejected</strong></summary>

GitHub blocks files > 100 MB. We use **Git LFS**. Install LFS and recommit:

```bash
git lfs install
git lfs track "*.pt" "*.npy"
git add .gitattributes
git add path/to/large_file.pt
git commit -m "track with LFS"
git push
```
</details>

<details>
<summary><strong>CUDA out of memory</strong></summary>

- Reduce `--batch` size  
- Use `--imgsz 416`  
- Enable `--cache ram` only if memory allows  
</details>

<details>
<summary><strong>Prediction string has wrong order of characters</strong></summary>

Ensure `label_map.py` matches your dataset’s character order. The default map is included.
</details>

---

## Contributing

Pull requests are welcome! Please:

1. Open an **issue** describing your feature/bug.
2. Create a branch (`git checkout -b fix/bug‑foo`).
3. Commit with conventional messages (`feat:`, `fix:`).
4. Run unit tests (`pytest`).
5. Submit a PR, filling the template.

We follow the [**Contributor Covenant Code of Conduct**](CODE_OF_CONDUCT.md).

---

## License

This project is licensed under the **MIT License**.  
See [`LICENSE`](LICENSE) for full text.

---

## Citation

If you use this repository, please cite:

```bibtex
@misc{najar2025persianlpr,
  author       = {Abolfazl Najar},
  title        = {Persian License‑Plate Recognition with YOLOv11},
  year         = {2025},
  howpublished = {GitHub},
  url          = {https://github.com/your‑user/iranian‑lpr‑yolov11}
}
```

---

## Contact

|                    |                                    |
|--------------------|------------------------------------|
| **Author**         | Abolfazl Najar |
| **Email**          | anajar2198@gmail.com |
| **LinkedIn**       | <https://www.linkedin.com/in/abolfazl-najar> |
| **Lab**            | Intelligent Power Electronics & Electric Machine Laboratory, University of Georgia |

Feel free to reach out for collaboration or questions!
