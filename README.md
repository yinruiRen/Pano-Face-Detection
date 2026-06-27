# Pano Face Detection

While performing face anonymization on the RGB images from the [Holo360D](https://github.com/Jou719/Holo360D) dataset, we found a lack of suitable models for face detection in large-scale panoramic images. To address this, we manually annotated 800 images to build a custom dataset, and the resulting performance is currently more than sufficient for the face blurring task.

## Features

- Fine-tuning YOLOv8 on a custom small dataset for face detection
- Use standard YOLO detection dataset format.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/yinruiRen/Pano-Face-Detection.git
cd Pano-Face-Detection
pip install -e .
```

Install PyTorch according to your CUDA version if it is not already installed. See the official PyTorch installation
instructions for the correct command for your environment.

## Dataset Format

The training script expects a YOLO detection dataset:

```text
data/pano_face_detection/
+-- dataset.yaml
+-- train/
|   +-- images/
|   +-- labels/
+-- val/
    +-- images/
    +-- labels/
```

Each label file should have the same stem as its image and contain YOLO-format annotations:

```text
class_id x_center y_center width height
```

All coordinates are normalized to `[0, 1]`. For this project, the only class is:

```yaml
names:
  0: face
```

Before training, update `data/pano_face_detection/dataset.yaml` so that `path` points to your dataset root:

```yaml
path: /path/to/pano_face_detection
train: train/images
val: val/images

names:
  0: face
```

## Training

The default training entry point is `train.py`. It runs two stages:

1. Frozen warmup: freezes early YOLO layers and trains the detection head.
2. Full fine-tuning: unfreezes the model and continues training all layers.

Prepare initial YOLOv8 weights first:

```bash
checkpoints/yolov8n.pt
```

Then run:

```bash
python train.py \
  --data data/pano_face_detection/dataset.yaml \
  --weights checkpoints/yolov8s.pt \
  --device 0
```

By default, outputs are saved under:

```text
runs/pano_face/
+-- stage1_freeze/
|   +-- weights/
|       +-- best.pt
|       +-- last.pt
+-- stage2_full/
    +-- weights/
        +-- best.pt
        +-- last.pt
```

## Inference

Use `infer.py` for a simple single-image detection example（This is a two-stage training model, and we directly use the model weights obtained after the second stage. ）:

```bash
python infer.py \
  --weights path/to/train_checkpoints \
  --image path/to/image.jpg
```

## License

This project is based on Ultralytics YOLOv8. Please check `LICENSE` and the Ultralytics license requirements before
redistributing trained models or derived code.
