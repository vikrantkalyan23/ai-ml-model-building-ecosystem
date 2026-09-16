# Ultralytics

## 1. What is Ultralytics?

**Ultralytics** is a computer-vision library best known for YOLO models.

YOLO is commonly used for object detection, image classification, segmentation, and pose estimation.

> **Simple definition:** Ultralytics makes it easy to train and use YOLO computer-vision models.

---

## 2. Basic Information

| Item | Details |
|---|---|
| **Name** | Ultralytics |
| **Best known for** | YOLO models |
| **Type** | Computer-vision model toolkit |
| **Common tasks** | Detection, segmentation, classification, pose |
| **Python package** | ultralytics |
| **GPU support** | Yes |
| **Best for** | Practical object detection workflows |

---

# 3. What Problem Does Ultralytics Solve?

Object detection means finding objects and their locations in an image.

```text
Image
  |
  v
YOLO model
  |
  v
Boxes + labels + confidence scores
```

Example:

```text
Input image -> detect people, cars, dogs, helmets, products
```

---

# 4. What is YOLO?

YOLO stands for:

```text
You Only Look Once
```

It predicts objects in one efficient pass through the image.

```text
Image
  |
  v
Single model pass
  |
  v
Detected objects
```

YOLO is popular because it is fast and practical.

---

# 5. Basic Prediction Example

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("image.jpg")

for result in results:
    result.show()
```

Common model sizes:

```text
n -> nano, fastest/smallest
s -> small
m -> medium
l -> large
x -> extra large, strongest/slower
```

---

# 6. Training Example

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
```

Dataset config idea:

```yaml
path: dataset
train: images/train
val: images/val

names:
  0: person
  1: car
```

---

# 7. Common Tasks

| Task | Meaning |
|---|---|
| **Detection** | Find boxes around objects |
| **Segmentation** | Find object masks |
| **Classification** | Predict image class |
| **Pose estimation** | Detect body/key points |
| **Tracking** | Follow objects across video frames |

---

# 8. Important Training Parameters

| Parameter | Meaning |
|---|---|
| **data** | Dataset YAML file |
| **epochs** | Number of training passes |
| **imgsz** | Image size |
| **batch** | Batch size |
| **device** | CPU/GPU device |
| **patience** | Early stopping patience |
| **project/name** | Output folder settings |

---

# 9. Evaluation Metrics

Common object-detection metrics:

| Metric | Meaning |
|---|---|
| **Precision** | How many predicted boxes are correct |
| **Recall** | How many real objects were found |
| **mAP** | Mean Average Precision |
| **IoU** | Box overlap quality |

Simple IoU idea:

```text
IoU = overlap area / combined area
```

---

# 10. Ultralytics vs OpenCV

| Feature | Ultralytics | OpenCV |
|---|---|---|
| Main role | YOLO model training/inference | Image/video processing |
| Object detection | Excellent | Traditional or integration |
| Deep learning training | Yes | Limited |
| Image transformations | Basic | Excellent |
| Best together? | Yes | Yes |

---

# 11. Advantages

- Easy YOLO training and inference
- Good documentation and examples
- Supports multiple vision tasks
- Works with images and videos
- Strong practical object-detection workflow

---

# 12. Disadvantages

- Needs labeled image datasets for training
- GPU is often needed for good training speed
- Not for general tabular ML
- Model choice and dataset quality strongly affect results

---

# 13. When to Use Ultralytics

Use Ultralytics when:

```text
You need object detection
You need YOLO models
You need segmentation or pose estimation
You want quick computer-vision training
```

Avoid it when:

```text
You need classical tabular ML
You need NLP or LLMs
You only need basic image resizing/filtering
```

---

# 14. Quick Revision Table

| Topic | Meaning |
|---|---|
| **Ultralytics** | YOLO computer-vision toolkit |
| **YOLO** | You Only Look Once |
| **Detection** | Boxes around objects |
| **Segmentation** | Pixel masks |
| **mAP** | Object detection score |
| **IoU** | Bounding-box overlap |
| **data.yaml** | Dataset configuration file |

---

# 15. Final Summary

```text
Ultralytics is best known for YOLO models.

Use it for:
    - Object detection
    - Image segmentation
    - Image classification
    - Pose estimation
    - Video tracking

OpenCV processes images.
Ultralytics detects objects with deep learning.
```
