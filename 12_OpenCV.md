# OpenCV

## 1. What is OpenCV?

**OpenCV** stands for **Open Source Computer Vision Library**.

It is used for image processing, video processing, camera input, object detection workflows, and computer-vision applications.

> **Simple definition:** OpenCV helps computers read, edit, analyze, and understand images and videos.

---

## 2. Basic Information

| Item | Details |
|---|---|
| **Name** | OpenCV |
| **Full form** | Open Source Computer Vision Library |
| **Type** | Computer vision library |
| **Best for** | Image and video processing |
| **Python package** | cv2 |
| **Common tasks** | Read images, resize, blur, detect edges, process video |
| **Deep learning** | Some support, but not its main role |

---

# 3. What Problem Does OpenCV Solve?

Images are just arrays of pixel values.

OpenCV gives tools to process those pixels.

```text
Image/video
    |
    v
OpenCV processing
    |
    v
Enhanced image, detected edges, objects, features, or video frames
```

Examples:

- Resize images
- Convert color spaces
- Detect edges
- Blur or sharpen images
- Read webcam frames
- Track objects
- Prepare images for deep-learning models

---

# 4. Reading and Showing Images

```python
import cv2

image = cv2.imread("image.jpg")

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

OpenCV reads images in BGR color order by default.

```text
OpenCV default: BGR
Common display: RGB
```

---

# 5. Basic Image Operations

```python
import cv2

image = cv2.imread("image.jpg")

resized = cv2.resize(image, (224, 224))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
edges = cv2.Canny(gray, 100, 200)
```

Flow:

```text
Original image
      |
      v
Resize / grayscale / blur / edge detection
      |
      v
Processed image
```

---

# 6. Common OpenCV Functions

| Function | Meaning |
|---|---|
| **cv2.imread** | Read image |
| **cv2.imwrite** | Save image |
| **cv2.resize** | Resize image |
| **cv2.cvtColor** | Convert color space |
| **cv2.GaussianBlur** | Blur image |
| **cv2.Canny** | Edge detection |
| **cv2.VideoCapture** | Read video/camera |
| **cv2.rectangle** | Draw rectangle |
| **cv2.putText** | Draw text |

---

# 7. Video Capture

```python
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

---

# 8. Drawing on Images

```python
import cv2

image = cv2.imread("image.jpg")

cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), 2)
cv2.putText(
    image,
    "Object",
    (50, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)
```

---

# 9. OpenCV vs Deep Learning Frameworks

| Feature | OpenCV | PyTorch/TensorFlow |
|---|---|---|
| Image preprocessing | Excellent | Good |
| Video/camera handling | Excellent | Limited |
| Traditional computer vision | Excellent | Limited |
| Neural-network training | Limited | Excellent |
| Object detection models | Can run/integrate | Train/fine-tune |

---

# 10. Advantages

- Excellent for image/video processing
- Fast and practical
- Large set of computer-vision functions
- Good camera/video support
- Useful with deep-learning pipelines

---

# 11. Disadvantages

- Not mainly for training deep neural networks
- Some APIs are low-level
- GUI functions can behave differently across systems
- Complex modern AI tasks often need PyTorch/TensorFlow/Ultralytics

---

# 12. When to Use OpenCV

Use OpenCV when:

```text
You need image preprocessing
You need video/camera handling
You need edge detection or image transformation
You need to draw boxes or labels
You are preparing images for ML models
```

Avoid it when:

```text
You need to train large deep-learning models
You need LLMs
You need high-level object detection training
```

---

# 13. Quick Revision Table

| Topic | Meaning |
|---|---|
| **OpenCV** | Computer vision library |
| **cv2** | Python module name |
| **BGR** | Default OpenCV color order |
| **imread** | Read image |
| **resize** | Change image size |
| **Canny** | Edge detection |
| **VideoCapture** | Read video/camera frames |

---

# 14. Final Summary

```text
OpenCV is for image and video processing.

Use it for:
    - Reading images
    - Resizing images
    - Edge detection
    - Camera/video processing
    - Drawing boxes and labels
    - Preparing images for ML
```
