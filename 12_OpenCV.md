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

---

# 15. Images as Arrays

OpenCV represents an image as a NumPy array.

```text
Grayscale image:
    height x width

Color image:
    height x width x channels
```

Example:

```python
import cv2

image = cv2.imread("image.jpg")

print(image.shape)
```

Output idea:

```text
(720, 1280, 3)

720  = height
1280 = width
3    = color channels
```

---

# 16. Color Spaces

OpenCV reads color images as BGR.

Many other libraries use RGB.

```python
bgr = cv2.imread("image.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
```

Common color spaces:

| Color space | Use |
|---|---|
| **BGR** | OpenCV default |
| **RGB** | Matplotlib/PIL display |
| **GRAY** | Simpler image processing |
| **HSV** | Color filtering |

---

# 17. Thresholding

Thresholding converts pixels based on a cutoff.

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresholded = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)
```

Simple meaning:

```text
Pixel > threshold -> white
Pixel <= threshold -> black
```

Useful for:

- Document processing
- Shape extraction
- Simple segmentation
- Preprocessing scanned images

---

# 18. Contours

Contours are boundaries of shapes.

```python
contours, hierarchy = cv2.findContours(
    thresholded,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
```

Flow:

```text
Image -> grayscale -> threshold/edges -> contours -> shape analysis
```

Contour use cases:

- Count objects
- Detect shapes
- Find document boundaries
- Measure object area

---

# 19. Morphological Operations

Morphology changes binary image shapes.

| Operation | Meaning |
|---|---|
| **Erosion** | Shrinks white regions |
| **Dilation** | Expands white regions |
| **Opening** | Removes small noise |
| **Closing** | Fills small holes |

Example:

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
```

---

# 20. OpenCV with Deep Learning

OpenCV is often used before and after deep-learning models.

```text
OpenCV reads image
      |
OpenCV resizes/normalizes
      |
Deep-learning model predicts
      |
OpenCV draws boxes/labels
```

Example:

```python
image = cv2.imread("image.jpg")
resized = cv2.resize(image, (640, 640))

# model prediction happens here

cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), 2)
```

---

# 21. Common Mistakes

| Mistake | Problem | Fix |
|---|---|---|
| Forgetting BGR/RGB difference | Wrong colors | Convert color space |
| Hardcoding image paths | File read fails | Check path and `image is None` |
| Wrong width/height order | Distorted resize/crops | Remember OpenCV size is `(width, height)` |
| Too much blur | Removes useful detail | Tune kernel size |
| Poor lighting | Bad detection | Normalize or improve image capture |
