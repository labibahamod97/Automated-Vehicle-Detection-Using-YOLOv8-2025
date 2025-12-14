
#YOLOv8 Vehicle Detection System

This project implements a **vehicle detection and classification system** using the **YOLOv8 deep learning model**. The system detects and classifies multiple types of vehicles from images with high accuracy and efficiency.



##  Features

* Vehicle detection using YOLOv8
* Supports multiple vehicle classes
* Custom dataset training
* Evaluation metrics and visualizations
* Inference on unseen images
* Runs on Google Colab



##  Problem Statement

Manual traffic monitoring is inefficient and error-prone. This project aims to automate vehicle detection using machine learning to support intelligent transportation systems, traffic surveillance, and road safety applications.



##  Dataset Description

A custom annotated dataset was used for training and evaluation.

### Dataset Structure

```
yolo v8 vehicle/
 ├── train/
 │    ├── images/
 │    └── labels/
 ├── val/
 │    ├── images/
 │    └── labels/
 ├── test/
 │    ├── images/
 │    └── labels/
 └── data.yaml
```

### Classes

* Ambulance
* Bus
* Car
* Motorcycle
* Truck



##  Methodology

1. Dataset preparation and annotation
2. Model selection (YOLOv8)
3. Training using transfer learning
4. Model evaluation using standard metrics
5. Vehicle detection on test images

---

##  Results

* High accuracy for common vehicle types
* Stable loss reduction during training
* Strong performance shown in confusion matrix
* Precision, Recall, F1-score, and mAP curves generated



##  Evaluation Metrics

* Precision
* Recall
* F1 Score
* mAP@50
* mAP@50-95
* Confusion Matrix
* PR Curve

---

##  Training Configuration

* Framework: Ultralytics YOLOv8
* Image Size: 640 × 640
* Epochs: 50
* Batch Size: 16
* Platform: Google Colab

---

##  How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/yolov8-vehicle-detection.git
cd yolov8-vehicle-detection
```

### 2. Install Dependencies

```bash
pip install ultralytics
```

### 3. Train the Model

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="yolo v8 vehicle/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
```

### 4. Run Detection

```python
model.predict(source="test/images", save=True)


##  Outputs

Training outputs are saved in:


runs/detect/train/


Includes:

* Confusion Matrix
* PR Curve
* F1 Curve
* Loss Curves
* Detection Results



##  Future Work

* Real-time video detection
* Larger and more diverse dataset
* Model optimization for edge devices
* Web-based deployment



## 📎 References

* Ultralytics YOLOv8 Documentation
* PyTorch Framework

