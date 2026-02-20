# 🌱 AI-Based Crop Disease Detection (Deep Learning)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Deep Learning](https://img.shields.io/badge/DeepLearning-MobileNetV2-orange)
![Framework](https://img.shields.io/badge/Framework-TensorFlow-red)
![Domain](https://img.shields.io/badge/Domain-Computer%20Vision-green)

A **deep learning-based image classification system** that detects tomato leaf diseases using **transfer learning with MobileNetV2**.

The model classifies tomato leaf images into:

- 🍂 Tomato Early Blight  
- 🍂 Tomato Late Blight  
- 🌿 Healthy Tomato Leaf  

Final Test Accuracy: **~96.6%**

---

## 🚀 Features

- 🧠 Transfer learning using MobileNetV2 (ImageNet pretrained)
- 🔄 Data augmentation for improved generalization
- 🎯 Fine-tuning of last 20 layers
- 📊 Confusion matrix & classification report
- 📈 Training & validation performance curves
- 🖼️ Single image prediction demo
- 📦 Saved trained model (`.keras` format)
- 🧩 Clean and modular project structure

---

## 🧠 How the Model Works

### 1️⃣ Data Preparation
- Dataset: PlantVillage (Tomato subset)
- Split into:
  - Train
  - Validation
  - Test
- Images resized to **224x224**
- Pixel normalization (Rescaling 1/255)

---

### 2️⃣ Transfer Learning

- Base Model: **MobileNetV2**
- Pretrained on ImageNet
- Initial layers frozen during first phase
- Custom classification head added:
  - GlobalAveragePooling
  - Dense (ReLU)
  - Dropout
  - Softmax (3 classes)

---

### 3️⃣ Fine-Tuning Strategy

- Last 20 layers unfrozen
- Lower learning rate (`1e-5`)
- Additional 5 epochs for better feature adaptation

This improves generalization without overfitting.

---

## 📊 Model Performance

| Metric | Value |
|--------|--------|
| Test Accuracy | ~96.6% |
| Macro F1-score | ~0.96 |
| Weighted F1-score | ~0.97 |

---

### 🔍 Confusion Matrix Insights

- ✅ Perfect recall for healthy leaves
- 🔄 Minor confusion between Early & Late blight (visually similar diseases)
- 📉 No major class imbalance bias
- 📈 Strong precision and recall across all classes

Model shows stable convergence and minimal overfitting.

---

## 📁 Project Structure

```text
AI-Based-Crop-Disease-Detection/
│
├── model/
│   └── tomato_disease_model.keras
│
├── notebooks/
│   ├── 01_Model_Training_and_Evaluation.ipynb
│   └── 02_Predict_Single_Image.ipynb
│
├── images/
│   ├── early_blight_sample.jpg
│   ├── late_blight_sample.jpg
│   └── healthy_sample.jpg
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/AI-Based-Crop-Disease-Detection.git
cd AI-Based-Crop-Disease-Detection
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Train the Model

Open:

```
notebooks/01_Model_Training_and_Evaluation.ipynb
```

Run all cells to:

- Train model
- Apply fine-tuning
- Evaluate on test dataset
- Generate confusion matrix
- Save final model

---

### 4️⃣ Run Single Image Prediction

Open:

```
notebooks/02_Predict_Single_Image.ipynb
```

Modify the image path if needed and run to see predictions.

---

## 📊 Evaluation Metrics Included

- Training Accuracy Curve
- Validation Accuracy Curve
- Training & Validation Loss Curve
- Confusion Matrix
- Precision, Recall, F1-score
- Final Test Accuracy

This ensures model performance is evaluated beyond raw accuracy.

---

## 📦 Dataset Used

- **PlantVillage Dataset**
- Tomato leaf images
- 3 Classes:
  - Early Blight
  - Late Blight
  - Healthy

Dataset used offline for reproducibility.

*Note: Dataset not included in repository due to large size.*

---

## 🚧 Deployment Notes

- Model saved in modern `.keras` format
- Can be deployed using:
  - Streamlit
  - Flask / FastAPI
  - TensorFlow Lite (mobile devices)
- For real-world deployment, training on field images is recommended.

---

## 📌 Future Improvements

- Deploy as a Streamlit web application
- Convert model to TensorFlow Lite for mobile usage
- Expand to multiple crop types
- Add Grad-CAM for model explainability
- Train on real farm images for robustness

---

## 🛠️ Tech Stack

- Python 3.10
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 👤 Author

**Nandu**  
GitHub: https://github.com/nandithburla  

---
