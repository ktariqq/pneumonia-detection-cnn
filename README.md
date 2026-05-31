# Pneumonia Detection from Chest X-Rays Using Deep Learning

Custom CNN and VGG16 transfer learning models for automated pneumonia detection from chest X-ray images with Grad-CAM explainability, class imbalance handling, and medical imaging evaluation metrics.

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-8A2BE2.svg)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-9370DB.svg)
![License](https://img.shields.io/badge/License-MIT-4B0082.svg)

<div align="center">

━━━━━━━━━━━━━━ ✦ ✧ ✦ ━━━━━━━━━━━━━━

</div>

## 🟣 Overview

This project focuses on the development and evaluation of deep learning models for binary classification of chest X-ray images into **NORMAL** and **PNEUMONIA** categories.

The system compares a custom Convolutional Neural Network (CNN) against a transfer learning approach based on VGG16 pretrained ImageNet weights. Additional techniques such as data augmentation, class weighting, early stopping, learning rate scheduling, and Grad-CAM visualization are incorporated to improve performance and interpretability.

The project demonstrates a complete medical imaging pipeline from data preprocessing and training to evaluation and model explainability.
<br><br>

## 🟣 Key Features

- Binary pneumonia detection from chest X-rays
- Custom CNN architecture implementation
- VGG16 transfer learning model
- Class imbalance handling using weighted loss functions
- Data augmentation for improved generalization
- Batch normalization and dropout regularization
- Early stopping and adaptive learning rate scheduling
- Grad-CAM explainability for model interpretation
- Comprehensive medical evaluation metrics
<br><br>

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---------|---------|---------|---------|---------|---------|
| VGG16 Transfer Learning | 96.5% | 95.8% | 98.2% | 97.0% | 0.987 |
| Custom CNN | 94.2% | 93.1% | 96.5% | 94.8% | 0.972 |

<br><br>

## 🟣 Clinical Evaluation Summary

- Sensitivity (Recall): 98.2%
- Specificity: 92.1%
- Positive Predictive Value (PPV): 95.8%
- Negative Predictive Value (NPV): 97.5%
- AUC-ROC: 0.987

The transfer learning model achieved strong performance in identifying pneumonia cases while maintaining high reliability for normal cases.
<br><br>

## 🟣 Dataset Information

| Property | Value |
|-----------|---------|
| Dataset | Chest X-Ray Pneumonia |
| Total Images | 5,856 |
| Normal Images | 1,583 |
| Pneumonia Images | 4,273 |
| Classes | 2 |
| Split | Train / Validation / Test |
| Source | Guangzhou Women and Children's Medical Center |

<br><br>

## 🟣 Deep Learning Pipeline

<p align="center"><img align="center" src="results/training_history.png" style="width:75%; height:auto;"></p>

The workflow includes:

- Dataset preprocessing and normalization
- Data augmentation for training images
- CNN and VGG16 model training
- Validation and checkpoint saving
- Performance evaluation on unseen test data
- ROC and confusion matrix analysis
- Grad-CAM visualization for interpretability

<br><br>

## 🟣 Grad-CAM Explainability

Grad-CAM heatmaps are generated to visualize regions that influence model predictions.

This allows verification that the network focuses on clinically relevant lung regions rather than image artifacts, improving confidence in model decisions.

<p align="center"><img align="center" src="results/gradcam_visualization.png" style="width:75%; height:auto;"></p>

<br><br>

## 🟣 Evaluation Outputs

The project automatically generates:

- Training history plots
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Grad-CAM visualizations
- Sample prediction visualizations
- Model comparison reports

<br><br>

## 🟣 Project Structure

```text
pneumonia-detection-cnn/
│
├── notebooks/
│   └── pneumonia_detection.ipynb
│
├── models/
│   ├── cnn_models.py
│   └── saved_checkpoints/
│
├── utils/
│   ├── data_utils.py
│   └── visualization.py
│
├── data/
│   └── chest_xray/
│
├── results/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── gradcam_visualization.png
│   └── model_comparison.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

<br><br>

## 🟣 Technologies Used

- Python
- TensorFlow
- Keras
- VGG16 Transfer Learning
- NumPy
- Pandas
- OpenCV
- Pillow
- Matplotlib
- Seaborn
- Scikit-Learn
- tf-keras-vis
- Jupyter Notebook

<br><br>

## 🟣 Future Improvements

- Fine-tune VGG16 feature extraction layers
- Evaluate EfficientNet architectures
- Implement ensemble learning approaches
- Add uncertainty estimation methods
- Extend to multi-class pneumonia classification
- Develop REST API deployment pipeline
- Build clinical web dashboard
- Add DICOM image support

<br><br>

## 🟣 Learning Outcomes

- Medical image classification workflows
- Transfer learning with pretrained CNNs
- Handling class imbalance in healthcare datasets
- Model evaluation using clinical metrics
- Explainable AI using Grad-CAM
- Deep learning experimentation and comparison
- End-to-end machine learning pipeline development

<br><br>

## 🟣 Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/pneumonia-detection-cnn.git
cd pneumonia-detection-cnn
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Download Dataset
Download the Chest X-Ray Pneumonia dataset from Kaggle:
- URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Extract to `data/chest_xray/`
- Expected structure:
```text
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
├── NORMAL/
└── PNEUMONIA/
```
### 4. Run Notebook
```bash
jupyter notebook notebooks/pneumonia_detection.ipynb
```

Training takes ~45 minutes on GPU, ~2 hours on CPU.
