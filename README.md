# VoxGender-Net: Voice-Based Gender Classification

A professional implementation of a Deep Learning model using **TensorFlow** to classify gender (Male/Female) based on acoustic properties of voice and speech. 

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue.svg)](https://www.kaggle.com/code/alihassanshahid/trained-nnet-on-gender-classification-using-voice)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)

## 📌 Project Overview
The objective of this project is to identify a voice as male or female based on specific acoustic properties. The model is built using a Multi-Layer Perceptron (MLP) neural network architecture.

### Dataset Features
The model analyzes 20 distinct acoustic properties, including:
* **Mean Frequency** (kHz)
* **Spectral Entropy** and **Flatness**
* **Frequency Centroid**
* **Fundamental Frequency** (Mean, Min, Max)

## 🛠️ Tech Stack
* **Framework:** TensorFlow / Keras
* **Data Handling:** Pandas, NumPy
* **Preprocessing:** Scikit-Learn (StandardScaler, LabelEncoder)
* **Visualization:** Matplotlib, Seaborn

## 🚀 Model Architecture
The neural network is structured as follows:
1. **Input Layer:** 20 features
2. **Hidden Layer 1:** 128 Neurons (ReLU activation) + 30% Dropout
3. **Hidden Layer 2:** 64 Neurons (ReLU activation) + 30% Dropout
4. **Output Layer:** 1 Neuron (Sigmoid activation for Binary Classification)

## 📊 Performance
The model achieves high accuracy by utilizing standardized features and dropout layers to prevent overfitting.
* **Test Accuracy:** ~98.42%
* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam

## 📂 Repository Structure
```text
├── trained-nnet-on-gender-classification-using-voice.ipynb  # Core Notebook
├── README.md                                               # Documentation
└── dataset/                                                # (Optional) Voice.csv
