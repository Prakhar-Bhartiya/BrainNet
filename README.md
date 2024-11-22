
# Biometric Liveness Detection using EEG Data

In this project we built an android application which was able to solve the issue of biometric liveness detection from an EEG dataset.

Keywords —- *Mobile Computing, Machine
Learning, EEG, sensors, Android Application, Biometric*

## Features
- Designed and deployed an Android application for biometric liveness detection.
- Integrated EEG data for biometric verification.
- Used **Generative Adversarial Networks (GANs)**, **Variational Autoencoders (VAEs)**, **Autoencoders**, and **feature extraction** techniques.
- Achieved **98% accuracy** in detecting biometric liveness.
- Models Evaluated - Logistic Regression, K-Means, SVM, KNN.
- Features Evaluated - Alpha + Beta + Delta Band, Power Spectral Density, Coiflets

## Tech Stack

| Technology | Description |
|------------|-------------|
| ![Android](https://img.shields.io/badge/Android-3DDC84?style=for-the-badge&logo=android&logoColor=white) | Android SDK for building the mobile app |
| ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) | Python for machine learning model development |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) | TensorFlow for training GANs and Autoencoders |
| ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white) | Keras for building deep learning models |
| ![GANs](https://img.shields.io/badge/GANs-blue?style=for-the-badge&logo=neural-network&logoColor=white) | Generative Adversarial Networks for generating EEG-based liveness patterns |
| ![VAEs](https://img.shields.io/badge/VAEs-green?style=for-the-badge&logo=neural-network&logoColor=white) | Variational Autoencoders for feature extraction and dimensionality reduction |

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Prakhar-Bhartiya/BrainNet.git
   cd BrainNet
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the machine learning models:
   ```bash
   python train.py
   ```
