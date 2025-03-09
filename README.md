Breast Cancer Prediction Using Histopathological Images
📌 Project Overview
Breast cancer is a leading cause of mortality among women, and early diagnosis is crucial for effective treatment. This project focuses on automated breast cancer classification using histopathological images from the BreaKHis dataset. The study integrates deep learning-based feature extraction with traditional machine learning classifiers, incorporating Principal Component Analysis (PCA) for dimensionality reduction and JAYA optimization for hyperparameter tuning.

🚀 Key Features
Deep Learning Feature Extraction: Utilizes CNN and VGG16 models to extract meaningful features from histopathology images.
Machine Learning Classifiers: Implements Random Forest (RF), K-Nearest Neighbors (KNN), and Extreme Gradient Boosting (XGB) for classification.
Dimensionality Reduction: Uses PCA to retain 95% of variance while reducing feature dimensions.
Optimization: Employs JAYA optimization to fine-tune hyperparameters and improve classifier performance.
Performance Evaluation: Compares different model combinations based on accuracy, precision, recall, and F1-score.
🏗️ System Architecture
Dataset Preprocessing: Data augmentation techniques like rotation, flipping, and rescaling.
Feature Extraction: Extracting high-dimensional feature vectors using CNN and VGG16.
Dimensionality Reduction: Applying PCA to reduce redundancy while preserving key information.
Classification: Using RF, KNN, and XGB to classify images into benign or malignant.
Optimization: Implementing JAYA algorithm for feature selection and hyperparameter tuning.
📊 Performance Metrics
After implementing JAYA optimization, the models achieved the following results:

Model	Accuracy Before JAYA (%)	Accuracy After JAYA (%)
CNN + RF + PCA	85.90	86.37
CNN + XGB + PCA	86.83	87.18
CNN + KNN + PCA	85.61	85.85
VGG16 + RF + PCA	68.62	75.06
VGG16 + XGB + PCA	81.32	83.82
VGG16 + KNN + PCA	72.10	80.68
🛠️ Technology Stack
Programming Language: Python
Frameworks & Libraries:
TensorFlow & Keras: Deep learning model implementation
Scikit-Learn & XGBoost: Machine learning classification and feature processing
OpenCV & Albumentations: Image preprocessing and augmentation
Jupyter Notebook & Google Colab: Model training and experimentation
Dataset: BreaKHis
📂 Project Files
File Name	Description
CNN_Implementation.ipynb	CNN-based feature extraction model
VGG16_Implementation.ipynb	VGG16-based feature extraction model
CNN + RF (with PCA).ipynb	CNN + Random Forest classifier with PCA
CNN + XGB (with PCA).ipynb	CNN + XGBoost classifier with PCA
CNN + KNN (with PCA).ipynb	CNN + KNN classifier with PCA
VGG16 + RF (with PCA).ipynb	VGG16 + Random Forest classifier with PCA
VGG16 + XGB (with PCA).ipynb	VGG16 + XGBoost classifier with PCA
VGG16 + KNN (with PCA).ipynb	VGG16 + KNN classifier with PCA
CNN + JAYA + PCA + XGB.ipynb	CNN + XGBoost optimized with JAYA
CNN + JAYA + PCA + RF.ipynb	CNN + Random Forest optimized with JAYA
CNN + JAYA + PCA + KNN.ipynb	CNN + KNN optimized with JAYA
VGG16 + JAYA + PCA + XGB.ipynb	VGG16 + XGBoost optimized with JAYA
VGG16 + JAYA + PCA + RF.ipynb	VGG16 + Random Forest optimized with JAYA
VGG16 + JAYA + PCA + KNN.ipynb	VGG16 + KNN optimized with JAYA
Comparison_Table.JPG	Performance comparison of models
