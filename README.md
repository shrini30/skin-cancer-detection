CNN for Malignant Mole Identification

Overview

This project utilizes a Convolutional Neural Network (CNN) to classify skin moles as benign or malignant based on image data. The model is trained on a dataset of labeled skin lesion images to assist in early detection of skin cancer.

Features

Preprocessing of skin images (resizing, normalization, augmentation)

CNN-based classification model using TensorFlow/Keras

Training and evaluation using a labeled dataset

Performance metrics such as accuracy, precision, recall, and F1-score

Deployment options for real-world applications

Dataset

The model is trained on a publicly available skin lesion dataset, such as the ISIC (International Skin Imaging Collaboration) dataset. Ensure the dataset contains labeled images of benign and malignant moles.

Requirements

To run this project, install the following dependencies:

pip install tensorflow keras numpy pandas matplotlib scikit-learn opencv-python

Model Architecture

The CNN model consists of the following layers:

Convolutional Layers – Extracts features from the input images

Pooling Layers – Reduces dimensionality while retaining essential features

Fully Connected Layers – Classifies the image as benign or malignant

Softmax Activation – Outputs probabilities for each class

Training the Model

Run the following script to train the model:

python train.py

Training parameters such as batch size, epochs, and learning rate can be modified in config.py.

Evaluation

After training, evaluate the model using:

python evaluate.py

This script will output performance metrics such as accuracy, precision, recall, and confusion matrix.
