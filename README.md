## Applied Machine Learning: TT100K Traffic Sign Detector & Classifier

This project implements a machine learning pipeline using Convolutional Neural Networks (CNNs) to detect and classify traffic signs in images from the TT100K dataset. The system comprises two main components: a bounding box regression model for localizing signs in full images, and a classification model that recognizes individual sign types cropped from those images. Additionally, it includes a FastAPI web server for interactive predictions and multiple scripts for training, evaluating, and visualizing both models.

Key features include:

    A localization model using a ResNet-50 backbone.

    A lightweight CNN classifier capable of recognizing 232 traffic sign types.

    A FastAPI interface for real-time inference on uploaded images.

    Python scripts to train and evaluate both classification and localization pipelines.

# The Dataset and Preprocessing
The project defines custom PyTorch Dataset classes that facilitate efficient loading and preparation of the TT100K dataset, tailored for both classification and localization tasks.

**TT100KDataset**
This dataset class is used for the localization task. It loads full-resolution images and parses corresponding bounding box annotations from JSON files. Each item returned by this dataset includes the raw image and its list of bounding boxes (formatted as xmin, ymin, xmax, ymax). The class supports splitting the data into train and validation sets and includes options for applying image transformations (e.g., resizing, normalization) on-the-fly. This makes it suitable for training the bounding box regression model without any need for separate preprocessing scripts.

**TT100KSignDataset**
This class is designed specifically for the classification task. Rather than returning full images, it crops out individual traffic signs from the original images based on the bounding boxes and returns each cropped image with its associated class label. This allows efficient training of the classifier on clean, focused sign patches. Like TT100KDataset, this class handles all annotation parsing and image I/O internally, and is fully compatible with PyTorch’s DataLoader.

**Visualization Tools**
The dataset_visualizer.py module provides utilities for visualizing images and their annotations. This is critical for debugging, verifying annotation integrity, and gaining qualitative insights into the dataset. It allows users to:

    Inspect random image samples with bounding boxes.

    Confirm correct parsing of JSON annotations.

    Evaluate class balance and dataset structure.

The visualizations are generated using matplotlib, and output clear graphical representations with bounding boxes and overlaid class labels. This is valuable for both development and presentations.

# The Classification Model
The classification model is built with a CNN implemented in the CNNClassifier class. This model comprises two convolutional layers with ReLU activation and max-pooling, followed by fully connected layers that output class probabilities for 232 traffic sign types.
The TrainingPipeline class manages the end-to-end training process. It loads data via the TT100KSignDataset, applies preprocessing steps like grayscale conversion, resizing, and normalization, splits the data into training and validation sets, and trains the model over multiple epochs. Model performance is periodically evaluated against random baselines.
After training, the evaluate_classification_model.py script can be used to assess the model’s accuracy on unseen images. It loads the trained model, filters the test data, calculates classification accuracy, and visualizes predictions. This script ensures the model generalizes effectively to new samples and supports qualitative evaluation.

# The Localization Model
This model is designed to perform bounding box regression, a core task in object localization. It uses a pretrained ResNet-50 convolutional neural network as a fixed (frozen) feature extractor, stripping off the final classification layers and replacing them with a lightweight head that predicts four continuous values representing the normalized coordinates of an object’s bounding box: (xmin, ymin, xmax, ymax). The final layer uses a Sigmoid activation to ensure the output values fall between 0 and 1, making them resolution-independent. During training (train_localization_model.py), the model learns by minimizing the Smooth L1 loss between its predictions and ground truth bounding boxes using the TT100K road sign dataset. Evaluation (validation_of_localization.py) compares the model’s predicted boxes to the true annotations using Intersection over Union (IoU) as a metric, and also benchmarks performance against random box predictions. Overall, this setup enables a simple yet effective localization pipeline that can generalize to unseen road sign images by learning to regress spatial object boundaries from visual features.

# Fast API
The main.py file is the entry point to an interactive web server built using FastAPI, which provides a simple interface for testing the trained models. This server exposes an endpoint that allows users to upload images from the TT100K dataset and receive predictions for both the bounding box location of traffic signs (localization) and their class labels (classification).

When the server starts, it loads two pretrained models:

    A localization model (based on a frozen ResNet-50) that outputs the coordinates of a predicted bounding box.

    A classification model that predicts the class label of a cropped traffic sign image.

For inference, the server performs two separate tasks:

    It passes the uploaded image through the localization model to predict the most prominent bounding box in the scene.

    It then uses ground-truth bounding boxes from TT100K annotations (not the predicted one) to crop out individual signs and run them through the classification model, returning predicted class labels.

This separation ensures the classification model gets clean, labeled input, while the localization model demonstrates its ability to generalize on unseen images. The API returns both outputs as a JSON response, and includes automatic Swagger documentation via FastAPI at /docs.

To start the API, first make sure to have installed: uvicorn, fastapi[standard], python-multipart, torch, torchvision and pillow. On top of that, the dataset folder 'tt100k' needs to be put under 'data/'. To launch the API, make sure to be in the main working directory (Applied-Machine-Learning/) and in the terminal run the command 'uvicorn main:app'. In the terminal, it will output an IP address, such as 'http://127.0.0.1:8000/', which you can either 'ctrl + click' or paste in a webbrowser. Automatically, it will redirect to the Swagger UI with the automated documentation. 

Here, you find a short description about the API, as well as the documentation for two functions: 'get / root' and 'post / predict'. 'get / root' is only a function to immediately redirect the FastAPI to its Swagger UI / Documentation. This is to give quick and easy access to the API documentation and interactive interface instead of manually having to switch to the documentation each time. 

The 'post / predict' function is the interactive interface where we can upload an image. For now, as we are using two individual base models, please upload an image of the actual dataset. The function will use our trained localization model to predict its bounding boxes. After that, it will take the ground truth bounding boxes from the dataset and use the trained classification model to predict the type of traffic sign. 

The response / result will show below, with the first bounding box showing the pixels (xmin, ymin, xmax, ymax) of the original image (as our baseline model only localizes the first bounding box). Below that, it will show all predictions of traffic sign types from all bounding boxes from the ground truths of the image in the dataset. The prediction (e.g. p150) corresponds to the traffic sign in the dataset 'marks' file.
---
## Requirements
All dependencies, including specific versions, are listed in the requirements.txt file. This ensures a consistent environment for training, evaluation, and deployment.

## 📁 Project Structure
```bash
├───data  # Stores .csv
├───models  # Stores .pth
├───road_sign_detection
│   ├───data
│       ├───dataset_loader.py
│       └───dataset_visualizer.py
│   └───models
│       ├───classification_base_model.py
│       └───localization_base_model.py
├───tests
│   ├───data
│   ├───features
│   ├───models
│   └───test_main.py
├───.gitignore
├───.pre-commit-config.yaml
├───main.py
├───evaluate_classification_model.py
├───requirements.txt
├───train_classification_model.py
├───train_localization_model.py
├───validation_of_localization.py
├───README.md
```
***Made by: Maurice Theo Meijer (s5480604), David van Wuijkhuijse(s5592968), Emily Heugen (s5587042), Yannick van Maanen (s5579082)***
        ***Group 27***