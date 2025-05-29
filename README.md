## Applied Machine Learning: TT100K Traffic Sign Detector & Classifier

This project implements a Convolutional Neural Network (CNN) based pipeline to detect and classify traffic signs in images using the TT100K dataset.
The project includes:
   - A bounding box regression model using a frozen ResNet-50.
   - A classifier CNN trained to recognize 232 traffic sign types.
   - A FastAPI server for running predictions.
   - Scripts for training and evaluating both classification and localization models.

# The Classification Model

# The Localization Model

# Fast API
---
## Requirements


## 📁 Project Structure
```bash
├───data  # Stores .csv
├───models  # Stores .pth
├───project_name
│   ├───data  # For data processing, not storing .csv
│       ├───dataset_loader.py
│       └───dataset_visualizer.py
│   ├───features
│   └───models  # For model creation, not storing .pkl
├───reports
├───tests
│   ├───data
│   ├───features
│   └───models
├───.gitignore
├───.pre-commit-config.yaml
├───main.py
├───train_model.py
├───Pipfile
├───Pipfile.lock
├───README.md
```


**Good luck and happy coding! 🚀**