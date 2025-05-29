## Applied Machine Learning: Traffic Sign Detection & Classification

This project builds a traffic sign recognition system using deep learning. It handles two core tasks:

    Localization: Identify where traffic signs are located in an image by predicting bounding boxes.

    Classification: Classify each identified sign into a specific type (e.g., "stop", "speed limit", etc.).

A FastAPI-based web interface allows users to interact with the models by uploading images and receiving predictions in real time.

---

## 📁 Project Structure
```bash
├───data  # Stores .csv
├───models  # Stores .pkl
├───notebooks  # Contains experimental .ipynbs
├───project_name
│   ├───data  # For data processing, not storing .csv
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