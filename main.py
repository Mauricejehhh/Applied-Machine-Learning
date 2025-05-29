from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.responses import RedirectResponse
import torch
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from torchvision import transforms
import os
import json
from typing import List, Dict, Any

from road_sign_detection.models.classification_base_model import CNNClassifier
from road_sign_detection.models.localization_base_model import cnn_model

# Initialize FastAPI app
app = FastAPI(
    title="TT100K Traffic Sign Detector & Classifier",
    summary="API for detecting and classifying traffic signs using CNN models trained on the TT100K dataset.",
    description="""
## Usage
- Upload an image.
- Choose whether to run classification alongside localization.
- Receive bounding box coordinates and traffic sign type predictions.

## Limitations
- Classifier works best with clearly visible signs.
- Predictions on complex scenes or blurry images may be inaccurate.

## Dataset
TT100K 2021 dataset: https://cg.cs.tsinghua.edu.cn/traffic-sign/
    """,
    version="alpha",
)

# Load trained models
localization_model: torch.nn.Module = cnn_model()
classification_model: torch.nn.Module = CNNClassifier(number_of_classes=232)

localization_model.load_state_dict(torch.load("models/localization_model.pth",
                                              weights_only=False))
classification_model.load_state_dict(torch.load("models/classification_model.pth"))

localization_model.eval()
classification_model.eval()

# Preprocessing pipelines
preprocess_localization = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

preprocess_classification = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Paths
root: str = os.getcwd() + '/road_sign_detection/data/tt100k_2021/'
annotations_pth: str = root + 'annotations_all.json'
traffic_signs_pth: str = root + '/marks/'


@app.get("/", description="Root endpoint that redirects to documentation.")
async def root() -> RedirectResponse:
    """
    Redirects the root URL to the Swagger API documentation.

    Returns:
        RedirectResponse: Redirect to '/docs'.
    """
    return RedirectResponse(url='/docs')


@app.post("/predict", response_model=Dict[str, Any])
async def predict(file: UploadFile = File(..., description="Upload a .jpg image of a traffic scene")) -> Dict[str, Any]:
    """
    Predicts bounding box coordinates and traffic sign types from an uploaded image.

    Args:
        file (UploadFile): An image file (JPEG) uploaded via POST.

    Returns:
        dict: Contains bounding box coordinates and predicted traffic sign types.
    """
    # Load annotations
    with open(annotations_pth, 'r') as f:
        annotations = json.load(f)

    # Extract image ID from filename
    image_id: str = file.filename.replace(".jpg", "")

    # Read and convert image
    try:
        image_bytes = await file.read()
        raw_image: Image.Image = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = raw_image.size
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image")

    # Check if the image has annotations
    if image_id not in annotations["imgs"]:
        raise HTTPException(status_code=404, detail="Image ID not found in annotations; it has no traffic signs.")

    # Preprocess and predict bounding box
    input_tensor = preprocess_localization(raw_image).unsqueeze(0).to(torch.float32)
    with torch.no_grad():
        predicted_box_percentages = localization_model(input_tensor).tolist()[0]
        x1 = int(predicted_box_percentages[0] * width)
        y1 = int(predicted_box_percentages[1] * height)
        x2 = int(predicted_box_percentages[2] * width)
        y2 = int(predicted_box_percentages[3] * height)
        predicted_box_pixels: List[int] = [x1, y1, x2, y2]

    traffic_signs: List[str] = []
    sorted_classes: List[str] = sorted(annotations["types"])

    # Classify traffic signs using ground truth bounding boxes
    for obj in annotations["imgs"][image_id]["objects"]:
        bbox = obj["bbox"]
        xmin, ymin, xmax, ymax = map(int, [bbox['xmin'],
                                           bbox['ymin'],
                                           bbox['xmax'],
                                           bbox['ymax']])
        cropped: Image.Image = raw_image.crop((xmin, ymin, xmax, ymax))

        processed_crop = preprocess_classification(cropped).unsqueeze(0).to(torch.float32)
        with torch.no_grad():
            class_output = classification_model(processed_crop)
            class_index = torch.argmax(class_output, dim=1).item()

        traffic_signs.append(sorted_classes[class_index])

    return {
        "bounding_box": predicted_box_pixels,
        "predicted_classes": traffic_signs
    }
