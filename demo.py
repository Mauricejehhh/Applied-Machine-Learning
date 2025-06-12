
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError, ImageDraw
from torchvision import transforms
import json
import os
from io import BytesIO

from road_sign_detection.models.classification_base_model import CNNClassifier
from road_sign_detection.models.localization_base_model import BboxRegression

# Fix for Torch Error
torch.classes.__path__ = []

# ---- Paths ----
root = os.getcwd() + '/road_sign_detection/data/tt100k_2021/'
annotations_pth = root + 'annotations_all.json'
traffic_signs_pth = root + '/marks/'

# ---- Load models ----
localization_model = BboxRegression()
classification_model = CNNClassifier(number_of_classes=232)

#localization_model.load_state_dict(torch.load("models/localization_model.pth", weights_only=True))
localization_model.load_state_dict(torch.load("models/localization_model_final_ensemble.pth", weights_only=True))
classification_model.load_state_dict(torch.load("models/classification_model.pth", weights_only=True))

localization_model.eval()
classification_model.eval()

# ---- Preprocessing transforms ----
preprocess_localization = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

preprocess_classification = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ---- Load annotations ----
with open(annotations_pth, 'r') as f:
    annotations = json.load(f)

sorted_classes = sorted(annotations["types"])

# ---- Streamlit UI ----
st.title("TT100K Traffic Sign Detector & Classifier")
st.write("Upload an image from the TT100K dataset or a new image and choose whether to classify traffic signs.")

uploaded_file = st.file_uploader("Choose a .jpg traffic image", type=["jpg"])
do_classify = st.checkbox("Also classify traffic signs", value=True)


# ---- Helper Functions ----
def draw_bounding_box(image: Image.Image, box: list[int], color: str = "red", width: int = 5) -> Image.Image:
    """
    Draw a single bounding box on a PIL image.

    :param image: The original image to draw on.
    :param box: Bounding box [x1, y1, x2, y2].
    :param color: Color of the box (default: "red").
    :param width: Line width (default: 3).
    """
    if len(box) != 4:
        raise ValueError("Bounding box must be a list of four integers: [x1, y1, x2, y2].")

    x1, y1, x2, y2 = box
    if x2 < x1 or y2 < y1:
        raise ValueError(f"Invalid bounding box: x1={x1}, x2={x2}, y1={y1}, y2={y2}. Coordinates must satisfy x2 > x1 and y2 > y1.")

    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    st.image(img_copy, caption="Original Image With Predicted Bounding Box", use_container_width=True)

def demo_preprocessing(img: Image.Image, stage: str) -> None:
    """
    Shows preprocessing steps (grayscale + resize) in Streamlit.
    :param img: The raw input image
    :param stage: Check if it's for classification or Localization
    """
    if stage == "Localization":
        st.image(img, caption="Uploaded Image", use_container_width=True)
        grayscale_image = transforms.Grayscale(num_output_channels=1)(raw_image)
        st.image(grayscale_image, caption="Grayscaled Image", use_container_width=True)
        resized_image = transforms.Resize((64, 64))(grayscale_image)
        st.image(resized_image, caption="Resized Grayscaled Image (224x224)", use_container_width=True)
    
    if stage == "Classification":
        st.image(img, caption="Cropped Image", use_container_width=True)
        resized_img = transforms.Resize((32, 32))(img)
        st.image(resized_img, caption="Resized Cropped Image (64x64)", use_container_width=True)


def predict_localization(raw_img: Image.Image, dataset_image: bool) -> list[list[int]]:
    """
    Predicts bounding boxes for traffic signs in an image.
    :param raw_img: The raw input image
    :return: List of bounding boxes in pixel coordinates [x1, y1, x2, y2]
    """
    input_tensor = preprocess_classification(raw_img).unsqueeze(0).to(torch.float32)
    with torch.no_grad():
        bbox_percentages = localization_model(input_tensor).tolist()
        
        # Override for testing
        bbox_percentages = [[0.4, 0.4, 0.5, 0.5]]

        predicted_bbox = None
        
        # Convert percentage to pixels (Post Processing)
        for i in range(len(bbox_percentages)):
            x1 = int(bbox_percentages[i][0] * width)
            y1 = int(bbox_percentages[i][1] * height)
            x2 = int(bbox_percentages[i][2] * width)
            y2 = int(bbox_percentages[i][3] * height)
            if x2 <= x1 or y2 <= y1:
                st.error(f"Invalid bounding box: x1={x1}, x2={x2}, y1={y1}, y2={y2}. " 
                                 "Coordinates must satisfy x2 > x1 and y2 > y1.")
                if not dataset_image:
                    st.stop()
            else:
                predicted_bbox = [x1, y1, x2, y2]

    # Override for testing
    predicted_bbox = [1038, 1028, 1137, 1148]
    
    if predicted_bbox is not None:
        st.success(f"Predicted bounding boxes in pixels (xmin, ymin, xmax, ymax): {predicted_bbox}")
    else:
        st.error("Continue without predicted location")
    return predicted_bbox

def predict_classifying(cropped_picture: Image.Image) -> str:
    """
    Classifies a cropped traffic sign image.
    :param cropped_picture: Cropped image of a traffic sign
    :return: Predicted class label
    """
    processed_crop = preprocess_classification(cropped_picture).unsqueeze(0).to(torch.float32)
    with torch.no_grad():
        class_output = classification_model(processed_crop)
        class_index = torch.argmax(class_output, dim=1).item()

        # Convert index to class labels (Post Processing)
        return sorted_classes[class_index]

def show_classification_results(crop: Image.Image, predicted_class: str, ground_truth_class: str) -> None:
    """
    Displays predicted and ground truth class information with icons.
    :param crop: Cropped traffic sign image
    :param predicted_class: Predicted traffic sign class
    :param ground_truth_class: Ground truth traffic sign class
    """
    pred_mark_path = os.path.join(traffic_signs_pth, f"{predicted_class}.png")
    gt_mark_path = os.path.join(traffic_signs_pth, f"{ground_truth_class}.png")

    try:
        pred_mark_img = Image.open(pred_mark_path)
    except FileNotFoundError:
        pred_mark_img = None

    try:
        gt_mark_img = Image.open(gt_mark_path)
    except FileNotFoundError:
        gt_mark_img = None

    st.markdown("### Predicted vs. Ground Truth Sign Classes")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(crop, caption="Cropped GT Sign", width=100)

    with col2:
        if pred_mark_img:
            st.image(pred_mark_img, caption=f"Predicted: {predicted_class}", width=100)
        else:
            st.markdown(f"**Predicted:** {predicted_class}\n\n*No mark image*")

    with col3:
        if gt_mark_img:
            st.image(gt_mark_img, caption=f"GT: {ground_truth_class}", width=100)
        else:
            st.markdown(f"**GT:** {ground_truth_class}\n\n*No mark image*")

    with col4:
        st.markdown("### ✅" if predicted_class == ground_truth_class else "### ❌")

    st.markdown("---")


# ---- Fallback if image is not in dataset ----
def new_image_demo(image: Image.Image) -> None:
    """
    Handles images that are not present in the dataset annotations.
    Shows predicted bounding boxes and optionally classified signs.
    :param image: Input image not in dataset
    """
    st.markdown("### Image not in dataset.")
    demo_preprocessing(image, stage="Localization")

    bbox = predict_localization(image, dataset_image=False)

    # Override for testing
    bbox = [1038, 1028, 1137, 1148]

    draw_bounding_box(image, bbox)

    cropped_img = image.crop(bbox)

    demo_preprocessing(cropped_img, stage="Classification")

    st.markdown("### Predicted Bounding Boxes")
    col1, col2 = st.columns(2)
    
    predicted_sign = None

    with col1:
        st.image(cropped_img, caption=f"Cropped Image {bbox}", width=350)

    if do_classify:
        predicted_sign = predict_classifying(cropped_img)

    if predicted_sign:
        mark_path = os.path.join(traffic_signs_pth, f"{predicted_sign}.png")
        try:
            mark_img = Image.open(mark_path)
        except FileNotFoundError:
            mark_img = None

        with col2:
            if mark_img:
                st.image(mark_img, caption=f"Predicted Sign: {predicted_sign}", width=350)
            else:
                st.markdown(f"**Predicted Sign:** {predicted_sign}\n\n*No mark image*")


# ---- Main logic ----
if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.read()
        raw_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = raw_image.size

        image_id = uploaded_file.name.replace(".jpg", "")
        if image_id not in annotations["imgs"]:
            new_image_demo(raw_image)
        else:
            demo_preprocessing(raw_image, stage="Localization")
            predicted_box = predict_localization(raw_image, dataset_image=True)
            #predicted_box = [[1000, 1000, 1200, 1200], [900, 900, 1000, 1000]]

            st.markdown("### Predicted vs. Ground Truth Bounding Boxes")
            col1, col2 = st.columns(2)

            with col1:
                st.header("Predicted Crops")
                if predicted_box is not None:
                    crop = raw_image.crop(predicted_box)
                    st.image(crop, caption=f"Predicted Box {predicted_box}", use_container_width=True)
                    st.markdown("---")
                else:
                    st.write("No correct prediction")

            with col2:
                st.header("Ground Truth Crops")

            for obj in annotations["imgs"][image_id]["objects"]:
                gt_class = obj["category"]
                bbox = obj["bbox"]
                gt_box = tuple(map(int, [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]))

                with col2:
                    crop = raw_image.crop(gt_box)
                    st.image(crop, caption=f"GT Box {list(gt_box)}", use_container_width=True)

                if do_classify:
                    predicted_class = predict_classifying(crop)
                    show_classification_results(crop, predicted_class, gt_class)
                
    except UnidentifiedImageError:
        st.error("Invalid image. Please upload a valid .jpg file.")
