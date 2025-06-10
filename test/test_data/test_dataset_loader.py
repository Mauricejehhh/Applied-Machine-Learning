import pytest
import json
from typing import Tuple
from PIL import Image


from road_sign_detection.data.dataset_loader import (
    TT100KDataset,
    TT100KSignDataset,
    TT100KFRCNNDataset
)


@pytest.fixture
def dummy_annotations(tmp_path) -> Tuple[str, str]:
    """
    Creates a temporary image and annotations JSON file for testing datasets.

    Returns:
        Tuple[str, str]: Paths to the annotation file and image root directory.
    """
    img = Image.new('RGB', (100, 100), color='white')
    img_path = tmp_path / "test_img.jpg"
    img.save(img_path)

    annotation = {
        "types": ["stop", "yield"],
        "imgs": {
            "1": {
                "path": "test_img.jpg",
                "objects": [
                    {
                        "bbox": {"xmin": 10, "ymin": 20, "xmax": 60, "ymax": 80},
                        "category": "stop"
                    }
                ]
            }
        }
    }

    anno_path = tmp_path / "annos.json"
    with open(anno_path, "w") as f:
        json.dump(annotation, f)

    return str(anno_path), str(tmp_path)


def test_tt100k_dataset_loading(dummy_annotations: Tuple[str, str]) -> None:
    """
    Tests that TT100KDataset loads images and bounding box annotations correctly.
    """
    anno_file, root_dir = dummy_annotations
    dataset = TT100KDataset(anno_file, root_dir)

    image, target = dataset[0]
    assert isinstance(image, Image.Image), "Image is not a PIL.Image"
    assert "boxes" in target and "labels" in target, "Missing boxes or labels in target"
    assert target["boxes"].shape == (4,), "Bounding box should be a 1D tensor of length 4"
    assert target["labels"].ndim == 0, "Label tensor should be 0D (scalar)"


def test_tt100k_sign_dataset_loading(dummy_annotations: Tuple[str, str]) -> None:
    """
    Test that TT100KSignDataset loads images and classification labels correctly.
    """
    anno_file, root_dir = dummy_annotations
    dataset = TT100KSignDataset(anno_file, root_dir)

    image, label = dataset[0]
    assert isinstance(image, Image.Image), "Image is not a PIL.Image"
    assert isinstance(label, int), "Label should be an integer"


def test_tt100k_frcnn_dataset_loading(dummy_annotations: Tuple[str, str]) -> None:
    """
    Tests that TT100KFRCNNDataset loads images and targets with multiple bounding boxes and labels.
    """
    anno_file, root_dir = dummy_annotations
    dataset = TT100KFRCNNDataset(anno_file, root_dir)

    image, target = dataset[0]
    assert isinstance(image, Image.Image), "Image is not a PIL.Image"
    assert "boxes" in target and "labels" in target, "Missing boxes or labels in target"
    assert target["boxes"].ndim == 2, "Expected 2D tensor for bounding boxes"
    assert target["labels"].ndim == 1, "Expected 1D tensor for labels"
