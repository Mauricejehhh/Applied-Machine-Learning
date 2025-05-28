import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, List
from skimage import io, color
from skimage.transform import resize
from matplotlib import patches


class ImagePreprocessor:
    def __init__(self, output_size=(512, 512), to_grayscale=True):
        """Initialize the preprocessor.

        Args:
            output_size (tuple): Target size for resized images.
            to_grayscale (bool): Whether to convert images to grayscale.
        """
        self.output_size = output_size
        self.to_grayscale = to_grayscale

    def preprocess(self, img_path: str) -> np.ndarray:
        """Preprocess an image: load, optionally grayscale, and resize.

        Args:
            img_path (str): Path to the image file.

        Returns:
            np.ndarray: Preprocessed image.
        """
        image = io.imread(img_path)
        if self.to_grayscale:
            image = color.rgb2gray(image)
        return resize(image, self.output_size, anti_aliasing=True)


class TT100KVisualizer:
    """Visualizes and processes images from the TT100K dataset."""
    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path
        self.annotations = self._load_annotations()
        self.image_ids = self._load_image_ids()
        self.preprocessor = ImagePreprocessor()

    def _load_annotations(self) -> Dict[str, Any]:
        annotations_path = os.path.join(self.dataset_path,
                                        'annotations_all.json')
        with open(annotations_path, 'r') as f:
            return json.load(f)

    def _load_image_ids(self) -> List[str]:
        ids_path = os.path.join(self.dataset_path, 'train', 'ids.txt')
        with open(ids_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def _draw_bounding_boxes(self,
                             ax,
                             traffic_signs: List[Dict[str, Any]]) -> None:
        for sign in traffic_signs:
            bbox = sign["bbox"]
            xmin = bbox["xmin"] / 4
            ymin = bbox["ymin"] / 4
            xmax = bbox["xmax"] / 4
            ymax = bbox["ymax"] / 4
            sign_type = sign["category"]

            print(f"Traffic Sign Type: {sign_type}")
            print(f"xmin: {xmin}")
            print(f"ymin: {ymin}")
            print(f"xmax: {xmax}")
            print(f"ymax: {ymax}")

            bbox_patch = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=1, edgecolor='red', facecolor='none'
            )
            ax.add_patch(bbox_patch)

    def visualize(self) -> None:
        for img_id in self.image_ids:
            img_path = os.path.join(self.dataset_path,
                                    "train",
                                    f"{img_id}.jpg")
            processed_image = self.preprocessor.preprocess(img_path)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(processed_image, cmap='gray')
            ax.set_title(f"Image ID: {img_id}")

            print(f"\nCurrent Image: {img_id}")
            traffic_signs = self.annotations.get('imgs',
                                                 {}).get(img_id,
                                                         {}).get("objects", [])

            if not traffic_signs:
                print("No bounding boxes found for this image.")
            else:
                print(f"Amount of Traffic Signs: {len(traffic_signs)}")
                self._draw_bounding_boxes(ax, traffic_signs)

            plt.show()


if __name__ == "__main__":
    dataset_path = os.path.join(os.getcwd(), 'data_storage', 'tt100k_2021')
    visualizer = TT100KVisualizer(dataset_path)
    visualizer.visualize()
