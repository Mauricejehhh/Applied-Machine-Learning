import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, List
from skimage import io, color
from skimage.transform import resize
from matplotlib import patches


def preprocess_and_crop_image(image, bbox):
    # Grayscaling & Normalizing (Normalized within rgb2gray function)
    gray_image = color.rgb2gray(image)
    xmin = int(bbox["xmin"] / 4)
    ymin = int(bbox["ymin"] / 4)
    xmax = int(bbox["xmax"] / 4)
    ymax = int(bbox["ymax"] / 4)

    # Resizing from 2048x2048 to 512x512
    resized_image = resize(gray_image, (512, 512), anti_aliasing=True)
    resized_image = resized_image[ymin:ymax, xmin:xmax]
    resized_image = resize(resized_image, (64, 64))
    resized_image = torch.Tensor(resized_image)
    resized_image = torch.stack([resized_image] * 3, axis=0)
    return resized_image


def preprocess_image(img_path: str) -> np.ndarray:
    """ Preprocess an image from the TT100K dataset.
    This function loads an image, converts it to grayscale, and resizes it.

    Args:
        img_path (str): Path to the image file.

    Returns:
        np.ndarray: A resized grayscale image of shape (512, 512).
    """
    # Grayscaling & Normalizing (Normalized within rgb2gray function)
    image = io.imread(img_path)
    gray_image = color.rgb2gray(image)

    # Resizing from 2048x2048 to 512x512
    resized_image = resize(gray_image, (512, 512), anti_aliasing=True)
    return resized_image


class TT100KVisualizer:
    """Visualizes and processes images from the TT100K dataset."""
    def __init__(self, dataset_path: str) -> None:
        """ Initializes the TT100KVisualizer.

        Args:
            dataset_path (str): Base path to the TT100K dataset.
        """
        self.dataset_path = dataset_path
        self.annotations = self._load_annotations()
        self.image_ids = self._load_image_ids()

    def _load_annotations(self) -> Dict[str, Any]:
        """ Loads the annotation JSON.

        Returns:
            Dict[str, Any]: Dictionary containing image annotations.
        """
        with open(
                os.path.join(self.dataset_path, 'annotations_all.json')
                ) as json_data:
            return json.load(json_data)

    def _load_image_ids(self) -> List[str]:
        """ Loads the image IDs from the text file.

        Returns:
            List[str]: List of image IDs.
        """
        ids_path = os.path.join(self.dataset_path, 'train/ids.txt')
        with open(ids_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def visualize(self) -> None:
        """ Visualizes images and their corresponding traffic
        sign annotations.
        """
        for img_id in self.image_ids:
            img_rel_path = f"train/{img_id}.jpg"
            img_abs_path = os.path.join(self.dataset_path, img_rel_path)

            gray_resized_image = preprocess_image(img_abs_path)

            # Display processed image
            fig, ax = plt.subplots()
            fig.set_figwidth(5)
            ax.imshow(gray_resized_image, cmap='gray')
            ax.set_title(f"Image ID: {img_id}")

            print(f"\nCurrent Image: {img_id}")

            traffic_signs = None
            # Print and show traffic sign information
            if img_id in self.annotations['imgs'].keys():
                traffic_signs = self.annotations["imgs"][img_id]["objects"]
                print(f"Amount of Traffic Signs: {len(traffic_signs)}")

            if traffic_signs is None:
                print('Image ID was not found in the annotations file'
                      ' or no bounding boxes exist.')
            else:
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
                    # Draw bbox onto image
                    bbox_patch = patches.Rectangle(
                        (xmin, ymin),
                        (xmax - xmin),
                        (ymax - ymin),
                        linewidth=1,
                        edgecolor='red',
                        facecolor='none'
                    )
                    ax.add_patch(bbox_patch)
            plt.show()


if __name__ == "__main__":
    dataset_path = os.path.join(os.getcwd(), 'data_storage/tt100k_2021')
    visualizer = TT100KVisualizer(dataset_path)
    visualizer.visualize()
