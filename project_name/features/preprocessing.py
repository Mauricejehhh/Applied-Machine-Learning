from skimage import io
from skimage.transform import resize
import skimage
import matplotlib.pyplot as plt
import json
import os


def preprocess_image(img_path):
    """Preprocess an image from the TT100K dataset.

    This function performs grayscale conversion and resizing on a given image.
    It is used by the dataset_loader.py `torch.utils.data.Dataset` class that
    loads the TT100K data.

    Args:
        img_path (str): Path to the image file.

    Returns:
        numpy.ndarray: A resized grayscale image of shape (512, 512).
    """
    image = io.imread(img_path)

    # Grayscaling & Normalizing (Normalized within rgb2gray function)
    i, (im1) = plt.subplots(1)
    i.set_figwidth(5)
    gray_image = skimage.color.rgb2gray(image)
    plt.imshow(gray_image, cmap='gray')

    # Resizing from 2048x2048 to 512x512
    resized_image = resize(
        gray_image, (512, 512), anti_aliasing=True
    )
    plt.imshow(resized_image, cmap='gray')

    return resized_image


if __name__ == "__main__":
    dataset_pth = os.getcwd() + '/project_name/data/tt100k_2021/'

    with open(dataset_pth + 'annotations_all.json') as json_data:
        annotations = json.load(json_data)

    with open(dataset_pth + 'train/ids.txt') as f:
        for id in f:
            id = id.strip()  # Removes newline and whitespace
            img_path = 'train/' + str(id) + '.jpg'
            image = io.imread(dataset_pth + img_path)

            # Grayscaling & Normalizing
            i, (im1) = plt.subplots(1)
            i.set_figwidth(5)
            gray_image = skimage.color.rgb2gray(image)
            plt.imshow(gray_image, cmap='gray')

            # Resize from 2048x2048 to 512x512
            resized_image = resize(
                gray_image, (512, 512), anti_aliasing=True
            )
            plt.imshow(resized_image, cmap='gray')

            print(f"\nCurrent Image: {id}")
            traffic_signs = annotations["imgs"][str(id)]["objects"]
            print(f"Amount of Traffic Signs: {len(traffic_signs)}")

            for i in range(len(traffic_signs)):
                bbox = traffic_signs[i]["bbox"]
                xmin = bbox["xmin"] / 4
                ymin = bbox["ymin"] / 4
                xmax = bbox["xmax"] / 4
                ymax = bbox["ymax"] / 4
                sign_type = traffic_signs[i]["category"]

                print(f"Traffic Sign Type: {sign_type}")
                print(f"xmin: {xmin}\nymin: {ymin}\nxmax: {xmax}\nymax: {ymax}")

            plt.show()
