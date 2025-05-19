from skimage import io
from skimage.transform import resize
import torch
import skimage
import matplotlib.pyplot as plt
import json
import os


def preprocess_and_crop_image(image, bbox):
    # Grayscaling & Normalizing (Normalized within rgb2gray function)
    gray_image = skimage.color.rgb2gray(image)
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


def preprocess_image(image):
    """ Function derived from previous commits of the preprocessing code.
    This function is used in the dataset_loader.py torch.utils.data.Dataset
    class that loads the TT100K data.
    """
    # Grayscaling & Normalizing (Normalized within rgb2gray function)
    gray_image = skimage.color.rgb2gray(image)

    # Resizing from 2048x2048 to 512x512
    resized_image = resize(gray_image, (512, 512), anti_aliasing=True)
    resized_image = torch.Tensor(resized_image)
    resized_image = torch.stack([resized_image] * 3, axis=0)
    return resized_image


if __name__ == "__main__":
    dataset_pth = os.getcwd() + '/project_name/data/tt100k_2021/'

    with open(dataset_pth + 'annotations_all.json') as json_data:
        annotations = json.load(json_data)

    with open(dataset_pth + 'train/ids.txt') as f:
        for id in f:
            id = id[:-1]  # Removes \n at the end of each id. Neither replace, strip, nor rstrip were working.
            img_path = 'train/' + str(id) + '.jpg'
            image = io.imread(dataset_pth + img_path)

            # Grayscaling & Normalizing (Normalized within rgb2gray function)
            i, (im1) = plt.subplots(1)
            i.set_figwidth(5)
            gray_image = skimage.color.rgb2gray(image)
            plt.imshow(gray_image, cmap='gray')

            # Resizing from 2048x2048 to 512x512
            resized_image = resize(gray_image, (512, 512), anti_aliasing=True)  # Not sure if we're able to scale down further
            plt.imshow(resized_image, cmap='gray')

            # Open annotations file
            print(f"\nCurrent Image: {id}")
            traffic_signs = annotations["imgs"][str(id)]["objects"]
            print(f"Amount of Traffic Signs: {len(traffic_signs)}")

            # Loop over Traffic Signs and print their type & bounding box
            for i in range(len(traffic_signs)):
                bbox = traffic_signs[i]["bbox"]
                xmin = bbox["xmin"] / 4  # Divide by 4 to account for resizing (2048 --> 512)
                ymin = bbox["ymin"] / 4
                xmax = bbox["xmax"] / 4
                ymax = bbox["ymax"] / 4
                sign_type = traffic_signs[i]["category"]
                print(f"Traffic Sign Type: {sign_type}")
                print(f"xmin: {xmin}\nymin: {ymin}\nxmax: {xmax}\nymax: {ymax}")
            plt.show()
