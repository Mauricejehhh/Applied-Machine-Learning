from skimage import io
from skimage.transform import resize
import skimage
import matplotlib.pyplot as plt
import json

dataset_pth = '/Users/yanni/OneDrive/Documenten/Datasets/TT100K/tt100k_2021/tt100k_2021/'

with open(dataset_pth + 'train/ids.txt') as f:
    for id in f:
        id = id[:-1] # Removes \n at the end of each id. Neither replace, strip, nor rstrip were working.
        img_path = 'train/' + str(id) + '.jpg'
        image = io.imread(dataset_pth + img_path)
        
        # Grayscaling & Normalizing (Normalized within rgb2gray function)
        i, (im1) = plt.subplots(1)
        i.set_figwidth(5)
        gray_image = skimage.color.rgb2gray(image)
        plt.imshow(gray_image, cmap = 'gray')

        # Resizing from 2048x2048 to 512x512
        resized_image = resize(gray_image, (512, 512), anti_aliasing=True) # Not sure if we're able to scale down further
        plt.imshow(resized_image, cmap = 'gray')

        # Open annotations file
        with open(dataset_pth + 'annotations_all.json') as json_data:
            annotations = json.load(json_data)

            print(f"\nCurrent Image: {id}")
            traffic_signs = len(annotations["imgs"][str(id)]["objects"])
            print(f"Amount of Traffic Signs: {traffic_signs}")

            # Loop over Traffic Signs and print their type & bounding box
            for i in range(traffic_signs):
                bbox = annotations["imgs"][str(id)]["objects"][i]["bbox"]
                xmin = bbox["xmin"] / 4 # Divide by 4 to account for resizing (2048 --> 512)
                ymin = bbox["ymin"] / 4
                xmax = bbox["xmax"] / 4
                ymax = bbox["ymax"] / 4
                sign_type = annotations["imgs"][str(id)]["objects"][i]["category"]
                print(f"Traffic Sign Type: {sign_type}")
                print(f"xmin: {xmin}\nymin: {ymin}\nxmax: {xmax}\nymax: {ymax}")
        plt.show()

        





