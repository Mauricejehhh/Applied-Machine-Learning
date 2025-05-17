import os
import json
from PIL import Image
from torch.utils.data import Dataset
from .preprocessing import preprocess_image


class TT100KDataset(Dataset):
    def __init__(self, annotations_file, root_dir, transform=None):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations['imgs'])

    def __getitem__(self, idx):
        entry = self.annotations['imgs'].values()[idx]
        img_path = os.path.join(self.root_dir, entry['path'])
        #image = Image.open(img_path).convert('RGB')
        preprocessed_image = preprocess(img_path)
        if self.transform:
            image = self.transform(preprocessed_image)
        return image
