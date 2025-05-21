import os
import json
from torch.utils.data import Dataset
from .preprocessing import preprocess_image
from typing import Optional, Callable, Any


class TT100KDataset(Dataset):
    """Custom dataset for the TT100K traffic sign dataset."""

    def __init__(self,
                 annotations_file: str,
                 root_dir: str,
                 transform: Optional[Callable] = None):
        """
        Initializes the TT100KDataset.

        Args:
            annotations_file (str): Path to the JSON file with annotations.
            root_dir (str): Directory with all the images.
            transform (Optional[Callable], optional): Optional transform to be
            applied on a sample. Defaults to None.
        """
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_ids = list(self.annotations['imgs'].keys())
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns:
            int: Number of images in the dataset.
        """
        return len(self.annotations['imgs'])

    def __getitem__(self, idx: int) -> Any:
        """
        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Any: Transformed image sample.
        """
        img_id = self.image_ids[idx]
        entry = self.annotations['imgs'][img_id]
        img_path = os.path.join(self.root_dir, entry['path'])
        image = preprocess_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image
