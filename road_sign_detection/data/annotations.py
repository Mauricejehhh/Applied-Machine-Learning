import os
import json
from typing import Dict, Any
from sklearn.model_selection import train_test_split


def check_annotations(
    root: str,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    full_path: str = 'annotations_all.json',
    train_path: str = 'train_val_annotations.json',
    test_path: str = 'test_annotations.json'
) -> None:
    """
    Validates the existence of annotation files and splits the dataset into train/val and test sets.
    If the train/val and test files already exist with the expected split sizes, they are reused.
    Otherwise, it creates new split annotation files.

    Args:
        root (str): Root directory where annotation files are stored.
        train_size (float): Proportion of data to use for training. Defaults to 0.70.
        val_size (float): Proportion of data to use for validation. Defaults to 0.15.
        test_size (float): Proportion of data to use for testing. Defaults to 0.15.
        full_path (str): Filename of the complete annotation JSON. Defaults to 'annotations_all.json'.
        train_path (str): Output filename for train/val split. Defaults to 'train_val_annotations.json'.
        test_path (str): Output filename for test split. Defaults to 'test_annotations.json'.

    Raises:
        AssertionError: If the sum of split sizes is not equal to 1.0.
        FileNotFoundError: If the full annotations file does not exist.
        AssertionError: If the split result does not match the expected total image count.
    """
    assert (train_size + val_size + test_size) == 1.0, \
        'Splits must sum to 1.0!'

    anno_path = os.path.join(root, full_path)
    train_anno_path = os.path.join(root, train_path)
    test_anno_path = os.path.join(root, test_path)

    if not os.path.exists(anno_path):
        raise FileNotFoundError(f'Annotations file at {anno_path} does not exist.')

    with open(anno_path, 'r') as f:
        annotations: Dict[str, Any] = json.load(f)

    if os.path.exists(train_anno_path) and os.path.exists(test_anno_path):
        print('Pre-existing annotations for train/val and test splits found.')
        with open(train_anno_path, 'r') as f:
            train_annotations: Dict[str, Any] = json.load(f)

        with open(test_anno_path, 'r') as f:
            test_annotations: Dict[str, Any] = json.load(f)

        tv_split = len(train_annotations['imgs']) / len(annotations['imgs']) * 100
        t_split = len(test_annotations['imgs']) / len(annotations['imgs']) * 100

        if abs(tv_split - ((train_size + val_size) * 100)) < 1.0 and \
           abs(t_split - (test_size * 100)) < 1.0:
            print('Found annotation files with the following matching sizes:',
                  f'\n Train/Val size: {tv_split:.2f}%',
                  f'\n Test size: {t_split:.2f}%')
            return
    else:
        print('No annotations file has been found yet.')

    print('Creating new annotation files with the following sizes:',
          f'\n Train/Val size: {train_size + val_size}',
          f'\n Test size: {test_size}')

    all_img_ids = list(annotations['imgs'].keys())
    train_val_ids, test_ids = train_test_split(
        all_img_ids,
        train_size=train_size + val_size,
        test_size=test_size,
        random_state=42
    )

    train_val_annos: Dict[str, Any] = {
        'types': annotations['types'],
        'imgs': {img_id: annotations['imgs'][img_id]
                 for img_id in train_val_ids}
    }

    test_annos: Dict[str, Any] = {
        'types': annotations['types'],
        'imgs': {img_id: annotations['imgs'][img_id]
                 for img_id in test_ids}
    }

    train_len, test_len = len(train_val_annos['imgs']), len(test_annos['imgs'])
    total_split = train_len + test_len
    original = len(annotations['imgs'])

    assert total_split == original, \
        f'Incorrect split. Got sum of {total_split}, expected {original}'

    with open(train_anno_path, 'w') as f:
        json.dump(train_val_annos, f, indent=2)

    with open(test_anno_path, 'w') as f:
        json.dump(test_annos, f, indent=2)

    print('New splits for annotations have been created!')
    print(f'Amount of train/val images: {train_len}')
    print(f'Amount of test images: {test_len}')
