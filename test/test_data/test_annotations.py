import unittest
import tempfile
import os
import json
from road_sign_detection.data.annotations import check_annotations


class TestCheckAnnotations(unittest.TestCase):
    def setUp(self) -> None:
        """Set up a temporary directory and write a mock annotations file."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.root = self.test_dir.name
        self.full_path = os.path.join(self.root, 'annotations_all.json')

        # Create mock annotation data with 100 images
        self.mock_annotations = {
            "types": ["cat", "dog"],
            "imgs": {
                str(i): {"label": "cat" if i % 2 == 0 else "dog"} for i in range(100)
            }
        }

        with open(self.full_path, 'w') as f:
            json.dump(self.mock_annotations, f)

    def tearDown(self) -> None:
        """Clean up the temporary directory after each test."""
        self.test_dir.cleanup()

    def test_creates_splits_correctly(self) -> None:
        """
        Test that the function creates train/val and test splits,
        and the splits match expected proportions.
        """
        check_annotations(self.root)

        train_val_path = os.path.join(self.root, 'train_val_annotations.json')
        test_path = os.path.join(self.root, 'test_annotations.json')

        self.assertTrue(os.path.exists(train_val_path), "Train/Val file not created")
        self.assertTrue(os.path.exists(test_path), "Test file not created")

        with open(train_val_path, 'r') as f:
            train_val = json.load(f)
        with open(test_path, 'r') as f:
            test = json.load(f)

        total = len(train_val['imgs']) + len(test['imgs'])
        self.assertEqual(total, len(self.mock_annotations['imgs']), "Total count mismatch")

        self.assertAlmostEqual(
            len(train_val['imgs']) / total, 0.85, delta=0.05,
            msg="Train/Val split is outside acceptable range"
        )
        self.assertAlmostEqual(
            len(test['imgs']) / total, 0.15, delta=0.05,
            msg="Test split is outside acceptable range"
        )

    def test_missing_file_raises(self) -> None:
        """Test that a FileNotFoundError is raised if the full annotations file is missing."""
        os.remove(self.full_path)

        with self.assertRaises(FileNotFoundError):
            check_annotations(self.root)


if __name__ == '__main__':
    unittest.main()
