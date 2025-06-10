import os
import shutil
import tempfile
import torch
import json

from train_classification_model import TrainingPipeline as ClassificationPipeline
from train_localization_model import KFoldTrainer as LocalizationPipeline


def test_classification_pipeline() -> None:
    """
    Integration test for the classification training pipeline.
    
    - Trains the classification model for 1 epoch.
    - Verifies the model file is saved.
    - Loads the saved model and performs a dummy forward pass.
    - Asserts the output shape matches expected number of classes.
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        model_path = os.path.join(tmp_dir, 'classification_model_test.pth')

        pipeline = ClassificationPipeline(
            data_root=os.path.join(os.getcwd(), 'data_storage', 'tt100k_2021'),
            model_save_path=model_path,
            epochs=1
        )
        pipeline.run()

        assert os.path.isfile(model_path), "Classification model file was not saved."

        annotation_path = os.path.join(pipeline.data_root, 'train_val_annotations.json')
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        num_classes = len(annotations['types'])

        from road_sign_detection.models.classification_base_model import CNNClassifier
        model = CNNClassifier(num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 64, 64)
            output = model(dummy_input)
        assert output.shape == (1, num_classes), "Classification model output shape mismatch."

        print("Classification pipeline integration test passed.")
    finally:
        shutil.rmtree(tmp_dir)


def test_localization_pipeline() -> None:
    """
    Integration test for the localization training pipeline.
    
    - Trains the localization model for 1 epoch.
    - Verifies the model file is saved.
    - Loads the saved model and performs a dummy forward pass.
    - Asserts the output batch size matches input batch size.
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        model_path = os.path.join(tmp_dir, 'localization_model_test.pth')

        pipeline = LocalizationPipeline(
            data_root=os.path.join(os.getcwd(), 'data_storage', 'tt100k_2021'),
            model_save_path=model_path,
            epochs=1
        )
        pipeline.run()

        assert os.path.isfile(model_path), "Localization model file was not saved."

        annotation_path = os.path.join(pipeline.data_root, 'train_val_annotations.json')
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        num_classes = len(annotations['types'])

        from road_sign_detection.models.localization_base_model import LocalizationNet
        model = LocalizationNet(num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 64, 64)
            output = model(dummy_input)
        assert output.shape[0] == 1, "Localization model batch size mismatch."

        print("Localization pipeline integration test passed.")
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_classification_pipeline()
    test_localization_pipeline()
    print("All integration tests passed.")
