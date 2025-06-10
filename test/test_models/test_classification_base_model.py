import torch
from road_sign_detection.models.classification_base_model import CNNClassifier


def test_cnn_classifier_instantiation() -> None:
    """
    Tests if CNNClassifier can be instantiated properly.
    """
    model = CNNClassifier(number_of_classes=232)
    assert isinstance(model, CNNClassifier), "Model is not an instance of CNNClassifier"


def test_cnn_classifier_forward_pass_shape() -> None:
    """
    Tests if the forward pass returns the expected output shape.
    """
    model = CNNClassifier(number_of_classes=232)
    dummy_input: torch.Tensor = torch.randn(4, 3, 64, 64)  # batch_size=4
    output: torch.Tensor = model(dummy_input)
    assert output.shape == (4, 232), "Output shape mismatch for CNNClassifier"


def test_cnn_classifier_requires_grad() -> None:
    """
    Tests if all model parameters require gradients for training.
    """
    model = CNNClassifier(number_of_classes=232)
    grad_flags = [param.requires_grad for param in model.parameters()]
    assert all(grad_flags), "All CNNClassifier parameters should require gradients"