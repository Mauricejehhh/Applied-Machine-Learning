import torch
from road_sign_detection.models.localization_base_model import BboxRegression


def test_bbox_regression_instantiation() -> None:
    """
    Tests if BboxRegression model is instantiated correctly.
    """
    model = BboxRegression()
    assert isinstance(model, BboxRegression), "Model is not an instance of BboxRegression"


def test_bbox_regression_forward_pass_shape() -> None:
    """
    Tests whether the forward pass of BboxRegression outputs the correct shape.
    """
    model = BboxRegression()
    dummy_input: torch.Tensor = torch.randn(2, 3, 224, 224)
    output: torch.Tensor = model(dummy_input)
    assert output.shape == (2, 4), "Output shape mismatch for BboxRegression"


def test_bbox_regression_output_range() -> None:
    """
    Tests if the output values are within the expected range [0, 1].
    """
    model = BboxRegression()
    dummy_input: torch.Tensor = torch.randn(2, 3, 224, 224)
    output: torch.Tensor = model(dummy_input)
    assert torch.all((0.0 <= output) & (output <= 1.0)), "Output values not in [0, 1]"


def test_bbox_regression_backbone_frozen() -> None:
    """
    Tests if the backbone of the BboxRegression model is frozen (non-trainable).
    """
    model = BboxRegression()
    frozen_params = [p.requires_grad for p in model.base_model.parameters()]
    assert not any(frozen_params), "Backbone parameters should be frozen"
