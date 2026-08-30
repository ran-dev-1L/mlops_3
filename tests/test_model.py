import torch

from model import SimpleCNN, get_model


def test_output_shape():
    model = SimpleCNN(num_classes=10)
    dummy_input = torch.randn(4, 1, 28, 28)  # batch of 4 grayscale 28x28 images
    output = model(dummy_input)
    assert output.shape == (4, 10)


def test_get_model_returns_simple_cnn():
    model = get_model(architecture="simple_cnn", num_classes=10)
    assert isinstance(model, SimpleCNN)


def test_get_model_invalid_architecture_raises():
    try:
        get_model(architecture="not_a_real_model", num_classes=10)
        assert False, "Expected ValueError for unknown architecture"
    except ValueError:
        pass