import pytest

from stockvision.models import available_models, get_model


def test_registry_contains_expected_models():
    expected = {"lstm", "bilstm", "gru", "cnn_lstm", "transformer", "tcn", "linear"}
    assert expected <= set(available_models())


def test_get_model_unknown_raises():
    with pytest.raises(KeyError):
        get_model("does-not-exist")


def test_metadata_has_name_and_description():
    for name in available_models():
        meta = get_model(name).metadata
        assert meta.name == name
        assert meta.description
