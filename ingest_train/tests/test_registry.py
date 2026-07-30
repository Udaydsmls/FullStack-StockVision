import pytest

from stockvision.models import available_models, get_model


def test_all_nine_architectures_are_registered():
    expected = {
        "lstm", "bilstm", "gru", "cnn_lstm", "transformer", "tcn", "linear",
        "prophet", "autoarima",
    }
    assert expected == set(available_models())


def test_keras_models_can_build_a_graph():
    for name in available_models():
        model = get_model(name)
        if model["backend"] == "keras":
            assert model["build"] is not None


def test_classical_models_fit_instead_of_building():
    for name in ("prophet", "autoarima"):
        model = get_model(name)
        assert model["backend"] != "keras"
        assert model["fit"] is not None
        assert model["predict_next"] is not None


def test_every_model_has_a_name_and_description():
    for name in available_models():
        model = get_model(name)
        assert model["name"] == name
        assert model["description"]


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        get_model("does-not-exist")
