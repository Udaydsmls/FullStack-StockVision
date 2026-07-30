"""A small plugin registry so a new architecture is a one-file change."""

MODELS = {}


def register(name, description, backend="keras", build=None, fit=None, predict_next=None):
    """Add one architecture to the registry.

    Keras models pass ``build(window, num_features)`` and are exported to ONNX.
    Classical models (Prophet, AutoARIMA) pass ``fit(df)`` and
    ``predict_next(fitted, df)`` instead, and are pickled with joblib.
    """
    if name in MODELS:
        raise ValueError(f"Model already registered: {name}")
    MODELS[name] = {
        "name": name,
        "description": description,
        "backend": backend,
        "build": build,
        "fit": fit,
        "predict_next": predict_next,
    }


def get_model(name):
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {available_models()}")
    return MODELS[name]


def available_models():
    return sorted(MODELS)
