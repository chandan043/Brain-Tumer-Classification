from keras.models import load_model


def load_model_and_labels(model_path: str, labels_path: str):
    """Loads the model and labels."""
    model = load_model(model_path, compile=False)
    with open(labels_path, "r") as file:
        class_names = [line.strip() for line in file.readlines()]
    return model, class_names
