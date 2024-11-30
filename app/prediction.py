from PIL import Image, ImageOps
import numpy as np


def preprocess_image(image_data):
    """Preprocesses the image for prediction."""
    size = (224, 224)
    image = Image.open(image_data).convert("RGB")
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array


def predict_image(file, model, class_names):
    """Predicts the class of the image."""
    image_data = file.file
    image_array = preprocess_image(image_data)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score
