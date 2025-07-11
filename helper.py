import numpy as np
from PIL import Image

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def load_labels(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]
