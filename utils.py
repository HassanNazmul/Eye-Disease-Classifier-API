# utils.py

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

IMAGE_SIZE = 224

# Your class labels (in order)
class_names = [
    "Central Serous Chorioretinopathy",
    "Diabetic Retinopathy",
    "Disc Edema",
    "Glaucoma",
    "Healthy",
    "Macular Scar",
    "Myopia",
    "Pterygium",
    "Retinal Detachment",
    "Retinitis Pigmentosa"
]


def preprocess_image(image):
    # Resize, normalize and add batch dimension
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = img_to_array(image)
    image_array = image_array / 255.0  # normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # shape: (1, 224, 224, 3)
    return image_array.astype(np.float32)
