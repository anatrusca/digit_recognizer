import numpy as np
from PIL import Image

def preprocess_image(img_array):
    img = Image.fromarray(img_array).convert("L").resize((28, 28))
    img_np = np.array(img)
    img_np = 1.0 - img_np / 255.0
    return img_np.reshape(1, 28, 28, 1)
