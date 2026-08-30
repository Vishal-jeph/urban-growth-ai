from PIL import Image
import numpy as np


def load_image(image_source):

    image = Image.open(image_source).convert("RGB")

    image = np.array(image)

    return image