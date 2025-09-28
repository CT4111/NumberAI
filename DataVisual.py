import numpy as np
from PIL import Image

def displayImage(image_array):
    img_array = (random_image * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.show()
