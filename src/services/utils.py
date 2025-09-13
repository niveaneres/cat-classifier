from PIL import Image
import cv2
import io
import numpy as np
 

def transform(img):
    image = Image.open(io.BytesIO(img)).convert("RGB")
    roi = np.array(image)
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    return roi