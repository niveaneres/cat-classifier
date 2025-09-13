import cv2
import numpy as np
import os
from .utils import transform


class ModelHandler:
    def __init__(self):
        self.prototxt = os.getenv("PROTOTXT_DIR") 
        self.model = os.getenv("MODEL_DIR")
        self.min_conf = 0.8
        self.load()

    def load(self):
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)

    def inference(self, img):
        roi = transform(img)
        blob = cv2.dnn.blobFromImage(cv2.resize(roi, (416, 416)), 0.013843, (416, 416), (110, 110, 110), swapRB=True)
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.min_conf:
                idx = int(detections[0, 0, i, 1])
                if idx == 8:
                    return "cat"
        return "it's not a cat"
            
