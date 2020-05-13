import cv2
import numpy as np

class DataAugmentation():
    def __init__(self):
        pass

    def flip_image_horizontal(self, image):
        """Return the image flipped horizontally"""
        temp_img = image.reshape(48,48)
        flipped_image = cv2.flip(temp_img, 1)
        return np.array(flipped_image.flatten())

    def augment_dataset(self, X, y):
        """Return original dataset augmented with images flipped horizontally"""
        flipped_images = np.array([self.flip_image_horizontal(img) for img in X])
        return np.concatenate((X, flipped_images)), np.concatenate((y,y))