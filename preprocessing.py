from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

def add_padding(image, padding=50):
    background_color = image[0, 0]
    padded = np.ones((image.shape[0] + 2 * padding, image.shape[1] + 2 * padding, 3), dtype=np.uint8) * background_color
    padded[padding:image.shape[0] + padding, padding:image.shape[1] + padding] = image
    return padded

def too_thick_check(image):
    white = (image == 255).sum()
    total = image.size
    return (white / total) > 0.05

def too_thin_check(image):
    white = (image == 255).sum()
    total = image.size
    return (white / total) < 0.01

def process_image(image, max_iterations=10):
    count = 0
    for _ in range(max_iterations):
        if too_thick_check(image):
            erosion_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            image = cv2.erode(image, erosion_element)
            count = count + 1
        elif too_thin_check(image):
            dilation_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            image = cv2.dilate(image, dilation_element)
            count = count + 1
        else:
            break
    print(count)
    return image

def rescale(image):
    ratio = image.width / image.height
    new_width = int(800 * ratio)
    resized = image.resize((new_width, 800))
    return resized

def get_background_color(image_path):
    image = Image.open(image_path)
    background_color = image.getpixel((0, 0))
    return background_color