# import the necessary packages
import tensorflow as tf
from tensorflow.keras.models import load_model
# from imutils.contours import sort_contours
import numpy as np
# import imutils
import cv2
from PIL import Image
import matplotlib.pyplot as plt

"""
Helper functions for ocr project
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

SMALL_HEIGHT = 800

def implt(img, cmp=None, t=''):
    """Show image using plt."""
    plt.imshow(img, cmap=cmp)
    plt.title(t)
    plt.show()


def resize(img, height=SMALL_HEIGHT, always=False):
    """Resize image to given height."""
    if (img.shape[0] > height or always):
        rat = height / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), height))

    return img


def ratio(img, height=SMALL_HEIGHT):
    """Getting scale ratio."""
    return img.shape[0] / height


def img_extend(img, shape):
    """Extend 2D image (numpy array) in vertical and horizontal direction.
    Shape of result image will match 'shape'
    Args:
        img: image to be extended
        shape: shape (touple) of result image
    Returns:
        Extended image
    """
    x = np.zeros(shape, np.uint8)
    x[:img.shape[0], :img.shape[1]] = img
    return x

"""
Detect words on the page
return array of words' bounding boxes
"""
def detection(image, join=False):
    blurred_img = cv2.GaussianBlur(image, (5, 5), 18)
    edge_img = _edge_detect(image)
    result, edge_img = cv2.threshold(edge_img, 50, 255, cv2.THRESH_BINARY)
    bnw_img = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE,
                              np.ones((15,15), np.uint8))
    return _text_detect(bnw_img, image, join)

def sort_words(boxes):
    avg_height = sum([y2 - y1 for _, y1, _, y2 in boxes]) / len(boxes)
    boxes.view('i8,i8,i8,i8').sort(order=['f1'], axis=0)
    curr_line = boxes[0][1]
    lines = []
    temp_line = []
    for box in boxes:
        if box[1] > curr_line + avg_height:
            lines.append(temp_line)
            temp_line = [box]
            curr_line = box[1]
            continue
        temp_line.append(box)
    lines.append(temp_line)
    for line in lines:
        line.sort(key=lambda box: box[0])
    return lines

def _edge_detect(image):
    return np.max(np.array([_sobel_detect(image[:,:, 0]),
                            _sobel_detect(image[:,:, 1]),
                            _sobel_detect(image[:,:, 2])]), axis=0)

def _sobel_detect(channel):
    sobel_x = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobel_y = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobel_x, sobel_y)
    sobel[sobel > 255] = 255
    return np.uint8(sobel)

def union(a,b):
    x_coord = min(a[0], b[0])
    y_coord = min(a[1], b[1])
    width = max(a[0]+a[2], b[0]+b[2]) - x_coord
    height = max(a[1]+a[3], b[1]+b[3]) - y_coord
    return [x_coord, y_coord, width, height]

def _intersect(a,b):
    x_coord = max(a[0], b[0])
    y_coord = max(a[1], b[1])
    width = min(a[0]+a[2], b[0]+b[2]) - x_coord
    height = min(a[1]+a[3], b[1]+b[3]) - y_coord
    if width < 0 or height < 0:
        return False
    return True

def _group_rectangles(rects):
    test = [False for i in range(len(rects))]
    final_groups = []
    i = 0
    while i < len(rects):
        if not test[i]:
            j = i+1
            while j < len(rects):
                if not test[j] and _intersect(rects[i], rects[j]):
                    rects[i] = union(rects[i], rects[j])
                    test[j] = True
                    j = i
                j += 1
            final_groups += [rects[i]]
        i += 1

    return final_groups

def _text_detect(img, image, join=False):
    small_img = resize(img, 2000)
    mask = np.zeros(small_img.shape, np.uint8)
    contours, hierarchy = cv2.findContours(np.copy(small_img),cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    index = 0
    boxes = []
    while (index >= 0):
        x_coord, y_coord, width, height = cv2.boundingRect(contours[index])
        cv2.drawContours(mask, contours, index, (255, 255, 255), cv2.FILLED)
        maskROI = mask[y_coord:y_coord+height, x_coord:x_coord+width]
        r = cv2.countNonZero(maskROI) / (width * height)
        if (r > 0.05
            and 1600 > width > 10
            and 1600 > height > 10
            and height/width < 3
            and width/height < 10
            and (60 // height) * width < 1000):
            x_coord -= 25
            y_coord -= 30
            width += 50
            height += 50
            boxes += [[x_coord, y_coord, width, height]]
        index = hierarchy[0][index][0]
    if join:
        boxes = _group_rectangles(boxes)

    bounding_boxes = np.array([0, 0, 0, 0])
    for (x_coord, y_coord, width, height) in boxes:
        cv2.rectangle(image, (x_coord, y_coord), (x_coord + width, y_coord + height), (0, 255, 0), 2)
        bounding_boxes = np.vstack((bounding_boxes,
                                    np.array([x_coord, y_coord, x_coord + width, y_coord + height])))

    boxes = bounding_boxes.dot(ratio(image, small_img.shape[0])).astype(np.int64)
    return boxes[1:]
# def detection(image, join=False):
#     """Detecting the words bounding boxes.
#     Return: numpy array of bounding boxes [x, y, x+w, y+h]
#     """
#     # Preprocess image for word detection
#     blurred = cv2.GaussianBlur(image, (5, 5), 18)
#     edge_img = _edge_detect(blurred)
#     ret, edge_img = cv2.threshold(edge_img, 50, 255, cv2.THRESH_BINARY)
#     bw_img = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE,
#                               np.ones((15,15), np.uint8))

#     return _text_detect(bw_img, image, join)


# def sort_words(boxes):
#     """Sort boxes - (x, y, x+w, y+h) from left to right, top to bottom."""
#     mean_height = sum([y2 - y1 for _, y1, _, y2 in boxes]) / len(boxes)

#     boxes.view('i8,i8,i8,i8').sort(order=['f1'], axis=0)
#     current_line = boxes[0][1]
#     lines = []
#     tmp_line = []
#     for box in boxes:
#         if box[1] > current_line + mean_height:
#             lines.append(tmp_line)
#             tmp_line = [box]
#             current_line = box[1]
#             continue
#         tmp_line.append(box)
#     lines.append(tmp_line)

#     for line in lines:
#         line.sort(key=lambda box: box[0])

#     return lines


# def _edge_detect(im):
#     """
#     Edge detection using sobel operator on each layer individually.
#     Sobel operator is applied for each image layer (RGB)
#     """
#     return np.max(np.array([_sobel_detect(im[:,:, 0]),
#                             _sobel_detect(im[:,:, 1]),
#                             _sobel_detect(im[:,:, 2])]), axis=0)


# def _sobel_detect(channel):
#     """Sobel operator."""
#     sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
#     sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
#     sobel = np.hypot(sobelX, sobelY)
#     sobel[sobel > 255] = 255
#     return np.uint8(sobel)


# def union(a,b):
#     x = min(a[0], b[0])
#     y = min(a[1], b[1])
#     w = max(a[0]+a[2], b[0]+b[2]) - x
#     h = max(a[1]+a[3], b[1]+b[3]) - y
#     return [x, y, w, h]

# def _intersect(a,b):
#     x = max(a[0], b[0])
#     y = max(a[1], b[1])
#     w = min(a[0]+a[2], b[0]+b[2]) - x
#     h = min(a[1]+a[3], b[1]+b[3]) - y
#     if w<0 or h<0:
#         return False
#     return True

# def _group_rectangles(rec):
#     """
#     Uion intersecting rectangles.
#     Args:
#         rec - list of rectangles in form [x, y, w, h]
#     Return:
#         list of grouped ractangles
#     """
#     tested = [False for i in range(len(rec))]
#     final = []
#     i = 0
#     while i < len(rec):
#         if not tested[i]:
#             j = i+1
#             while j < len(rec):
#                 if not tested[j] and _intersect(rec[i], rec[j]):
#                     rec[i] = union(rec[i], rec[j])
#                     tested[j] = True
#                     j = i
#                 j += 1
#             final += [rec[i]]
#         i += 1

#     return final

# def _text_detect(img, image, join=False):
#     """Text detection using contours."""
#     small = resize(img, 2000)

#     # Finding contours
#     mask = np.zeros(small.shape, np.uint8)
#     cnt, hierarchy = cv2.findContours(np.copy(small),
#                                       cv2.RETR_CCOMP,
#                                       cv2.CHAIN_APPROX_SIMPLE)

#     index = 0
#     boxes = []
#     # Go through all contours in top level
#     while (index >= 0):
#         x, y, w, h = cv2.boundingRect(cnt[index])
#         cv2.drawContours(mask, cnt, index, (255, 255, 255), cv2.FILLED)
#         maskROI = mask[y:y+h, x:x+w]
#         # Ratio of white pixels to area of bounding rectangle
#         r = cv2.countNonZero(maskROI) / (w * h)

#         # Limits for text
#         if (r > 0.1
#             and 1600 > w > 10
#             and 1600 > h > 10
#             and h/w < 3
#             and w/h < 10
#             and (60 // h) * w < 1000):

#             # Increase the size of the bounding box by 5 pixels
#             x -= 25
#             y -= 30
#             w += 50
#             h += 50

#             boxes += [[x, y, w, h]]

#         index = hierarchy[0][index][0]

#     if join:
#         # Need more work
#         boxes = _group_rectangles(boxes)

#     # image for drawing bounding boxes
#     small = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
#     bounding_boxes = np.array([0, 0, 0, 0])
#     for (x, y, w, h) in boxes:
#         cv2.rectangle(small, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         bounding_boxes = np.vstack((bounding_boxes,
#                                     np.array([x, y, x+w, y+h])))

#     # implt(small, t='Bounding rectangles')

#     boxes = bounding_boxes.dot(ratio(image, small.shape[0])).astype(np.int64)
#     return boxes[1:]

def preprocess_image(image):
    image = cv2.resize(image, (300, 300))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32') / 255.0
    return image


