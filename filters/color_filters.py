import cv2
import numpy as np

def warm_filter(img):
    lookup = np.interp(img, [0, 255], [20, 255])
    return lookup.astype(np.uint8)

def cool_filter(img):
    lookup = np.interp(img, [0, 255], [0, 230])
    return lookup.astype(np.uint8)

def cartoon_filter(img):
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 7)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 5)
    color = cv2.bilateralFilter(img, 12, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon
