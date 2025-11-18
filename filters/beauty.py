import cv2
import numpy as np

def smooth_skin(frame):
    blur = cv2.bilateralFilter(frame, 15, 80, 80)
    return blur

def whiten_teeth(frame, lm):
    h, w = frame.shape[:2]
    mouth_pts = [lm[i] for i in [13, 14, 87, 317]]

    mask = np.zeros((h, w), dtype=np.uint8)
    pts = []
    for p in mouth_pts:
        pts.append((int(p.x * w), int(p.y * h)))

    cv2.fillConvexPoly(mask, np.array(pts), 255)
    frame[mask > 0] = np.clip(frame[mask > 0] + 40, 0, 255)
    return frame
