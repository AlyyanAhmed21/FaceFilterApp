import cv2
import numpy as np

def overlay_image(bg, fg, x, y, w, h):
    fg = cv2.resize(fg, (w, h))
    alpha = fg[:, :, 3] / 255.0

    for c in range(3):
        bg[y:y+h, x:x+w, c] = (
            fg[:, :, c] * alpha +
            bg[y:y+h, x:x+w, c] * (1 - alpha)
        )

def apply_glasses(frame, lm, asset):
    left = lm[33]
    right = lm[263]

    x1, y1 = int(left.x * frame.shape[1]), int(left.y * frame.shape[0])
    x2, y2 = int(right.x * frame.shape[1]), int(right.y * frame.shape[0])

    w = abs(x2 - x1) + 60
    h = int(w * 0.4)

    cx = int((x1 + x2) / 2 - w/2)
    cy = int((y1 + y2) / 2 - h/2)

    overlay_image(frame, asset, cx, cy, w, h)
