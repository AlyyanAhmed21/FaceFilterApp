import cv2
from filters.overlays import overlay_image

def apply_dog_filter(frame, lm, ears_asset, nose_asset):
    h, w = frame.shape[:2]

    # ears
    forehead = lm[10]
    x = int(forehead.x * w - ears_asset.shape[1] / 3)
    y = int(forehead.y * h - ears_asset.shape[0])
    overlay_image(frame, ears_asset, x, y, ears_asset.shape[1], ears_asset.shape[0])

    # nose
    nose = lm[1]
    x = int(nose.x * w - nose_asset.shape[1] / 2)
    y = int(nose.y * h - nose_asset.shape[0] / 2)
    overlay_image(frame, nose_asset, x, y, nose_asset.shape[1], nose_asset.shape[0])
