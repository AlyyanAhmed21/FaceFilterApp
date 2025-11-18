import cv2

def blur_background(frame, mask):
    blurred = cv2.GaussianBlur(frame, (45,45), 0)
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    return (frame * mask + blurred * (1 - mask)).astype(np.uint8)
