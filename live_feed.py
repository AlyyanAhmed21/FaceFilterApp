# live_feed.py
# Run: python live_feed.py
# Requirements: ultralytics, mediapipe, torch, torchvision, opencv-python, numpy
# Models expected in ./models/:
# - yolov8n-face-lindevs.pt
# - face_detection_full_range.tflite (not used directly here; MediaPipe used)
# - face_landmarker_v2_with_blendshapes.task (MediaPipe internal)
# - best_deeplabv3_mobilenet_voc_os16.pth  (optional; loaded if present)

import os
import time
from collections import deque

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO
import mediapipe as mp

# -------------------------
# Config / Paths
# -------------------------
MODEL_DIR = "models"
YOLO_FACE_PATH = os.path.join(MODEL_DIR, "yolov8n-face-lindevs.pt")
DEEPLAB_PTH_PATH = os.path.join(MODEL_DIR, "best_deeplabv3_mobilenet_voc_os16.pth")
# Optional overlay assets folder (png with alpha)
ASSETS_DIR = "filters_assets"

# -------------------------
# Device
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Device:", DEVICE)

# -------------------------
# Load YOLO Face Detector (Ultralytics)
# -------------------------
if os.path.exists(YOLO_FACE_PATH):
    yolo = YOLO(YOLO_FACE_PATH)
    print(f"[INFO] Loaded YOLO face model from {YOLO_FACE_PATH}")
else:
    print("[WARN] YOLO face model not found at", YOLO_FACE_PATH)
    print("[INFO] Falling back to ultralytics default 'yolov8n' (no face-tuned weights).")
    yolo = YOLO("yolov8n.pt")

# -------------------------
# Load segmentation model (DeepLabv3 MobileNet) - optional
# -------------------------
seg_model = None
use_segmentation = False
try:
    seg_model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=False, aux_loss=None)
    if os.path.exists(DEEPLAB_PTH_PATH):
        state = torch.load(DEEPLAB_PTH_PATH, map_location="cpu")
        # Attempt to load state_dict robustly
        if "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            state = state["model_state_dict"]
        try:
            seg_model.load_state_dict(state, strict=False)
            print(f"[INFO] Loaded segmentation weights from {DEEPLAB_PTH_PATH}")
        except Exception as e:
            # fallback: try to load entire checkpoint (some checkpnts store differently)
            try:
                seg_model.load_state_dict(torch.load(DEEPLAB_PTH_PATH, map_location="cpu"), strict=False)
                print(f"[INFO] Loaded segmentation checkpoint from {DEEPLAB_PTH_PATH}")
            except Exception as e2:
                print("[WARN] Could not strictly load segmentation weights:", e)
                print("Proceeding with randomly initialized segmentation model.")
    else:
        print("[INFO] segmentation weights not found; using default (random) model. You can place your .pth at:", DEEPLAB_PTH_PATH)

    seg_model.to(DEVICE).eval()
    use_segmentation = True
except Exception as e:
    print("[WARN] segmentation model not available or failed to load:", e)
    seg_model = None
    use_segmentation = False

# -------------------------
# MediaPipe Face Mesh (landmarks)
# -------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=4,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# -------------------------
# Helpers: smoothing landmarks (EMA)
# -------------------------
class LandmarkSmoother:
    def __init__(self, alpha=0.6, max_faces=4):
        # alpha: EMA factor (higher = more weight to new)
        self.alpha = alpha
        # store per-face deque of last coords (for simple identity matching we use box centers)
        self.prev_landmarks = {}  # face_id -> np.array([N,3])
        self.max_faces = max_faces

    def smooth(self, face_id, landmarks):
        # landmarks: list of landmark objects (x,y,z floats normalized)
        arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        if face_id not in self.prev_landmarks:
            self.prev_landmarks[face_id] = arr
            return arr
        prev = self.prev_landmarks[face_id]
        if prev.shape != arr.shape:
            self.prev_landmarks[face_id] = arr
            return arr
        sm = self.alpha * arr + (1.0 - self.alpha) * prev
        self.prev_landmarks[face_id] = sm
        return sm

    def clear_old(self):
        # could implement TTL-based removal if needed
        pass

smoother = LandmarkSmoother(alpha=0.55, max_faces=4)

# -------------------------
# Asset loader (optional PNG overlays with alpha)
# -------------------------
def load_asset(filename):
    p = os.path.join(ASSETS_DIR, filename)
    if not os.path.exists(p):
        return None
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # Ensure BGRA
    if img.shape[2] == 3:
        b,g,r = cv2.split(img)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        img = cv2.merge([b,g,r,alpha])
    return img

# Try load some common assets (optional)
glasses_asset = load_asset("glasses.png")
dog_ears_asset = load_asset("dog_ears.png")
dog_nose_asset = load_asset("dog_nose.png")
crown_asset = load_asset("crown.png")
sunglasses_asset = load_asset("sunglasses.png")

# -------------------------
# Overlay utilities
# -------------------------
def overlay_image_alpha(bg, fg, x, y, w=None, h=None):
    """
    Overlay fg (with alpha channel) onto bg at (x,y).
    fg expected BGRA.
    """
    if fg is None:
        return bg
    fg_h, fg_w = fg.shape[:2]
    if w is None:
        w = fg_w
    if h is None:
        h = fg_h
    if w <= 0 or h <= 0:
        return bg
    fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_AREA)
    if x < 0:
        # crop left
        fg = fg[:, -x:]
        w = fg.shape[1]
        x = 0
    if y < 0:
        fg = fg[-y:, :, :]
        h = fg.shape[0]
        y = 0
    if x + w > bg.shape[1]:
        fg = fg[:, : bg.shape[1] - x]
        w = fg.shape[1]
    if y + h > bg.shape[0]:
        fg = fg[: bg.shape[0] - y, :, :]
        h = fg.shape[0]

    if fg.shape[2] < 4:
        # no alpha -> simple paste
        bg[y:y+h, x:x+w] = fg[:, :, :3]
        return bg

    alpha = fg[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+h, x:x+w, c] = (alpha * fg[:, :, c] + (1 - alpha) * bg[y:y+h, x:x+w, c])
    return bg

# -------------------------
# Small AR Filters (functions)
# -------------------------
def apply_glasses(frame, lm):
    # use eyes landmarks 33 (left) and 263 (right) as base
    h, w = frame.shape[:2]
    left = lm[33]
    right = lm[263]
    x1, y1 = int(left[0] * w), int(left[1] * h)
    x2, y2 = int(right[0] * w), int(right[1] * h)
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    width = int(1.6 * abs(x2 - x1))
    height = int(width * 0.45)
    x = center_x - width // 2
    y = center_y - height // 2
    return overlay_image_alpha(frame, glasses_asset, x, y, width, height)

def apply_sunglasses(frame, lm):
    h, w = frame.shape[:2]
    left = lm[33]; right = lm[263]
    x1, y1 = int(left[0] * w), int(left[1] * h)
    x2, y2 = int(right[0] * w), int(right[1] * h)
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    width = int(1.8 * abs(x2 - x1))
    height = int(width * 0.5)
    x = center_x - width // 2
    y = center_y - height // 2
    return overlay_image_alpha(frame, sunglasses_asset, x, y, width, height)

def apply_dog(frame, lm):
    h, w = frame.shape[:2]
    # forehead (approx landmark 10) for ears
    forehead = lm[10]
    x = int(forehead[0] * w) - (dog_ears_asset.shape[1] // 3 if dog_ears_asset is not None else 50)
    y = int(forehead[1] * h) - (dog_ears_asset.shape[0] if dog_ears_asset is not None else 100)
    if dog_ears_asset is not None:
        frame = overlay_image_alpha(frame, dog_ears_asset, x, y)
    # nose (landmark 1)
    nose = lm[1]
    nx = int(nose[0] * w) - (dog_nose_asset.shape[1] // 2 if dog_nose_asset is not None else 30)
    ny = int(nose[1] * h) - (dog_nose_asset.shape[0] // 2 if dog_nose_asset is not None else 20)
    if dog_nose_asset is not None:
        frame = overlay_image_alpha(frame, dog_nose_asset, nx, ny)
    return frame

def apply_crown(frame, lm):
    if crown_asset is None:
        return frame
    forehead = lm[10]
    h, w = frame.shape[:2]
    cw = int(w * 0.5)
    ch = int(cw * crown_asset.shape[0] / crown_asset.shape[1])
    cx = int(forehead[0] * w - cw // 2)
    cy = int(forehead[1] * h - ch - 20)
    return overlay_image_alpha(frame, crown_asset, cx, cy, cw, ch)

def apply_lip_color(frame, lm, color=(0,0,255), alpha=0.6):
    # create mask from landmark lip points (MediaPipe FACEMESH_LIPS indices)
    h, w = frame.shape[:2]
    lip_idx = mp_face.FACEMESH_LIPS
    pts = []
    # For MediaPipe, lip_idx is list of pairs; we will collect unique indices
    unique_idx = sorted(set([a for pair in lip_idx for a in pair]))
    for i in unique_idx:
        lmpt = lm[i]
        pts.append([int(lmpt[0] * w), int(lmpt[1] * h)])
    if len(pts) < 3:
        return frame
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(pts), 255)
    colored = np.zeros_like(frame, dtype=np.uint8)
    colored[:, :] = color
    # blend
    frame = np.where(mask[..., None] == 255,
                     (alpha * colored + (1-alpha) * frame).astype(np.uint8),
                     frame)
    return frame

def beauty_smooth(frame, lm=None):
    # A gentle bilateral filter to smooth skin while preserving edges.
    # Optionally use segmentation mask for selective smoothing (not implemented here)
    return cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

def cartoonify(frame):
    # simple cartoon effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  9, 2)
    color = cv2.bilateralFilter(frame, d=9, sigmaColor=250, sigmaSpace=250)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges_color)
    return cartoon

def blur_background(frame, seg_mask):
    # seg_mask: float HxW [0..1] for foreground (face); blur rest
    if seg_mask is None:
        # fallback: moderate whole-frame blur
        return cv2.GaussianBlur(frame, (21,21), 0)
    h, w = frame.shape[:2]
    mask = cv2.resize(seg_mask, (w, h))
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    blurred = cv2.GaussianBlur(frame, (51,51), 0)
    out = (frame * mask3 + blurred * (1 - mask3)).astype(np.uint8)
    return out

# -------------------------
# Utility: compute segmentation mask (if available)
# -------------------------
def run_segmentation(frame):
    if not use_segmentation or seg_model is None:
        return None
    # seg_model expects normalized tensor
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    # choose reasonable size (keep aspect)
    target = 512
    scale = target / max(H, W)
    newh, neww = int(H * scale), int(W * scale)
    img_resized = cv2.resize(img, (neww, newh))
    t = torchvision.transforms.functional.to_tensor(img_resized).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = seg_model(t)['out'][0]  # [C, H, W]
        preds = out.argmax(0).cpu().numpy().astype(np.uint8)
    # We assume class 15 (person) or similar, depending on training — map all non-zero to foreground
    mask = (preds > 0).astype(np.float32)  # simplistic
    mask_up = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
    mask_up = np.clip(mask_up, 0, 1)
    return mask_up

# -------------------------
# Main Live Feed Loop
# -------------------------
def run_live_feed(device=DEVICE):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    # State toggles
    active_filter = 0
    # filter mapping:
    # 0: none
    # 1: glasses
    # 2: sunglasses
    # 3: beauty smooth
    # 4: lip color
    # 5: dog filter
    # 6: crown
    # 7: cartoon
    # 8: background blur (requires seg)
    last_time = time.time()
    fps_smooth = deque(maxlen=30)
    face_id_counter = 0

    print("[INFO] Controls: keys 0..9 toggle filters, q to quit")
    print(" 0: none | 1: glasses | 2: sunglasses | 3: beauty | 4: lip color | 5: dog | 6: crown | 7: cartoon | 8: bg blur")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        img_h, img_w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLO detection (fast)
        # yolo(frame) returns Results; take first
        try:
            results = yolo(frame, augment=False, verbose=False)[0]
        except Exception as e:
            # if calling with frame directly fails, use numpy interface
            results = yolo.predict(source=frame, verbose=False)[0]

        # prepare segmentation mask if needed
        seg_mask = None
        if active_filter == 8 and use_segmentation:
            seg_mask = run_segmentation(frame)

        # iterate detections
        # ultralytics results.boxes (list) with xyxy, conf, cls
        faces = []
        if hasattr(results, "boxes") and len(results.boxes) > 0:
            for b in results.boxes:
                # b.xyxy might be tensor
                xyxy = b.xyxy.cpu().numpy().astype(int)[0] if hasattr(b.xyxy, "cpu") else np.array(b.xyxy).astype(int)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                # clamp
                x1 = max(0, min(x1, img_w - 1)); y1 = max(0, min(y1, img_h - 1))
                x2 = max(0, min(x2, img_w - 1)); y2 = max(0, min(y2, img_h - 1))
                faces.append((x1, y1, x2, y2))

        # if no faces found, optionally fallback to MediaPipe face detection via whole frame for landmarks
        # We'll use YOLO ROIs for speed; for each face ROI, run MediaPipe on the cropped ROI and map landmarks
        final_frame = frame.copy()

        face_index = 0
        for (x1, y1, x2, y2) in faces:
            face_index += 1
            # draw box
            cv2.rectangle(final_frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

            # ROI for MediaPipe (smaller -> faster). MediaPipe expects RGB full image; we'll process resized ROI
            roi = frame_rgb[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            # MediaPipe expects full image scale coords, but since we feed ROI we must map back
            roi_h, roi_w = roi.shape[:2]
            # process
            mp_res = face_mesh.process(roi)

            if not mp_res.multi_face_landmarks:
                continue
            # If multiple landmarks returned in ROI, take first
            mp_lms = mp_res.multi_face_landmarks[0].landmark

            # Map landmarks to normalized coords in full frame
            mapped = []
            for lm in mp_lms:
                # lm.x, lm.y are relative to ROI
                fx = x1 + lm.x * (x2 - x1)
                fy = y1 + lm.y * (y2 - y1)
                mapped.append([fx / img_w, fy / img_h, lm.z])  # normalized w.r.t full frame
            mapped = np.array(mapped, dtype=np.float32)

            # Use a face id based on position (very simple)
            fid = f"face_{face_index}"
            smoothed = smoother.smooth(fid, [type('p', (), {'x':p[0], 'y':p[1], 'z':p[2]}) for p in mapped])
            # smoothed is Nx3 array with normalized coords
            # convert to list of dict-like for filter functions
            lm_list = [{"x":p[0], "y":p[1], "z":p[2]} for p in smoothed]

            # prepare a simpler array of landmarks for math (np.array of [x,y])
            lm_np = np.array([[p["x"], p["y"], p["z"]] for p in lm_list], dtype=np.float32)

            # Apply chosen filter(s)
            if active_filter == 1:
                if glasses_asset is not None:
                    final_frame = apply_glasses(final_frame, lm_np)
                else:
                    # fallback: draw line between eyes
                    left = lm_np[33]; right = lm_np[263]
                    xL, yL = int(left[0]*img_w), int(left[1]*img_h)
                    xR, yR = int(right[0]*img_w), int(right[1]*img_h)
                    cv2.line(final_frame, (xL,yL),(xR,yR),(0,0,0),10)
            elif active_filter == 2:
                if sunglasses_asset is not None:
                    final_frame = apply_sunglasses(final_frame, lm_np)
            elif active_filter == 3:
                final_frame = beauty_smooth(final_frame)
            elif active_filter == 4:
                final_frame = apply_lip_color(final_frame, lm_np, color=(0,0,200), alpha=0.6)
            elif active_filter == 5:
                final_frame = apply_dog(final_frame, lm_np)
            elif active_filter == 6:
                final_frame = apply_crown(final_frame, lm_np)
            elif active_filter == 7:
                final_frame = cartoonify(final_frame)
            elif active_filter == 8:
                if seg_mask is not None:
                    final_frame = blur_background(final_frame, seg_mask)
                else:
                    # fallback: mild blur of background area (outside box)
                    bg = final_frame.copy()
                    bg = cv2.GaussianBlur(bg, (41,41), 0)
                    mask = np.zeros_like(final_frame[:, :, 0], dtype=np.uint8)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    mask3 = np.repeat((mask==255)[:, :, None], 3, axis=2)
                    final_frame = np.where(mask3, final_frame, bg)
            # else 0: no effect

        # FPS
        now = time.time()
        fps = 1.0 / (now - last_time) if now != last_time else 0.0
        last_time = now
        fps_smooth.append(fps)
        fps_avg = sum(fps_smooth) / len(fps_smooth)

        cv2.putText(final_frame, f"FPS: {fps:.1f} (avg {fps_avg:.1f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Draw UI hint
        cv2.putText(final_frame, f"Filter: {active_filter}", (10, final_frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("FaceFilterApp - Live AR", final_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # toggle filters by numeric keys
        if key in [ord(str(i)) for i in range(10)]:
            active_filter = int(chr(key))
            print("[INFO] Active filter set to", active_filter)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_feed()
