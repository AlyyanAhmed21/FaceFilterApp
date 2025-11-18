import cv2
import torch
import torchvision
import mediapipe as mp
import numpy as np
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load face segmentation model (mobile-friendly)
seg_model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True).to(device)
seg_model.eval()

# MediaPipe face mesh for detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def process_image(image_path):
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    start = time.time()
    results = face_mesh.process(rgb)

    overlay = img.copy()
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Example: simple lips overlay using landmarks
            lips_indices = mp_face_mesh.FACEMESH_LIPS
            lips_points = []
            h, w, _ = img.shape
            for idx1, idx2 in lips_indices:
                pt1 = face_landmarks.landmark[idx1]
                lips_points.append([int(pt1.x*w), int(pt1.y*h)])
            lips_points = np.array(lips_points)
            cv2.fillPoly(overlay, [lips_points], color=(0,0,255))
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

    fps = 1/(time.time()-start)
    output_path = os.path.join("outputs", os.path.basename(image_path))
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"FPS: {fps:.2f}")
    return output_path, fps
