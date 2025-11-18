# template.py
import os

# Base directories
dirs = [
    "FaceFilterApp/app/templates",
    "FaceFilterApp/app/static",
    "FaceFilterApp/models",
    "FaceFilterApp/filters",
    "FaceFilterApp/utils"
]

# Create directories
for d in dirs:
    os.makedirs(d, exist_ok=True)

# requirements.txt
requirements = """fastapi
uvicorn
torch
torchvision
opencv-python
mediapipe
numpy
jinja2
"""
with open("FaceFilterApp/requirements.txt","w") as f:
    f.write(requirements)

# app/main.py
main_py = """from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from utils.inference import process_image
import shutil
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    output_path, fps = process_image(filepath)
    return FileResponse(output_path)
"""
with open("FaceFilterApp/app/main.py","w") as f:
    f.write(main_py)

# app/templates/index.html
index_html = """<!DOCTYPE html>
<html>
<head>
    <title>Face Filter App</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
<h2>Upload an image or video</h2>
<form action="/upload" enctype="multipart/form-data" method="post">
    <input type="file" name="file">
    <input type="submit" value="Apply Filter">
</form>
</body>
</html>"""
with open("FaceFilterApp/app/templates/index.html","w") as f:
    f.write(index_html)

# app/static/style.css
style_css = """body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
input[type=file] { margin: 20px 0; }"""
with open("FaceFilterApp/app/static/style.css","w") as f:
    f.write(style_css)

# utils/inference.py
inference_py = """import cv2
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
"""
with open("FaceFilterApp/utils/inference.py","w") as f:
    f.write(inference_py)

# run.py
run_py = """import uvicorn
if __name__ == '__main__':
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
"""
with open("FaceFilterApp/run.py","w") as f:
    f.write(run_py)

print("Template project created successfully in FaceFilterApp/")
