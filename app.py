from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from sklearn.cluster import KMeans

app = FastAPI(title="Hybrid Vision & Weather API")

# CORS Enable karna taake Frontend (Vercel) isse connect ho sakay
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Architecture
class HybridVisionModel(nn.Module):
    def __init__(self, num_weather_classes=4, num_env_classes=2):
        super(HybridVisionModel, self).__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.weather_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_weather_classes)
        )
        self.env_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_env_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.weather_head(features), self.env_head(features)

# Device & Model Load karna
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridVisionModel()

try:
    model.load_state_dict(torch.load('hybrid_vision_model.pth', map_location=device))
    print("Model weights loaded successfully!")
except Exception as e:
    print(f"Error loading model weights: {e}")

model.to(device)
model.eval()

WEATHER_CLASSES = ['Sunny', 'Cloudy', 'Rainy', 'Foggy']
ENV_CLASSES = ['Indoor', 'Outdoor']

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# K-Means Color Percentage Function
def analyze_colors(image, k=3):
    resized = cv2.resize(image, (100, 100))
    pixels = resized.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=k, n_init=5)
    kmeans.fit(pixels)
    
    counts = np.bincount(kmeans.labels_)
    total_pixels = len(pixels)
    
    colors_info = []
    for i in range(k):
        color_bgr = kmeans.cluster_centers_[i].astype(int).tolist()
        percentage = round((counts[i] / total_pixels) * 100, 2)
        colors_info.append({"color_bgr": color_bgr, "percentage": percentage})
        
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)].astype(int).tolist()
    return dominant_color, colors_info

@app.get("/")
def home():
    return {"status": "Running", "message": "Hybrid Vision API is active!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert('RGB')
    
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Model Prediction
    input_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        w_out, e_out = model(input_tensor)
        weather_pred = WEATHER_CLASSES[torch.argmax(w_out).item() % len(WEATHER_CLASSES)]
        env_pred = ENV_CLASSES[torch.argmax(e_out).item() % len(ENV_CLASSES)]
        
    # Brightness Check
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    lighting_cond = "Bright" if brightness > 110 else "Dark/Low-Light"
    
    # Color Analysis
    dominant_bgr, top_colors = analyze_colors(image_bgr, k=3)
    
    return {
        "weather": weather_pred,
        "environment": env_pred,
        "lighting": lighting_cond,
        "brightness_value": round(brightness, 2),
        "dominant_color_bgr": dominant_bgr,
        "color_percentages": top_colors
    }
