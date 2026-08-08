import cv2
import torch
import numpy as np
import pyvista as pv
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import sys
sys.path.append("MiDaS")  # assuming MiDaS folder is in the same directory
from midas.dpt_depth import DPTDepthModel
# Load MiDaS model
model_type = "dpt_large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DPTDepthModel(
    path="weights/dpt_large-midas-2f21e586.pt",
    backbone="vitl16_384",
    non_negative=True,
)
model.eval().to(device)

# Transform
transform = Compose([
    Resize(384, 384, resize_target=None),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet()
])

# Load and preprocess image
img = cv2.imread("input.jpg")[:, :, ::-1]  # BGR to RGB
img_input = transform({"image": img})["image"]
input_tensor = torch.from_numpy(img_input).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    prediction = model(input_tensor)
    depth = prediction.squeeze().cpu().numpy()

# Resize depth to match original
depth = cv2.resize(depth, (img.shape[1], img.shape[0]))

# Create point cloud from depth
h, w = depth.shape
focal_length = 0.8 * w  # You can tweak this
cx, cy = w / 2, h / 2

i, j = np.meshgrid(np.arange(w), np.arange(h))
z = depth
x = (i - cx) * z / focal_length
y = (j - cy) * z / focal_length

points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
colors = img.reshape(-1, 3) / 255.0  # Normalize RGB

# Create PyVista point cloud
cloud = pv.PolyData(points)
cloud["RGB"] = colors

# 🧪 Optional: surface reconstruction
mesh = cloud.delaunay_2d(alpha=1.0)  # You can experiment with alpha

# Save to file
mesh.save("reconstructed_mesh.ply")  # or .vtk, .stl, etc.
print("✅ Saved to 'reconstructed_mesh.ply'")