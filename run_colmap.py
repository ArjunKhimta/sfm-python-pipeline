import os
import subprocess
import shutil
import sys

# --- CONFIGURATION ---
COLMAP_BIN = "colmap" 
image_dir = "images"
workspace = "workspace"

# Define internal paths
database_path = os.path.join(workspace, "database.db")
sparse_dir = os.path.join(workspace, "sparse")

print("--- 🚜 Starting SFM Tractor Project (Mac Lite Version) ---")

# 1. FORCE CLEANUP (Fixes the "Cannot read image" error)
if os.path.exists(workspace):
    print("🧹 Cleaning up old workspace data...")
    shutil.rmtree(workspace)

os.makedirs(workspace, exist_ok=True)
os.makedirs(sparse_dir, exist_ok=True)

def run_command(cmd, description):
    print(f"\n🚀 {description}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during: {description}")
        sys.exit(1)

# 2. Feature Extraction (CPU Only)
run_command([
    COLMAP_BIN, "feature_extractor",
    "--database_path", database_path,
    "--image_path", image_dir,
    "--ImageReader.camera_model", "PINHOLE",
    "--SiftExtraction.use_gpu", "0"
], "Extracting Features")

# 3. Feature Matching (CPU Only)
run_command([
    COLMAP_BIN, "exhaustive_matcher",
    "--database_path", database_path,
    "--SiftMatching.use_gpu", "0"
], "Matching Features")

# 4. Sparse Reconstruction
run_command([
    COLMAP_BIN, "mapper",
    "--database_path", database_path,
    "--image_path", image_dir,
    "--output_path", sparse_dir
], "Running Sparse Mapper")

# 5. Convert to PLY (So you can view it)
# We find the output folder (usually '0') and convert it
sparse_0_path = os.path.join(sparse_dir, "0")
output_ply = os.path.join(workspace, "tractor_sparse.ply")

if os.path.exists(sparse_0_path):
    print("\n📦 Converting Sparse Cloud to PLY for MeshLab...")
    subprocess.run([
        COLMAP_BIN, "model_converter",
        "--input_path", sparse_0_path,
        "--output_path", output_ply,
        "--output_type", "PLY"
    ])
    print(f"\n✅ SUCCESS! Your Mac cannot do Dense reconstruction (No NVIDIA GPU),")
    print(f"   but we successfully created the Sparse Point Cloud.")
    print(f"📂 OPEN THIS FILE IN MESHLAB: {output_ply}")
else:
    print("\n⚠️  Reconstruction failed. No model was created.")
    print("   (This usually means the images were too hard to match).")