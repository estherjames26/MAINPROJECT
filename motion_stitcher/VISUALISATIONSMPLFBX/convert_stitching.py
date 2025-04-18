import os
import pickle
import subprocess
import argparse
from FbxReadWriter import FbxReadWrite
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--pkl", type=str, help="Path to SMPL .pkl file")
args = parser.parse_args()

PKL_PATH = args.pkl
FBX_TEMPLATE = r"C:\Users\kemij\Programming\SMPL-to-FBX-main\SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx"
OUTPUT_FOLDER = r"C:\Users\kemij\Programming\MAINPROJECT\motion_stitcher\output\choreography"
BLENDER_SCRIPT = os.path.join(OUTPUT_FOLDER, "import_group_fbx.py")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load choreography data
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)
poses = data.get("smpl_poses")
trans = data.get("smpl_trans")

# Determine if group mode and frame count
if isinstance(poses, np.ndarray) and poses.ndim == 3:
    group_mode = True
    frame_count = poses.shape[1]
else:
    group_mode = False
    frame_count = poses.shape[0]

# Export FBX files
fbx_files = []
if group_mode:
    for i in range(poses.shape[0]):
        smpl_params = {
            "smpl_poses": poses[i],
            "smpl_trans": (trans[i] if isinstance(trans, np.ndarray) and trans.ndim == 3 else np.zeros((poses.shape[1], 3)))
        }
        fbx_name = f"dancer_{i+1}.fbx"
        writer = FbxReadWrite(FBX_TEMPLATE)
        writer.addAnimation(pkl_filename=f"dancer_{i+1}", smpl_params=smpl_params)
        writer.writeFbx(OUTPUT_FOLDER, fbx_name)
        writer.destroy()
        fbx_files.append(os.path.join(OUTPUT_FOLDER, fbx_name))
else:
    smpl_params = {"smpl_poses": poses, "smpl_trans": trans}
    fbx_name = "solo_dancer.fbx"
    writer = FbxReadWrite(FBX_TEMPLATE)
    writer.addAnimation(pkl_filename="solo_dancer", smpl_params=smpl_params)
    writer.writeFbx(OUTPUT_FOLDER, fbx_name)
    writer.destroy()
    fbx_files.append(os.path.join(OUTPUT_FOLDER, fbx_name))

# Generate Blender import script with dynamic frame range
with open(BLENDER_SCRIPT, "w") as f:
    f.write("import bpy\n\n")
    # Remove default cube
    f.write("if 'Cube' in bpy.data.objects:\n")
    f.write("    bpy.data.objects['Cube'].select_set(True)\n")
    f.write("    bpy.ops.object.delete()\n\n")
    # Set FPS and frame range based on choreography length
    f.write("bpy.context.scene.render.fps = 30\n")
    f.write("bpy.context.scene.render.fps_base = 1.0\n")
    f.write("bpy.context.scene.frame_start = 1\n")
    f.write(f"bpy.context.scene.frame_end = {frame_count}\n\n")
    # Import FBX files
    f.write("fbx_files = [\n")
    for path in fbx_files:
        f.write(f"    r'{path}',\n")
    f.write("]\n\n")
    f.write("spacing = 2.0\n")
    f.write("for i, path in enumerate(fbx_files):\n")
    f.write("    bpy.ops.import_scene.fbx(filepath=path)\n")
    f.write("    obj = bpy.context.selected_objects[0]\n")
    f.write("    obj.location.x += i * spacing\n")

# Launch Blender with the generated script
subprocess.Popen([
    r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
    "--python", BLENDER_SCRIPT
])
