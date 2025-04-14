import os
import pickle
import subprocess
import argparse
from FbxReadWriter import FbxReadWrite
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--pkl", type=str, help="Path to SMPL .pkl file")
args = parser.parse_args()

PKL_PATH = args.pkl
FBX_TEMPLATE = r"C:\Users\kemij\Programming\SMPL-to-FBX-main\SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx"
OUTPUT_FOLDER = r"C:\Users\kemij\Programming\MAINPROJECT\motion_stitcher\output\choreography"
BLENDER_SCRIPT = os.path.join(OUTPUT_FOLDER, "import_group_fbx.py")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)
    print("Loaded choreography:", data.keys())

poses = data.get("smpl_poses")
trans = data.get("smpl_trans")
group_mode = isinstance(poses, np.ndarray) and poses.ndim == 3 and poses.shape[0] >= 2

fbx_files = []

if group_mode:
    print("Group motion detected. Exporting FBX files for each dancer.")
    for i in range(poses.shape[0]):
        smpl_params = {
            "smpl_poses": poses[i],
            "smpl_trans": trans[i] if isinstance(trans, np.ndarray) and trans.ndim == 3 else np.zeros((poses.shape[1], 3))
        }
        fbx_name = f"dancer_{i+1}.fbx"
        writer = FbxReadWrite(FBX_TEMPLATE)
        writer.addAnimation(pkl_filename=f"dancer_{i+1}", smpl_params=smpl_params)
        writer.writeFbx(OUTPUT_FOLDER, fbx_name)
        writer.destroy()
        fbx_files.append(os.path.join(OUTPUT_FOLDER, fbx_name))
else:
    print("Single dancer motion detected. Exporting one FBX file.")
    smpl_params = {
        "smpl_poses": poses,
        "smpl_trans": trans
    }
    fbx_name = "solo_dancer.fbx"
    writer = FbxReadWrite(FBX_TEMPLATE)
    writer.addAnimation(pkl_filename="solo_dancer", smpl_params=smpl_params)
    writer.writeFbx(OUTPUT_FOLDER, fbx_name)
    writer.destroy()
    fbx_files.append(os.path.join(OUTPUT_FOLDER, fbx_name))

with open(BLENDER_SCRIPT, "w") as f:
    f.write("import bpy\n\n")
    f.write("fbx_files = [\n")
    for path in fbx_files:
        f.write(f"    r'{path}',\n")
    f.write("]\n\n")
    f.write("spacing = 2.0\n")
    f.write("for i, path in enumerate(fbx_files):\n")
    f.write("    bpy.ops.import_scene.fbx(filepath=path)\n")
    f.write("    obj = bpy.context.selected_objects[0]\n")
    f.write("    obj.location.x += i * spacing\n")

subprocess.Popen([r"C:\\Program Files\\Blender Foundation\\Blender 4.3\\blender.exe", "--python", BLENDER_SCRIPT])
