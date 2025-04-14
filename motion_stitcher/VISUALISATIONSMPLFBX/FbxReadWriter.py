import sys
from typing import Dict
from SmplObject import SmplObjects
import os
from scipy.spatial.transform import Rotation as R
import numpy as np

try:
    from FbxCommon import *
    import fbx
except ImportError:
    print("Error: module FbxCommon failed to import.\n")
    print("Copy the files located in the compatible sub-folder lib/python<version> into your python interpreter site-packages folder.")
    import platform
    if platform.system() == 'Windows' or platform.system() == 'Microsoft':
        print('For example: copy ..\\..\\lib\\Python27_x64\\* C:\\Python27\\Lib\\site-packages')
    elif platform.system() == 'Linux':
        print('For example: cp ../../lib/Python27_x64/* /usr/local/lib/python2.7/site-packages')
    elif platform.system() == 'Darwin':
        print('For example: cp ../../lib/Python27_x64/* /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')

class FbxReadWrite(object):
    def __init__(self, fbx_source_path):
        # Prepare the FBX SDK.
        lSdkManager, lScene = InitializeSdkObjects()
        self.lSdkManager = lSdkManager
        self.lScene = lScene

        # Load the scene.
        print("\nLoading File: {}".format(fbx_source_path))
        lResult = LoadScene(self.lSdkManager, self.lScene, fbx_source_path)
        if not lResult:
            raise Exception("An error occurred while loading the scene :(")

    def _write_curve(self, lCurve: fbx.FbxAnimCurve, data: np.ndarray):
        """
        Write keyframe data from a numpy array (data: (N,)) into an FBX animation curve.
        """
        lTime = fbx.FbxTime()
        lTime.SetGlobalTimeMode(fbx.FbxTime.eFrames60)  # Set to fps=60
        data = np.squeeze(data)

        lCurve.KeyModifyBegin()
        for i in range(data.shape[0]):
            lTime.SetFrame(i, fbx.FbxTime.eFrames60)
            keyIndex = lCurve.KeyAdd(lTime)[0]
            lCurve.KeySetValue(keyIndex, data[i])
            lCurve.KeySetInterpolation(keyIndex, fbx.FbxAnimCurveDef.eInterpolationCubic)
        lCurve.KeyModifyEnd()

    def addAnimation(self, pkl_filename: str, smpl_params: Dict, verbose: bool = False):
        lScene = self.lScene
        print("Adding animation for {}".format(pkl_filename))

        # Set fps to 60
        lGlobalSettings = lScene.GetGlobalSettings()
        if verbose:
            print("Before time mode: {}".format(lGlobalSettings.GetTimeMode()))
        lGlobalSettings.SetTimeMode(fbx.FbxTime.eFrames60)
        if verbose:
            print("After time mode: {}".format(lScene.GetGlobalSettings().GetTimeMode()))

        self.destroyAllAnimation()

        lAnimStackName = pkl_filename
        lAnimStack = fbx.FbxAnimStack.Create(lScene, lAnimStackName)
        lAnimLayer = fbx.FbxAnimLayer.Create(lScene, "Base Layer")
        lAnimStack.AddMember(lAnimLayer)
        lRootNode = lScene.GetRootNode()

        # Build joint names with the prefix
        joint_names = ["m_avg_" + name for name in SmplObjects.joints]

        # 1. Write smpl_poses (rotation animation)
        smpl_poses = smpl_params["smpl_poses"]
        for idx, joint_name in enumerate(joint_names):
            # Use recursive search to find the node.
            node = lRootNode.FindChild(joint_name, True)
            if node is None:
                print("Node {} not found, skipping rotation data.".format(joint_name))
                continue

            # Extract the rotation vectors for this joint.
            rotvec = smpl_poses[:, idx * 3 : idx * 3 + 3]
            euler_angles = []
            for f in range(smpl_poses.shape[0]):
                frame_rotvec = smpl_poses[f]               # (72,)
                joint_rotvecs = frame_rotvec.reshape(24, 3)
                single_joint_rotvec = joint_rotvecs[idx]   # (3,)
                r = R.from_rotvec(single_joint_rotvec)
                euler = r.as_euler('xyz', degrees=True)    # (3,)
                euler_angles.append(euler)

            euler_angles = np.vstack(euler_angles)

            # Write X, Y, Z rotation curves.
            for channel, col in zip(["X", "Y", "Z"], [0, 1, 2]):
                lCurve = node.LclRotation.GetCurve(lAnimLayer, channel, True)
                if lCurve:
                    self._write_curve(lCurve, euler_angles[:, col])
                else:
                    print("Failed to write rotation for {} channel {}.".format(joint_name, channel))

        # 2. Write smpl_trans (translation animation) for the root node.
        smpl_trans = smpl_params["smpl_trans"]
        # First try to find "m_avg_root". If not found, use the scene root.
        root_node = lRootNode.FindChild("m_avg_root", True)
        if root_node is None:
            print("Node 'm_avg_root' not found; using the scene root instead.")
            root_node = lRootNode

        # Write translation curves for X, Y, Z.
        # Adjust the column order if necessary. Here we assume smpl_trans is (num_frames, 3) in order [X, Y, Z].
        for channel, idx in zip(["X", "Y", "Z"], [0, 1, 2]):
            lCurve = root_node.LclTranslation.GetCurve(lAnimLayer, channel, True)
            if lCurve:
                self._write_curve(lCurve, smpl_trans[:, idx])
            else:
                print("Failed to write translation for node {} channel {}.".format(root_node.GetName(), channel))

    def writeFbx(self, write_base: str, filename: str):
        if not os.path.isdir(write_base):
            os.makedirs(write_base, exist_ok=True)
        # Append the .fbx extension.
        filename = os.path.splitext(filename)[0]  # Remove any .pkl or .fbx
        write_path = os.path.join(write_base, filename + ".fbx")
        print("Writing to {}".format(write_path))
        result = SaveScene(self.lSdkManager, self.lScene, write_path)
        print("Write result:", write_path)
        if not result:
            raise Exception("Failed to write to {}".format(write_path))

    def destroy(self):
        self.lSdkManager.Destroy()

    def destroyAllAnimation(self):
        lScene = self.lScene
        animStackCount = lScene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
        for i in range(animStackCount):
            lAnimStack = lScene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), i)
            lAnimStack.Destroy()

