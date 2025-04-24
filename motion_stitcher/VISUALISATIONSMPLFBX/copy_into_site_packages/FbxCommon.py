"""
Common utilities for FBX SDK operations.
This module provides functions for initializing FBX SDK and performing common operations.
"""

import fbx
import sys

def InitializeSdkObjects():
    """Initialize FBX SDK objects."""
    # Create the FBX SDK manager
    sdk_manager = fbx.FbxManager.Create()
    if not sdk_manager:
        sys.exit(0)
        
    # Create an IOSettings object
    ios = fbx.FbxIOSettings.Create(sdk_manager, fbx.IOSROOT)
    sdk_manager.SetIOSettings(ios)
    
    # Create a new scene
    scene = fbx.FbxScene.Create(sdk_manager, "My Scene")
    if not scene:
        sys.exit(0)
        
    return sdk_manager, scene

def SaveScene(sdk_manager, scene, filename):
    """Save scene to a file."""
    # Create an exporter
    exporter = fbx.FbxExporter.Create(sdk_manager, "")
    
    # Initialize the exporter
    result = exporter.Initialize(filename, -1, sdk_manager.GetIOSettings())
    if not result:
        print("Failed to initialize exporter")
        return False
    
    # Export the scene
    result = exporter.Export(scene)
    
    # Destroy the exporter
    exporter.Destroy()
    
    return result