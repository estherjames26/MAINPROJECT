"""Main entry point for the motion stitcher application."""
import os
import argparse
import pickle

from config import Config
from database import MotionDatabase
from generate import generate_choreography
from build_database import build_separate_databases
from smpl_visualizer_fixed import create_group_dance_video
from visblender import automatic_blender_visualization

def check_database():
    """Check if databases exist and build them if needed."""
    # Load config
    config = Config()
    
    databases_exist = True
    
    for num_dancers in [1, 2, 3]:
        db_path = os.path.join(config.database_dir, f"{num_dancers}_dancer_db.pkl")
        if not os.path.exists(db_path):
            databases_exist = False
            break
    
    if not databases_exist:
        print("Databases not found. Building databases...")
        build_separate_databases()
        return True
    else:
        print("Databases found.")
        return True

def create_choreography(audio_path, num_dancers=1, style=None, duration=None, use_blender=False):
    """Create a choreography from audio."""
    print(f"Creating choreography for {num_dancers} dancer(s) using {audio_path}")
    
    # Generate choreography using the simplified function from generate.py
    output_path = generate_choreography(
        audio_path=audio_path,
        num_dancers=num_dancers,
        style=style,
        duration=duration,
        visualize_blender=use_blender
    )
    
    if output_path:
        print(f"Choreography created and saved to {output_path}")
        return output_path
    else:
        print("Failed to create choreography")
        return None

def visualize_choreography(choreography_path, output_video_path=None):
    """Visualize a choreography file."""
    print(f"Visualizing choreography from {choreography_path}")
    
    # Load config
    config = Config()
    
    # Load choreography
    try:
        with open(choreography_path, 'rb') as f:
            choreography_data = pickle.load(f)
        
        # Extract motion data and metadata
        motion_data = choreography_data.get('motion')
        metadata = choreography_data.get('metadata', {})
        
        # Create output path if not specified
        if output_video_path is None:
            output_name = os.path.splitext(os.path.basename(choreography_path))[0] + '.mp4'
            output_video_path = os.path.join(config.output_dir, 'videos', output_name)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        
        # Create visualization
        create_group_dance_video({
            'motion': motion_data,
            'output': output_video_path,
            'fps': 30,
            'max_frames': 300
        })
        
        print(f"Visualization saved to {output_video_path}")
        return output_video_path
    except Exception as e:
        print(f"Error visualizing choreography: {e}")
        return None

def visualize_choreography_blender(choreography_path):
    """Visualize a choreography file in Blender."""
    print(f"Preparing to visualize in Blender: {choreography_path}")
    
    # Load config
    config = Config()
    
    # Set output folder for BVH files
    output_folder = os.path.join(config.output_dir, 'bvh')
    os.makedirs(output_folder, exist_ok=True)
    
    # Define model path
    model_path = os.path.join(config.models_dir, 'SMPL_MALE.pkl')
    
    # Convert and visualize
    try:
        automatic_blender_visualization(
            pkl_path=choreography_path,
            output_folder=output_folder,
            model_path=model_path,
            run_blender_app=True
        )
        return True
    except Exception as e:
        print(f"Error visualizing in Blender: {e}")
        return False

def process_audio_to_choreography(audio_path, num_dancers=1, style=None, visualize=True, use_blender=False):
    """Process audio file to generate and optionally visualize choreography."""
    # Check if databases exist, build if needed
    check_database()
    
    # Create choreography (blender visualization is now handled directly in generate_choreography)
    choreo_path = create_choreography(
        audio_path=audio_path,
        num_dancers=num_dancers,
        style=style,
        use_blender=use_blender
    )
    
    results = {}
    
    # Store results and visualize if requested
    if choreo_path:
        results['choreography_path'] = choreo_path
        
        # Standard video visualization (separate from Blender)
        if visualize:
            video_path = visualize_choreography(choreo_path)
            results['video_path'] = video_path
    
    return results

# For CLI purposes only (inference and visualization)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motion stitcher for dance choreography")
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create choreography subcommand
    create_parser = subparsers.add_parser("create", help="Create choreography from audio")
    create_parser.add_argument("--audio", "-a", required=True, help="Path to audio file")
    create_parser.add_argument("--output", "-o", help="Path to save choreography")
    create_parser.add_argument("--dancers", "-d", type=int, default=1, help="Number of dancers (1-3)")
    create_parser.add_argument("--style", "-s", help="Dance style")
    create_parser.add_argument("--duration", type=int, help="Target duration in frames")
    create_parser.add_argument("--visualize", "-v", action="store_true", help="Visualize the choreography")
    create_parser.add_argument("--blender", "-b", action="store_true", help="Visualize in Blender")
    
    # Visualize choreography subcommand
    visualize_parser = subparsers.add_parser("visualize", help="Visualize existing choreography")
    visualize_parser.add_argument("--choreography", "-c", required=True, help="Path to choreography file")
    visualize_parser.add_argument("--output", "-o", help="Path to save visualization")
    visualize_parser.add_argument("--blender", "-b", action="store_true", help="Visualize in Blender")
    
    # Build database subcommand
    build_parser = subparsers.add_parser("build", help="Build motion database")
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == "create":
        # Process audio to choreography
        process_audio_to_choreography(
            audio_path=args.audio,
            num_dancers=args.dancers,
            style=args.style,
            visualize=args.visualize,
            use_blender=args.blender
        )
    
    elif args.command == "visualize":
        # Visualize existing choreography
        if args.blender:
            visualize_choreography_blender(args.choreography)
        else:
            visualize_choreography(
                choreography_path=args.choreography,
                output_video_path=args.output
            )
    
    elif args.command == "build":
        # Check and build databases
        check_database()
    
    else:
        parser.print_help()
