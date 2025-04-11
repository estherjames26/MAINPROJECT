"""Check the length of choreography files in the pkl format."""
import os
import pickle
import glob
import argparse
import numpy as np

def check_pkl_file(file_path):
    """Check the length of a single choreography file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract motion data
        if isinstance(data, dict) and 'motion' in data:
            motion_data = data['motion']
        else:
            motion_data = data
        
        # Check dimensions
        if isinstance(motion_data, np.ndarray):
            shape = motion_data.shape
            
            # Determine number of dancers and frames
            if len(shape) == 4:  # [dancers, frames, joints, 3]
                num_dancers = shape[0]
                num_frames = shape[1]
            elif len(shape) == 3:
                if shape[2] == 3:  # [frames, joints, 3]
                    num_dancers = 1
                    num_frames = shape[0]
                else:  # [dancers, frames, joints*3]
                    num_dancers = shape[0]
                    num_frames = shape[1]
            elif len(shape) == 2:  # [frames, joints*3]
                num_dancers = 1
                num_frames = shape[0]
            else:
                return {
                    'file': os.path.basename(file_path),
                    'success': False,
                    'error': f"Unexpected shape: {shape}"
                }
            
            return {
                'file': os.path.basename(file_path),
                'path': file_path,
                'shape': shape,
                'dancers': num_dancers,
                'frames': num_frames,
                'duration': num_frames / 30,  #30 fps
            }
        else:
            return {
                'file': os.path.basename(file_path),
                'success': False,
                'error': f"Not a numpy array: {type(motion_data)}"
            }
    except Exception as e:
        return {
            'file': os.path.basename(file_path),
            'success': False,
            'error': str(e)
        }

def check_directory(directory_path, pattern="*.pkl"):
    """Check all choreography files in a directory."""
    # Find all pkl files
    pkl_files = glob.glob(os.path.join(directory_path, pattern))
    
    if not pkl_files:
        print(f"No files found matching {pattern} in {directory_path}")
        return []
    
    results = []
    
    # Process each file
    for file_path in pkl_files:
        result = check_pkl_file(file_path)
        results.append(result)
    
    return results

def print_results(results):
    """Print results in a readable format."""
    # Count successes and failures
    successes = [r for r in results if r['success']]
    failures = [r for r in results if not r['success']]
    
    print(f"\nChecked {len(results)} files:")
    print(f"  - {len(successes)} successful")
    print(f"  - {len(failures)} failed")
    
    # Print successful files
    if successes:
        print("\nSuccessful files:")
        for r in successes:
            print(f"  - {r['file']}: {r['dancers']} dancer(s), {r['frames']} frames, {r['duration']:.2f}s")
    
    # Print failed files
    if failures:
        print("\nFailed files:")
        for r in failures:
            print(f"  - {r['file']}: {r['error']}")
    
    # Print statistics if there are successful files
    if successes:
        durations = [r['duration'] for r in successes]
        print("\nStatistics:")
        print(f"  - Min duration: {min(durations):.2f}s")
        print(f"  - Max duration: {max(durations):.2f}s")
        print(f"  - Avg duration: {sum(durations)/len(durations):.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the length of choreography files")
    parser.add_argument("--dir", "-d", required=True, help="Directory containing choreography files")
    parser.add_argument("--pattern", "-p", default="*.pkl", help="File pattern to match")
    
    args = parser.parse_args()
    
    results = check_directory(args.dir, args.pattern)
    print_results(results)
