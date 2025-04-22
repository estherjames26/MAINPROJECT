#Script to build the motion database from datasets AIST++ and AIOZ
import os
import numpy as np
import glob
import pickle
from database import MotionDatabase
import config

def load_aist_data():
    # Load motion data from AIST++ dataset
    print("Processing AIST++ dataset for solo dancers...")
    
    motion_dir = config.AIST_MOTION_DIR
    
    # Check if directory exists
    if not os.path.exists(motion_dir):
        print(f"AIST++ motion directory not found: {motion_dir}")
        return []
    
    # Get all motion files
    motion_files = glob.glob(os.path.join(motion_dir, '*.pkl'))
    
    clips = []
    
    # Processes each motion file
    for i, motion_file in enumerate(motion_files):
        try:
            # Load motion data
            with open(motion_file, 'rb') as f:
                motion_data = pickle.load(f)
            
            # Extract filename without extension
            filename = os.path.basename(motion_file).split('.')[0]
            
            # Get motion style from filename
            style = filename.split('_')[0]
            
            # Create metadata
            metadata = {
                'source': 'aist++',
                'filename': filename,
                'style': style
            }
            
            # Add to clips
            clips.append((filename, motion_data, metadata))
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} AIST++ files")
        
        except Exception as e:
            print(f"Error processing {motion_file}: {e}")
    
    print(f"Added {len(clips)} AIST++ solo clips to database")
    return clips

def load_aioz_data():
    #Load motion data from AIOZ dataset
    print("Processing AIOZ dataset for group dancers...")
    
    motion_dir = os.path.join(config.AIOZ_DIR, 'motions_smpl')
    
    # Check if directory exists
    if not os.path.exists(motion_dir):
        print(f"AIOZ motion directory not found: {motion_dir}")
        return []
    
    # Get all motion files
    motion_files = glob.glob(os.path.join(motion_dir, '*.pkl'))
    
    clips = []
    
    # Process each motion file
    for i, motion_file in enumerate(motion_files):
        try:
            # Load motion data
            with open(motion_file, 'rb') as f:
                motion_data = pickle.load(f)
            
            # Extract filename without extension
            filename = os.path.basename(motion_file).split('.')[0]
            
            # Create metadata
            metadata = {
                'source': 'aioz',
                'filename': filename
            }
            
            # Add to clips
            clips.append((filename, motion_data, metadata))
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} AIOZ files")
        
        except Exception as e:
            print(f"Error processing {motion_file}: {e}")
    
    print(f"Added {len(clips)} AIOZ group clips to database")
    return clips

def build_database():
    #Build the motion database for AIST++ and AIOZ datasets
    print("Building motion database...")
    
    # Create the database
    db_path = os.path.join(config.DATABASE_DIR, 'motion_database.pkl')
    db = MotionDatabase(db_path)
    
    # Load data from datasets
    aist_clips = load_aist_data()
    aioz_clips = load_aioz_data()
    
    # Add clips to database
    for clip_id, motion_data, metadata in aist_clips + aioz_clips:
        db.add_clip(clip_id, motion_data, metadata)
    
    # Save database
    db.save()
    
    print(f"Database built with {len(aist_clips)} solo clips and {len(aioz_clips)} group clips")
    print(f"Saved to: {db_path}")
    

    print("Building separate databases for different dancer counts...")
    build_dancer_specific_databases(aist_clips, aioz_clips)

def build_dancer_specific_databases(aist_clips, aioz_clips):
    # Builds separate databases for different dancer counts
    # Define dancer counts to process
    dancer_counts = [1, 2, 3]
    
    for dancer_count in dancer_counts:
        print(f"\nBuilding database for {dancer_count} dancer(s)...")
        
        # Create the database
        db_path = os.path.join(config.DATABASE_DIR, f'{dancer_count}_dancer_db.pkl')
        db = MotionDatabase(db_path)
        
        # Add clips based on dancer count
        if dancer_count == 1:
            # Solo dancers from AIST++
            print("Processing AIST++ dataset for solo dancers...")
            for clip_id, motion_data, metadata in aist_clips:
                db.add_clip(clip_id, motion_data, metadata)
        else:
            # Group dancers from AIOZ
            print("Processing AIOZ dataset for group dancers...")
            for clip_id, motion_data, metadata in aioz_clips:
                # Extract the actual motion data from the dictionary if needed
                try:
                    motion = None
                    if isinstance(motion_data, dict):
                        # Dictionary format (AIST++ style)
                        if 'smpl_poses' in motion_data:
                            motion = motion_data['smpl_poses']
                        elif 'poses' in motion_data:
                            motion = motion_data['poses']
                        elif 'motion' in motion_data:
                            motion = motion_data['motion']
                        else:
                            # Unknown format, skip this clip
                            continue
                    else:
                        # Direct numpy array
                        motion = motion_data
                    
                    # Check if dancer count matches
                    if isinstance(motion, np.ndarray):
                        if len(motion.shape) == 3 and motion.shape[0] == dancer_count:
                            db.add_clip(clip_id, motion_data, metadata)
                    # If not a numpy array, skip
                except Exception as e:
                    # Skip any clips that cause errors
                    continue
        
        # Save database
        db.save()
        
        # Get count of clips
        info = db.get_clips_info()
        print(f"Database for {dancer_count} dancer(s) built with {len(info)} clips")
        print(f"Saved to: {db_path}")

if __name__ == "__main__":
    build_database()
