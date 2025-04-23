import os, glob, pickle, numpy as np
from database import MotionDatabase
import config


def load_aist_data():
    # Load motion data from AIST++ dataset
    print("Processing AIST++ dataset for solo dancers...")
    ignore_file = os.path.join(config.AIST_DIR, 'ignore_list.txt')
    ignore_ids = set()
    if os.path.exists(ignore_file):
        with open(ignore_file, 'r') as f:
            ignore_ids = {line.strip() for line in f if line.strip()}
    print(f"Loaded {len(ignore_ids)} ignore IDs from {ignore_file!r}")
    
    motion_dir = config.AIST_MOTION_DIR
    if not os.path.exists(motion_dir):
        print(f"AIST++ motion directory not found: {motion_dir}")
        return []
    
    motion_files = glob.glob(os.path.join(motion_dir, '*.pkl'))
    clips = []

    for i, motion_file in enumerate(motion_files):
        try:
            with open(motion_file, 'rb') as f:
                motion_data = pickle.load(f)

            filename = os.path.basename(motion_file).split('.')[0]
            if filename in ignore_ids:
                continue

            style = filename.split('_')[0]
            metadata = {'source': 'aist++', 'filename': filename, 'style': style}
            clips.append((filename, motion_data, metadata))

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} AIST++ files")
        except Exception as e:
            print(f"Error processing {motion_file}: {e}")
    
    print(f"Added {len(clips)} AIST++ solo clips to database")
    return clips

def load_aioz_data():
    # Load motion data from AIOZ dataset
    print("Processing AIOZ dataset for group dancers...")
    motion_dir = os.path.join(config.AIOZ_DIR, 'motions_smpl')

    if not os.path.exists(motion_dir):
        print(f"AIOZ motion directory not found: {motion_dir}")
        return []
    
    motion_files = glob.glob(os.path.join(motion_dir, '*.pkl'))
    clips = []

    for i, motion_file in enumerate(motion_files):
        try:
            with open(motion_file, 'rb') as f:
                motion_data = pickle.load(f)

            filename = os.path.basename(motion_file).split('.')[0]
            metadata = {'source': 'aioz', 'filename': filename}
            clips.append((filename, motion_data, metadata))

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} AIOZ files")
        except Exception as e:
            print(f"Error processing {motion_file}: {e}")
    
    print(f"Added {len(clips)} AIOZ group clips to database")
    return clips

def build_database():
    print("Building motion database...")
    db_path = os.path.join(config.DATABASE_DIR, 'motion_database.pkl')
    db = MotionDatabase(db_path)

    aist_clips = load_aist_data()
    aioz_clips = load_aioz_data()

    # Add clips to database using built-in add_clip method
    for clip_id, motion_data, metadata in aist_clips + aioz_clips:
        db.add_clip(clip_id, motion_data, metadata)

    # Save the general database
    db.save()
    print(f"Database built with {len(aist_clips)} solo clips and {len(aioz_clips)} group clips")
    print(f"Saved to: {db_path}")

    # Building dancer-specific databases
    print("Building separate databases for different dancer counts...")
    build_dancer_specific_databases(db, aist_clips, aioz_clips)

def build_dancer_specific_databases(db, aist_clips, aioz_clips):
    dancer_counts = [1, 2, 3]

    for dancer_count in dancer_counts:
        print(f"\nBuilding database for {dancer_count} dancer(s)...")
        dancer_db_path = os.path.join(config.DATABASE_DIR, f'{dancer_count}_dancer_db.pkl')
        dancer_db = MotionDatabase(dancer_db_path)

        # Filter and add clips based on dancer count
        if dancer_count == 1:
            for clip_id, motion_data, metadata in aist_clips:
                dancer_db.add_clip(clip_id, motion_data, metadata)
        else:
            for clip_id, motion_data, metadata in aioz_clips:
                try:
                    motion = motion_data.get('smpl_poses') if isinstance(motion_data, dict) else motion_data
                    if isinstance(motion, np.ndarray) and len(motion.shape) == 3 and motion.shape[0] == dancer_count:
                        dancer_db.add_clip(clip_id, motion_data, metadata)
                except Exception as e:
                    continue

        # Save dancer-specific databases
        dancer_db.save()
        print(f"Database for {dancer_count} dancer(s) built and saved to: {dancer_db_path}")

if __name__ == "__main__":
    build_database()
