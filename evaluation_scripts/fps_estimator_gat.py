import cv2
import os
import sys
import time
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append("../")

from SiamGAT.pysot.tracker.siamgat_tracker import SiamGATTracker
from SiamGAT.pysot.models.model_builder_gat import ModelBuilder
from SiamGAT.pysot.utils.model_load import load_pretrain
from SiamGAT.pysot.core.config import cfg



# Modifying the create_tracker function to only handle SiamGAT tracker
def create_tracker(tracker_type):
    """Instantiate and return a tracker object based on tracker type."""
    
    if tracker_type == 'SiamGAT':
        # Assuming you have a model snapshot to load
        model = ModelBuilder()
        
        snapshot_path = '.pth'  # Updating this snapshot path
        
        model = load_pretrain(model, snapshot_path).cuda().eval()

        return SiamGATTracker(model)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")

def read_first_bbox(gt_path):
    """Read the first bounding box from a groundtruth file."""
    
    try:
        
        with open(gt_path, 'r') as file:
            first_line = file.readline().strip()
        
            if not first_line:
                raise ValueError("Groundtruth file is empty")
            
            bbox = first_line.split(',')
        
            if len(bbox) != 4:
                raise ValueError(f"Invalid bbox format: {first_line}")
            
            x, y, w, h = map(int, bbox)
        
            return (x, y, w, h)
    
    except Exception as e:
        print(f"Error reading groundtruth file {gt_path}: {e}")
    
        return None

def process_sequence(sequence_path, tracker_type):
    """Process a single sequence of frames and calculate FPS metrics."""
    
    try:
        # Checking if img directory exists
        img_path = os.path.join(sequence_path, 'img')
        
        if not os.path.exists(img_path):
            print(f"Image directory not found: {img_path}")
        
            return None

        # Sorting frames in correct order
        frames = sorted([f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png'))])
        
        if not frames:
            print(f"No frames found in {img_path}")
        
            return None

        # Reading groundtruth bounding box
        gt_path = os.path.join(sequence_path, 'groundtruth.txt')
        init_bb = read_first_bbox(gt_path)
        
        if init_bb is None:
            return None

        # Initializing tracker
        tracker = create_tracker(tracker_type)
        first_frame = cv2.imread(os.path.join(img_path, frames[0]))
        success = tracker.init(first_frame, init_bb)
        
        if not success:
            print(f"Failed to initialize tracker for {sequence_path}")
        
            return None

        # Processing frames and measure time
        frame_times = []
        successful_tracks = 0
        
        for frame_name in tqdm(frames[1:], desc=f"Processing {os.path.basename(sequence_path)}"):
            frame = cv2.imread(os.path.join(img_path, frame_name))
        
            if frame is None:
                continue
            
            # Measuring tracking time
            start_time = time.time()
            success, _ = tracker.update(frame)
            end_time = time.time()
            
            if success:
                successful_tracks += 1
                frame_times.append(end_time - start_time)
        
        # Calculating FPS and success rate
        if not frame_times:
            print(f"No successful tracks for {sequence_path}")
            
            return {
                'sequence_name': os.path.basename(sequence_path),
                'avg_fps': 0,
                'success_rate': 0,
                'total_frames': len(frames)
            }
        
        avg_fps = 1.0 / np.mean(frame_times)
        tracking_success_rate = (successful_tracks / (len(frames) - 1)) * 100

        return {
            'sequence_name': os.path.basename(sequence_path),
            'avg_fps': round(avg_fps, 2),
            'success_rate': round(tracking_success_rate, 2),
            'total_frames': len(frames)
        }

    except Exception as e:
        print(f"Error processing sequence {sequence_path}: {e}")
        
        return None

def main():
    # Configurations
    root_dir = Path('../Lasot_mini')  # Directory containing sequence folders
    tracker_type = 'SiamGAT'  # Only test SiamGAT tracker
    output_dir = 'tracker_results'
    
    # Creating output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nProcessing {tracker_type} tracker...")
    
    results = []
    
    # Processing each sequence folder
    sequence_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    
    for sequence_folder in sequence_folders:
        sequence_path = os.path.join(root_dir, sequence_folder)
        
        print(f"\nProcessing sequence: {sequence_folder}")
        
        metrics = process_sequence(sequence_path, tracker_type)
        
        if metrics:
            results.append(metrics)
    
    # Saving results to CSV
    if results:
        csv_path = os.path.join(output_dir, f'{tracker_type}_results.csv')
        fieldnames = ['sequence_name', 'avg_fps', 'success_rate', 'total_frames']
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

if __name__ == "__main__":
    main()
