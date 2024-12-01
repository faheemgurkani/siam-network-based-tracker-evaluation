import cv2
import os
import sys
import time
import csv
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.append("../")

from TracKit.lib.models import Ocean, NetWrapper



# Initializing Ocean model and load weights
def initialize_ocean_model(weights_path):
    model = Ocean(align=False, online=True)  
    
    net_wrapper = NetWrapper(net_path="../model_weights/Ocean.pth")
    net_wrapper.load_network()  # Loading weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_wrapper.net = net_wrapper.net.to(device)
    net_wrapper.net.eval()  # Setting to evaluation mode
    
    return net_wrapper

def process_sequence(sequence_path, tracker_type, ocean_model):
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

        # Initialize FPS and tracking variables
        frame_times = []
        successful_tracks = 0
        
        for frame_name in tqdm(frames, desc=f"Processing {os.path.basename(sequence_path)}"):
            frame = cv2.imread(os.path.join(img_path, frame_name))
        
            if frame is None:
                continue

            # Perform inference using the Ocean model
            input_tensor = torch.from_numpy(frame).to(ocean_model.net.device).unsqueeze(0)
            with torch.no_grad():
                output = ocean_model.net(input_tensor)  # Inference step
                # Post-process the output to get bounding boxes (Assuming 'output' contains boxes)
                # Modify this depending on your Ocean model's output structure
                predicted_boxes = output  # Modify based on actual output format

            # Example tracking: you may need to adapt this based on your desired tracking
            if predicted_boxes:
                successful_tracks += 1
                # You can use predicted_boxes for tracking, e.g., assigning to `init_bb` in later frames

            # Measure FPS
            start_time = time.time()
            end_time = time.time()
            frame_times.append(end_time - start_time)
        
        # Calculate FPS and success rate
        if not frame_times:
            print(f"No successful tracks for {sequence_path}")
            return {'sequence_name': os.path.basename(sequence_path), 'avg_fps': 0, 'success_rate': 0, 'total_frames': len(frames)}

        avg_fps = 1.0 / np.mean(frame_times)
        tracking_success_rate = (successful_tracks / len(frames)) * 100

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
    output_dir = 'tracker_results'
    
    # Load Ocean model and weights
    ocean_model = initialize_ocean_model('/path/to/ocean_weights.pth')  # Replace with actual path

    # Creating output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nProcessing Ocean model for FPS estimation...")
    
    results = []
    
    # Processing each sequence folder
    sequence_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    
    for sequence_folder in sequence_folders:
        sequence_path = os.path.join(root_dir, sequence_folder)
        print(f"\nProcessing sequence: {sequence_folder}")
        
        metrics = process_sequence(sequence_path, 'Ocean', ocean_model)
        
        if metrics:
            results.append(metrics)
    
    # Saving results to CSV
    if results:
        csv_path = os.path.join(output_dir, 'ocean_model_results.csv')
        fieldnames = ['sequence_name', 'avg_fps', 'success_rate', 'total_frames']
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)



if __name__ == "__main__":
    main()
