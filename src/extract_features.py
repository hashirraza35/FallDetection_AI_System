import os
import glob
import re
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
UR_DATASET_DIR = Path(
    os.getenv("UR_DATASET_DIR", PROJECT_ROOT / "data" / "ur_dataset")
)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,  # Set to True for static frame image sequences
    model_complexity=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

SEQUENCE_LENGTH = 30  # 30-frame sliding window

def extract_landmarks(frame):
    """Processes a BGR image/frame and extracts 33x3 (99) pose coordinates."""
    if frame is None:
        return None
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    if not results.pose_landmarks:
        return None

    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

def natural_sort_key(s):
    """Sorts frame filenames numerically (e.g., frame_1, frame_2 ... frame_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def process_sequence_folder(folder_path, label):
    """Recursively finds all image frames inside a sequence folder."""
    X_seq = []
    
    # Recursively match image files inside nested directories
    image_files = []
    for ext in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
        image_files.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))

    image_files.sort(key=natural_sort_key)

    if not image_files:
        return []

    frame_buffer = []

    for img_path in image_files:
        frame = cv2.imread(img_path)
        coords = extract_landmarks(frame)
        if coords is not None:
            frame_buffer.append(coords)
            if len(frame_buffer) == SEQUENCE_LENGTH:
                X_seq.append((np.array(frame_buffer).flatten(), label))
                frame_buffer.pop(0)

    return X_seq

def process_video_file(video_path, label):
    """Reads frames from an AVI/MP4 video file."""
    X_seq = []
    cap = cv2.VideoCapture(video_path)
    frame_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        coords = extract_landmarks(frame)
        if coords is not None:
            frame_buffer.append(coords)
            if len(frame_buffer) == SEQUENCE_LENGTH:
                X_seq.append((np.array(frame_buffer).flatten(), label))
                frame_buffer.pop(0)

    cap.release()
    return X_seq

def process_dataset():
    X_data = []
    y_data = []

    adl_dir = UR_DATASET_DIR / "adl_files"
    fall_dir = UR_DATASET_DIR / "fall_files"

    print("==================================================")
    print("1. PROCESSING ADL FILES (LABEL 0)")
    print("==================================================")
    if os.path.exists(adl_dir):
        adl_folders = [os.path.join(adl_dir, d) for d in os.listdir(adl_dir) if os.path.isdir(os.path.join(adl_dir, d))]
        print(f"Found {len(adl_folders)} top-level ADL folders.")

        for idx, folder_path in enumerate(adl_folders):
            folder_name = os.path.basename(folder_path)
            results = process_sequence_folder(folder_path, label=0)
            for feat, lbl in results:
                X_data.append(feat)
                y_data.append(lbl)
            print(f"[{idx+1}/{len(adl_folders)}] Processed ADL Folder: {folder_name} -> {len(results)} sequence samples")

    print("\n==================================================")
    print("2. PROCESSING FALL FILES (LABEL 1)")
    print("==================================================")
    if os.path.exists(fall_dir):
        fall_folders = [os.path.join(fall_dir, d) for d in os.listdir(fall_dir) if os.path.isdir(os.path.join(fall_dir, d))]
        fall_videos = glob.glob(os.path.join(fall_dir, "**", "*.avi"), recursive=True) + glob.glob(os.path.join(fall_dir, "**", "*.mp4"), recursive=True)

        if fall_folders:
            print(f"Found {len(fall_folders)} Fall sequence folders.")
            for idx, folder_path in enumerate(fall_folders):
                folder_name = os.path.basename(folder_path)
                results = process_sequence_folder(folder_path, label=1)
                for feat, lbl in results:
                    X_data.append(feat)
                    y_data.append(lbl)
                print(f"[{idx+1}/{len(fall_folders)}] Processed Fall Folder: {folder_name} -> {len(results)} sequence samples")

        if fall_videos:
            print(f"Found {len(fall_videos)} Fall video files.")
            for idx, video_path in enumerate(fall_videos):
                video_name = os.path.basename(video_path)
                results = process_video_file(video_path, label=1)
                for feat, lbl in results:
                    X_data.append(feat)
                    y_data.append(lbl)
                print(f"[{idx+1}/{len(fall_videos)}] Processed Fall Video: {video_name} -> {len(results)} sequence samples")

    if not X_data:
        print("\nERROR: No features extracted.")
        return None

    feature_cols = [f"f_{i}" for i in range(len(X_data[0]))]
    df = pd.DataFrame(X_data, columns=feature_cols)
    df["label"] = y_data
    return df

if __name__ == "__main__":
    df_features = process_dataset()
    if df_features is not None:
        output_csv = PROJECT_ROOT / "data" / "urfd_extracted_features.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(output_csv, index=False)
        print(f"\nSUCCESS! Saved dataset '{output_csv}' with shape {df_features.shape}.")