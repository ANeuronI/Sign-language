import os
import numpy as np
import cv2
import mediapipe as mp
import time
import pandas as pd
from config import *

def extract_frames(video_path, max_frames=MAX_FRAMES):
    if DEBUG:
        print(f"Extracting frames from: {video_path}")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return np.zeros((max_frames, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            return np.zeros((max_frames, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frames.append(frame)
            frame_count += 1

            if frame_count >= max_frames:
                break

        cap.release()

        if DEBUG:
            print(f"Extracted {len(frames)} frames from {video_path}")

        if len(frames) < max_frames:
            padding = [np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) for _ in range(max_frames - len(frames))]
            frames.extend(padding)

        return np.array(frames)
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return np.zeros((max_frames, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

def extract_pose_landmarks(frames):
    landmarks_sequence = []

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=False
    ) as holistic:

        for frame in frames:
            if frame.sum() == 0:
                landmarks = np.zeros((75, 3))
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)

                pose_landmarks = np.zeros((33, 3))
                if results.pose_landmarks:
                    for i, landmark in enumerate(results.pose_landmarks.landmark):
                        pose_landmarks[i] = [landmark.x, landmark.y, landmark.z]

                left_hand = np.zeros((21, 3))
                if results.left_hand_landmarks:
                    for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                        left_hand[i] = [landmark.x, landmark.y, landmark.z]

                right_hand = np.zeros((21, 3))
                if results.right_hand_landmarks:
                    for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                        right_hand[i] = [landmark.x, landmark.y, landmark.z]

                landmarks = np.concatenate([pose_landmarks, left_hand, right_hand])

            landmarks_sequence.append(landmarks)

    return np.array(landmarks_sequence)

def find_text_for_video(video_file, df, text_col):
    video_base = os.path.splitext(video_file)[0]

    if 'uid' not in df.columns:
        print("Error: 'uid' column not found in CSV.")
        return f"[No text for {video_base}]"

    uid_matches = df[df['uid'].astype(str) == video_base]
    if not uid_matches.empty:
        return str(uid_matches.iloc[0][text_col])

    for _, row in df.iterrows():
        if row['uid'] in video_base:
            return row[text_col]

    print(f"Warning: No text found for video {video_file}. Using filename as placeholder.")
    return f"[No text for {video_base}]"

def collect_dataset(video_dir=VIDEOS_DIR, csv_file=CSV_FILE, save_preprocessed=True):
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return [], []

    if not os.path.exists(video_dir):
        print(f"Error: Video directory not found: {video_dir}")
        return [], []

    try:
        print(f"Reading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"CSV loaded. Entries: {len(df)}")

        text_col = None
        for candidate in ['text', 'sentence', 'caption', 'transcription']:
            if candidate in df.columns:
                text_col = candidate
                break

        if not text_col:
            print("Error: No suitable text column found.")
            return [], []

        print(f"Using text column: {text_col}")

        video_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]
        if not video_files:
            print("Error: No videos found.")
            return [], []

        X, y = [], []
        max_videos = 10 if DEBUG else len(video_files)

        for idx, video_file in enumerate(video_files[:max_videos]):
            try:
                sentence = find_text_for_video(video_file, df, text_col)
                video_path = os.path.join(video_dir, video_file)
                print(f"Processing video {idx+1}/{max_videos}: {video_file}")

                t0 = time.time()
                frames = extract_frames(video_path)
                print(f"  Frame extraction: {time.time() - t0:.2f}s")

                t1 = time.time()
                landmarks = extract_pose_landmarks(frames)
                print(f"  Landmark extraction: {time.time() - t1:.2f}s")

                X.append(landmarks)
                y.append(sentence)
                print(f"  Done: {video_file}")
                print(f"  Text: {sentence[:50]}{'...' if len(sentence) > 50 else ''}")

            except Exception as e:
                print(f"Error processing video {video_file}: {e}")

        if not X:
            print("Warning: No videos processed.")
            return [], []

        X = np.array(X)
        y = np.array(y)

        if save_preprocessed:
            print(f"Saving data to: {PREPROCESSED_DATA_FILE}")
            os.makedirs(os.path.dirname(PREPROCESSED_DATA_FILE), exist_ok=True)
            np.savez_compressed(PREPROCESSED_DATA_FILE, landmarks=X, sentences=y)
            print("Data saved.")

        return X, y

    except Exception as e:
        print(f"Error in collect_dataset: {e}")
        import traceback
        traceback.print_exc()
        return [], []

if __name__ == "__main__":
    print("Running data collection...")
    X, y = collect_dataset()
    print(f"Collected {len(X)} samples.")
    if len(y) > 0:
        print("Example:", y[0])
