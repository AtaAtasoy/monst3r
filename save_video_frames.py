import os
import cv2
import sys


def save_frames_from_video(video_path):
    # Remove extension and use as output directory
    base = os.path.splitext(video_path)[0]
    output_dir = base
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    print(f"Saved {frame_idx} frames to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python save_video_frames.py <video_file.mp4>")
        sys.exit(1)
    video_path = sys.argv[1]
    save_frames_from_video(video_path)
