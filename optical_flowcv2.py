import cv2
import numpy as np
import os
from tqdm import tqdm

def calculate_speed(prev_position, current_position, time_elapsed):
    return np.linalg.norm(np.array(current_position) - np.array(prev_position)) / time_elapsed

def merge_boxes(boxes, threshold=30):
    if not boxes:
        return []
    boxes.sort(key=lambda x: x[0])  # Sort boxes by the x-coordinate
    merged = [boxes[0]]
    for current in boxes[1:]:
        last = merged[-1]
        if current[0] <= last[0] + last[2] + threshold:
            merged[-1] = (min(last[0], current[0]), min(last[1], current[1]),
                          max(last[0] + last[2], current[0] + current[2]) - min(last[0], current[0]),
                          max(last[1] + last[3], current[1] + current[3]) - min(last[1], current[1]))
        else:
            merged.append(current)
    return merged

def draw_transparent_box(img, start_point, end_point, color, opacity):
    overlay = img.copy()
    cv2.rectangle(overlay, start_point, end_point, color, -1)
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
    return img

def custom_optical_flow(prev_gray, gray, feature_params, lk_params):
    p0 = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
        if p1 is not None and st.any():
            return p1[st == 1], p0[st == 1]
    return None, None

def process_video(video_path, output_path, live=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    ret, background = cap.read()
    if not ret:
        print("Failed to capture video")
        cap.release()
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4)))) if not live else None

    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)
    feature_params = {'maxCorners': 100, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7}
    lk_params = {'winSize': (15, 15), 'maxLevel': 2, 'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}
    prev_gray = background_gray

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            good_new, good_old = custom_optical_flow(prev_gray, gray, feature_params, lk_params)
            if good_new is not None:
                for new, old in zip(good_new, good_old):
                    cv2.line(frame, (int(old[0]), int(old[1])), (int(new[0]), int(new[1])), (0, 255, 0), 2)
                    cv2.circle(frame, (int(new[0]), int(new[1])), 5, (0, 255, 0), -1)

            if live:
                cv2.imshow('Live Motion Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                out.write(frame)

            prev_gray = gray
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    output_dir = 'processed_videos'
    os.makedirs(output_dir, exist_ok=True)

    mode = input("Enter 'video' to process a video file or 'webcam' for live feed: ").lower()
    if mode == 'video':
        video_path = input("Enter the full path to the video file: ")
        output_path = os.path.join(output_dir, os.path.basename(video_path))
        process_video(video_path, output_path, live=False)
        print(f"Video processing complete. Output saved to: {output_path}")
    elif mode == 'webcam':
        process_video(0, None, live=True)
