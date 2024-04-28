import cv2
import numpy as np
import os
from tqdm import tqdm
import time

# def calculate_speed(prev_position, current_position, time_elapsed):
#     distance = np.sqrt((current_position[0] - prev_position[0])**2 + (current_position[1] - prev_position[1])**2)
#     return distance / time_elapsed

def merge_boxes(boxes, threshold=30):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[0])
    merged = [boxes[0]]
    for box in boxes[1:]:
        if box[0] <= merged[-1][0] + merged[-1][2] + threshold:
            merged[-1] = (min(merged[-1][0], box[0]),
                          min(merged[-1][1], box[1]),
                          max(merged[-1][0] + merged[-1][2], box[0] + box[2]) - min(merged[-1][0], box[0]),
                          max(merged[-1][1] + merged[-1][3], box[1] + box[3]) - min(merged[-1][1], box[1]))
        else:
            merged.append(box)
    return merged

def draw_transparent_box(img, start_point, end_point, color, opacity):
    overlay = img.copy()
    cv2.rectangle(overlay, start_point, end_point, color, -1)
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

def process_video(video_path, output_path, live=False):
    cap = cv2.VideoCapture(video_path)
    ret, background = cap.read()
    if not ret:
        print("Failed to capture video")
        return

    if not live:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    else:
        out = None

    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)
    prev_frame = background_gray

    if live:
        print("Starting live motion detection. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.5, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Threshold to get motion areas
            _, motion_mask = cv2.threshold(mag, 1.5, 255, cv2.THRESH_BINARY)  # Increased threshold
            motion_mask = cv2.dilate(motion_mask, None, iterations=2)
            contours, _ = cv2.findContours(motion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bounding_boxes = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > 100]
            merged_boxes = merge_boxes(bounding_boxes)

            for (x, y, w, h) in merged_boxes:
                draw_transparent_box(frame, (x, y), (x+w, y+h), (0, 255, 0, 125), 0.4)

            cv2.imshow('Live Motion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            prev_frame = gray
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(total_frames), desc="Processing Video"):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 20, 3, 7, 1.5, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Threshold to get motion areas
            _, motion_mask = cv2.threshold(mag, 1.5, 255, cv2.THRESH_BINARY)  # Increased threshold
            motion_mask = cv2.dilate(motion_mask, None, iterations=2)
            contours, _ = cv2.findContours(motion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bounding_boxes = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > 100]
            merged_boxes = merge_boxes(bounding_boxes)

            for (x, y, w, h) in merged_boxes:
                draw_transparent_box(frame, (x, y), (x+w, y+h), (0, 255, 0, 125), 0.4)

            out.write(frame)
            prev_frame = gray

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    output_dir = 'processed_videos'
    os.makedirs(output_dir, exist_ok=True)

    mode = input("Enter 'video' to process a video file or 'webcam' for live feed: ").strip().lower()
    if mode == 'video':
        video_path = input("Enter the full path to the video file: ").strip()
        if video_path:
            output_path = os.path.join(output_dir, os.path.basename(video_path))
            start_time = time.time()
            process_video(video_path, output_path, live=False)
            end_time = time.time()
            print("Video processing complete. Output saved to:", output_path)
            print("Time taken:", end_time - start_time, "seconds")
        else:
            print("No video path provided.")
    elif mode == 'webcam':
        start_time = time.time()
        process_video(0, None, live=True)
        end_time = time.time()
        print("Time taken:", end_time - start_time, "seconds")
    else:
        print("Invalid mode selected.")
