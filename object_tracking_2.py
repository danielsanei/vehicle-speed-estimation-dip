import cv2
import torch
import numpy as np
import argparse, os
import sys
from contextlib import contextmanager
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from ultralytics import YOLO
from noise_preprocessing import denoise_gaussian, denoise_median


@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        nargs="?",
        default="content/highway.mp4",
        help="Path to input video"
    )
    parser.add_argument(
        "--output",
        type=str,
        nargs="?",
        help="path to output video",
        default="content/output.mp4"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="confidence threshold",
    )
    parser.add_argument(
        "--blur_id",
        type=int,
        default=None,
        help="class ID to apply Gaussian Blur",
    )
    parser.add_argument(
        "--class_id",
        type=int,
        default=None,
        help="class ID to track",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="none",
        choices=["none", "gaussian", "median"],
        help="Selecting processing filter"
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Disable video display window"
    )
    opt = parser.parse_args()
    return opt


def draw_corner_rect(img, bbox, line_length=30, line_thickness=5, rect_thickness=1,
                     rect_color=(255, 0, 255), line_color=(0, 255, 0)):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h

    if rect_thickness != 0:
        cv2.rectangle(img, bbox, rect_color, rect_thickness)

    # Top Left  x, y
    cv2.line(img, (x, y), (x + line_length, y), line_color, line_thickness)
    cv2.line(img, (x, y), (x, y + line_length), line_color, line_thickness)

    # Top Right  x1, y
    cv2.line(img, (x1, y), (x1 - line_length, y), line_color, line_thickness)
    cv2.line(img, (x1, y), (x1, y + line_length), line_color, line_thickness)

    # Bottom Left  x, y1
    cv2.line(img, (x, y1), (x + line_length, y1), line_color, line_thickness)
    cv2.line(img, (x, y1), (x, y1 - line_length), line_color, line_thickness)

    # Bottom Right  x1, y1
    cv2.line(img, (x1, y1), (x1 - line_length, y1), line_color, line_thickness)
    cv2.line(img, (x1, y1), (x1, y1 - line_length), line_color, line_thickness)

    return img  

def calculate_speed(distance, fps):
    return (distance *fps)*3.6


def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def read_frames(cap, filter_mode):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if filter_mode == "gaussian":
            with suppress_output():
                frame = denoise_gaussian(frame)
        elif filter_mode == "median":
            with suppress_output():
                frame = denoise_median(frame)
        yield frame 


def process_video(video_path, output_path, filter_mode="none", conf_threshold=0.50, 
                  class_id=None, blur_id=None, no_display=False, polygon=None):
    """
    Process a video and return metrics as a dictionary.
    
    Returns:
        dict: Dictionary containing all metrics
    """
    
    FRAME_WIDTH=30
    FRAME_HEIGHT=100

        # Use provided polygon or default based on resolution
    if polygon is not None:
        SOURCE_POLYGONE = polygon
    else:
        # Auto-detect based on video resolution
        cap_temp = cv2.VideoCapture(video_path)
        width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_temp.release()
        
        if width == 320:  # Kaggle dataset
            SOURCE_POLYGONE = np.array([[104, 215], [300, 235], [280, 45], [124, 25]], dtype=np.float32)
        else:  # 1920x1080 custom dataset default
            SOURCE_POLYGONE = np.array([[544, 1057], [954, 1057], [851, 557], [629, 557]], dtype=np.float32)
    
    BIRD_EYE_VIEW = np.array([[0, 0], [FRAME_WIDTH, 0], [FRAME_WIDTH, FRAME_HEIGHT], [0, FRAME_HEIGHT]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(SOURCE_POLYGONE, BIRD_EYE_VIEW)

    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Error: Unable to open video source: {video_path}')
        return None
    
    frame_generator = read_frames(cap, filter_mode)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    pts = SOURCE_POLYGONE.astype(np.int32) 
    pts = pts.reshape((-1, 1, 2))

    polygon_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [pts], 255)
    
    # video writer objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Initialize the DeepSort tracker
    tracker = DeepSort(max_age=50)
    # Load YOLO model
    with suppress_output():
        model = YOLO("yolov10n.pt")
        model.overrides['verbose'] = False
    # Load the COCO class labels
    classes_path = "configs/coco.names"
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3)) 
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    prev_positions={}
    speed_accumulator={}

    # tracking metrics
    total_detections = 0
    frames_with_detections = 0
    total_confidence = 0
    total_confidence_count = 0              
    speed_variances = {}
    
    while True:
        try:
            frame = next(frame_generator)
        except StopIteration:
            break
            
        # Run model on each frame (suppress output)
        with suppress_output():
            with torch.no_grad():
                results = model(frame)
        detect = []
        for pred in results:
            for box in pred.boxes:    
                x1, y1, x2, y2 = map(int, box.xyxy[0] )
                confidence = box.conf[0]
                total_confidence += float(confidence)
                total_confidence_count += 1
                label = box.cls[0]  

                # Filter out weak detections by confidence threshold and class_id
                if class_id is None:
                    if confidence < conf_threshold:
                        continue
                else:
                    if int(label) != class_id or confidence < conf_threshold:
                        continue            
                    
                if polygon_mask[(y1 + y2) // 2, (x1 + x2) // 2] == 255:
                    detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, int(label)])
                    total_detections += 1
                    
        if len(detect) > 0:
            frames_with_detections += 1
            
        tracks = tracker.update_tracks(detect, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id_det = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            if polygon_mask[(y1+y2)//2,(x1+x2)//2] == 0:
                tracks.remove(track)
            color = colors[class_id_det]
            B, G, R = map(int, color)
            text = f"{track_id} - {class_names[class_id_det]}"
            center_pt = np.array([[(x1+x2)//2, (y1+y2)//2]], dtype=np.float32)
            transformed_pt = cv2.perspectiveTransform(center_pt[None, :, :], M)
            if track_id in prev_positions:
                prev_position = prev_positions[track_id]
                distance = calculate_distance(prev_position, transformed_pt[0][0])
                speed = calculate_speed(distance, fps)
                if track_id in speed_accumulator:
                    speed_accumulator[track_id].append(speed)
                    if track_id not in speed_variances:
                        speed_variances[track_id] = []
                    speed_variances[track_id].append(speed)
                    if len(speed_accumulator[track_id]) > 100:
                        speed_accumulator[track_id].pop(0)
                else:
                    speed_accumulator[track_id] = []
                    speed_accumulator[track_id].append(speed)
            prev_positions[track_id] = transformed_pt[0][0]
            
            # Draw bounding box and text
            frame = draw_corner_rect(frame, (x1, y1, x2 - x1, y2 - y1), line_length=15, 
                                    line_thickness=3, rect_thickness=1, 
                                    rect_color=(B, G, R), line_color=(R, G, B))
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 10, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if track_id in speed_accumulator :
                avg_speed = sum(speed_accumulator[track_id]) / len(speed_accumulator[track_id])
                cv2.rectangle(frame, (x1 - 1, y1-40 ), (x1 + len(f"Speed: {avg_speed:.0f} km/h") * 10, y1-20), (0, 0, 255), -1)
                cv2.putText(frame, f"Speed: {avg_speed:.0f} km/h", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Apply Gaussian Blur
            if blur_id is not None and class_id_det == blur_id:
                if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)

        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        
        if not no_display:
            cv2.imshow('speed_estimation', frame)
        writer.write(frame)
        frame_count += 1
    
        if not no_display and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate final metrics
    metrics = {
        "filter_mode": filter_mode,
        "video_path": video_path,
        "total_frames": frame_count,
        "total_detections": total_detections,
        "avg_detections_per_frame": total_detections / frame_count if frame_count > 0 else 0,
        "avg_yolo_confidence": total_confidence / total_confidence_count if total_confidence_count > 0 else 0,
        "frames_with_detections": frames_with_detections,
        "detection_rate": frames_with_detections / frame_count if frame_count > 0 else 0,
        "speed_variances": {}
    }
    
    # Calculate speed variances
    for tid, speeds in speed_variances.items():
        if len(speeds) > 3:
            var = np.var(speeds)
            metrics["speed_variances"][tid] = var
    
    # Add aggregate speed variance
    if metrics["speed_variances"]:
        metrics["avg_speed_variance"] = np.mean(list(metrics["speed_variances"].values()))
        metrics["max_speed_variance"] = np.max(list(metrics["speed_variances"].values()))
    else:
        metrics["avg_speed_variance"] = 0
        metrics["max_speed_variance"] = 0

    cap.release()
    writer.release()
    if not no_display:
        cv2.destroyAllWindows()
    
    return metrics


def main():
    opt = parse_args()
    
    metrics = process_video(
        video_path=opt.video,
        output_path=opt.output,
        filter_mode=opt.filter.lower(),
        conf_threshold=opt.conf,
        class_id=opt.class_id,
        blur_id=opt.blur_id,
        no_display=opt.no_display
    )
    
    if metrics:
        # Print metrics
        print("\n### METRICS ###")
        print(f"Filter Mode: {metrics['filter_mode']}")
        print(f"Total detections: {metrics['total_detections']}")
        print(f"Average detections per frame: {metrics['avg_detections_per_frame']:.2f}")
        print(f"Average YOLO confidence: {metrics['avg_yolo_confidence']:.4f}")
        print(f"Frames with >= 1 detection: {metrics['frames_with_detections']}/{metrics['total_frames']}")
        print(f"Detection rate: {metrics['detection_rate']:.2%}")
        print(f"Average speed variance: {metrics['avg_speed_variance']:.2f}")
        print(f"Max speed variance: {metrics['max_speed_variance']:.2f}")
        
        if metrics["speed_variances"]:
            print("\nPer-track speed variances:")
            for tid, var in metrics["speed_variances"].items():
                print(f"  Track ID {tid}: {var:.2f}")


if __name__ == "__main__":
    main()