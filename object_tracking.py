import cv2
import torch
import numpy as np
import argparse, os
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from ultralytics import YOLO
from noise_preprocessing import denoise_gaussian    # import denoise filtering

# noise filtering flag
# FILTER_MODE = "none"
# FILTER_MODE = "gaussian"
# FILTER_MODE = "median"
###

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
    # add filter flag argument
    parser.add_argument(
        "--filter",
        type=str,
        default="none",
        choices=["none", "gaussian", "median"],
        help="Selecting processing filter"
    )
    ###
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


def read_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # select denoise filter
        if FILTER_MODE == "gaussian":
            frame = denoise_gaussian(frame)
        #elif FILTER_MODE == "median":
            #frame = denoise_median(frame)
        ###
        yield frame 


def main(_argv):

    FRAME_WIDTH=30
    FRAME_HEIGHT=100

    # original --> for 1920 x 1080 videos
    # SOURCE_POLYGONE = np.array([[18, 550], [1852, 608],[1335, 370], [534, 343]], dtype=np.float32)
    # BIRD_EYE_VIEW = np.array([[0, 0], [FRAME_WIDTH, 0], [FRAME_WIDTH, FRAME_HEIGHT],[0, FRAME_HEIGHT]], dtype=np.float32)

    # new --> for 320 x 240 videos (Kaggle dataset)
    # SOURCE_POLYGONE = np.array([[20, 200], [300, 220], [280, 100], [40, 80]], dtype=np.float32)
    # BIRD_EYE_VIEW = np.array([[0, 0],  [FRAME_WIDTH, 0], [FRAME_WIDTH, FRAME_HEIGHT], [0, FRAME_HEIGHT]], dtype=np.float32)
    # change bird eye view if speed is weird

    # new2 --> for iPhone videos (custom dataset)
    SOURCE_POLYGONE = np.array([[500, 450], [1600, 480], [1800, 980], [400, 960]], dtype=np.float32)
    BIRD_EYE_VIEW = np.array([[0, 0], [FRAME_WIDTH, 0], [FRAME_WIDTH, FRAME_HEIGHT],[0, FRAME_HEIGHT]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(SOURCE_POLYGONE, BIRD_EYE_VIEW)


    # Initialize the video capture
    video_input = opt.video

    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        print('Error: Unable to open video source.')
        return
    
  
    frame_generator = read_frames(cap)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    pts = SOURCE_POLYGONE.astype(np.int32) 
    pts = pts.reshape((-1, 1, 2))

    polygon_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [pts], 255)
    # video writer objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(opt.output, fourcc, fps, (frame_width, frame_height))

    # Initialize the DeepSort tracker
    tracker = DeepSort(max_age=50)
    # Load YOLO model
    model = YOLO("yolov10n.pt")
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
    total_detections = 0            # YOLO detection count
    frames_with_detections = 0
    total_confidence = 0            # YOLO confidence scores
    total_confidence_count = 0              
    speed_variances = {}            # variances in speed (indicate noise)
    ###
    
    while True:
        try:
            frame = next(frame_generator)
        except StopIteration:
            break
        # Run model on each frame
        with torch.no_grad():
            results = model(frame)
        detect = []
        for pred in results:
            for box in pred.boxes:    
                x1, y1, x2, y2 = map(int, box.xyxy[0] )
                confidence = box.conf[0]
                # METRIC: YOLO confidence
                total_confidence += float(confidence)
                total_confidence_count += 1
                ###
                label = box.cls[0]  

                # Filter out weak detections by confidence threshold and class_id
                if opt.class_id is None:
                    if confidence < opt.conf:
                        continue
                else:
                    if int(label) != opt.class_id or confidence < opt.conf:    #  class_id instead of int(label)
                        continue            
                    
                if polygon_mask[(y1 + y2) // 2, (x1 + x2) // 2] == 255:
                    detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, int(label)])
                    # METRIC: track current detection
                    total_detections += 1
                    ###
        # METRIC: whether current frame has detection
        if len(detect) > 0:
            frames_with_detections += 1
        ###
        tracks = tracker.update_tracks(detect, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()    # YOLO-predicted object class (i.e. car, person, motorcycle)
            x1, y1, x2, y2 = map(int, ltrb)     # bounding box coordinates
            if polygon_mask[(y1+y2)//2,(x1+x2)//2] == 0:
                tracks.remove(track)
            color = colors[class_id]
            B, G, R = map(int, color)
            text = f"{track_id} - {class_names[class_id]}"
            center_pt = np.array([[(x1+x2)//2, (y1+y2)//2]], dtype=np.float32)
            transformed_pt = cv2.perspectiveTransform(center_pt[None, :, :], M)
            if track_id in prev_positions:
                prev_position = prev_positions[track_id]
                distance = calculate_distance(prev_position, transformed_pt[0][0])
                speed = calculate_speed(distance, fps)
                if track_id in speed_accumulator:
                    speed_accumulator[track_id].append(speed)
                    # METRIC: speed variance (keep track of speeds, compute variance at the end)
                    if track_id not in speed_variances:
                        speed_variances[track_id] = []
                    speed_variances[track_id].append(speed)
                    ###
                    if len(speed_accumulator[track_id]) > 100:
                        speed_accumulator[track_id].pop(0)
                else:
                    speed_accumulator[track_id] = []
                    speed_accumulator[track_id].append(speed)
            prev_positions[track_id] = transformed_pt[0][0]
            # Draw bounding box and text
            frame = draw_corner_rect(frame, (x1, y1, x2 - x1, y2 - y1), line_length=15, line_thickness=3, rect_thickness=1, rect_color=(B, G, R), line_color=(R, G, B))
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 10, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if track_id in speed_accumulator :
                avg_speed = sum(speed_accumulator[track_id]) / len(speed_accumulator[track_id])
                #print(avg_speed)
                cv2.rectangle(frame, (x1 - 1, y1-40 ), (x1 + len(f"Speed: {avg_speed:.0f} km/h") * 10, y1-20), (0, 0, 255), -1)
                cv2.putText(frame, f"Speed: {avg_speed:.0f} km/h", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # Apply Gaussian Blur
            if opt.blur_id is not None and class_id == opt.blur_id:
                print("true")
                if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)

        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(frame, f"Height: {FRAME_HEIGHT}", (1500, 900), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Width: {FRAME_WIDTH}", (1530, 930), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('speed_estimation', frame)
        writer.write(frame)
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            fps_calc = frame_count / elapsed_time
            print(f"FPS: {fps_calc:.2f}")
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # METRICS: display results
    print("\n### METRICS ###")
    print("Filter Mode:", FILTER_MODE)
    print(f"Total detections: {total_detections}")                                  # detections
    print(f"Average detections per frame: {total_detections / frame_count:.2f}")
    if total_confidence_count > 0:                                                  # YOLO confidence
        avg_conf = total_confidence / total_confidence_count
        print(f"Average YOLO confidence: {avg_conf:.4f}")
    else:
        print("Average YOLO confidence: N/A (no detections)")
    print(f"Frames with >= 1 detection: {frames_with_detections}/{frame_count}")
    for tid, speeds in speed_variances.items():                                     # speed variance
        if len(speeds) > 3:
            var = np.var(speeds)
            print(f"Track ID {tid}: speed variance = {var:.2f}")
    ###

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    opt = parse_args()
    # select filter type
    FILTER_MODE = opt.filter.lower()
    ###
    main(opt)