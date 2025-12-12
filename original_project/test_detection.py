import cv2
from ultralytics import YOLO
import os

# Check if video file exists
video_path = "content/highway.mp4"  # Adjust this to your actual video path
print(f"Checking for video at: {video_path}")
print(f"File exists: {os.path.exists(video_path)}")

if os.path.exists(video_path):
    print(f"File size: {os.path.getsize(video_path)} bytes")

# Try to open video
cap = cv2.VideoCapture(video_path)
print(f"Video opened successfully: {cap.isOpened()}")

if cap.isOpened():
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"  Width: {width}")
    print(f"  Height: {height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {frame_count}")
    
    # Try to read first frame
    ret, frame = cap.read()
    print(f"Frame read successfully: {ret}")
    
    if ret:
        print(f"Frame shape: {frame.shape}")
        
        # Load YOLO model
        print("\nLoading YOLO model...")
        model = YOLO("yolov10n.pt")
        print("Model loaded successfully")
        
        # Run detection
        print("Running detection...")
        results = model(frame)
        print(f"Detection complete!")
        print(f"Number of detections: {len(results[0].boxes)}")
        
        # Show what was detected
        for i, box in enumerate(results[0].boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"  Object {i+1}: Class {cls}, Confidence {conf:.2f}")
        
        # Save annotated image
        annotated = results[0].plot()
        cv2.imwrite("test_detection.jpg", annotated)
        print("\nSaved annotated image to test_detection.jpg")
    else:
        print("ERROR: Could not read frame from video")
else:
    print("ERROR: Could not open video file")
    print("\nPlease check:")
    print("1. The video file path is correct")
    print("2. The video file is not corrupted")
    print("3. You have the correct codec installed")

cap.release()
print("\nDone!")