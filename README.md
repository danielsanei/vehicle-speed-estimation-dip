# Vehicle Speed Estimation Setup Guide

## Dataset
Using highway traffic videos from: https://www.kaggle.com/datasets/aryashah2k/highway-traffic-videos-dataset/data

## Quick Start

1. **Download any video from the Kaggle dataset**
   - Rename it to `highway.mp4`
   - Place it in the `content/` folder

2. **Run the comamnd**
```bash
   python object_tracking.py
```

3. **Output**
   - Annotated video will be saved as `content/output.mp4`

## Code Changes Made

We made the following changes to adapt the original code for the 320x240 Kaggle dataset videos:

### 1. Fixed undefined variable bug (line 166)
- Changed `if class_id != opt.class_id` 
- To `if int(label) != opt.class_id`

### 2. Adjusted SOURCE_POLYGONE coordinates for 320x240 resolution (line 100)
The original code was designed for 1920x1080 videos. We updated the polygon coordinates to match our smaller video resolution:

- Original: `[[18, 550], [1852, 608], [1335, 370], [534, 343]]`
- Updated to: `[[20, 200], [300, 220], [280, 100], [40, 80]]`

This ensures the region of interest polygon fits within the 320x240 frame.

### 3. Created configs/coco.names file
Added the COCO class names file required for object label mapping.

## Notes
- Default confidence threshold: 0.50
- Output shows bounding boxes, track IDs, and speed estimates in km/h
- Blue polygon outline shows the region of interest