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

## Scripts
Run `pip install moviepy` inside virtual environment, then run avi_to_mp4.py to convert AVI files to MP4 format.

# Distortions

## Denoising
All needed files are located inside: `distortions/denoising`. Here is the link to the custom dataset used: https://drive.google.com/drive/folders/1GcqmkXY_qymql-VENK1SV7A7QgcJAcuA?usp=sharing

### Overview
The denoising module (`noise_preprocessing.py`) provides preprocessing filters to reduce noise in video frames before object detection. This can improve detection accuracy and tracking stability on noisy footage.

### Available Filters
1. **None** - No filtering (baseline)
2. **Gaussian Blur** - Mathematically-driven approach that estimates noise level and applies adaptive Gaussian filtering
3. **Median Filter** - Empirically-driven approach using median filtering with a 3x3 kernel

### Files
- `noise_preprocessing.py` - Denoising functions
- `object_tracking_2.py` - Main vehicle speed estimation script with denoising support
- `batch_runner.py` - Automated batch processing script to run all 3 filters on all videos in specified dataset

### Running with Denoising
**Single video with specified filter:**
```bash
# No filtering (baseline)
python object_tracking_2.py --video content/highway.mp4 --output output.mp4 --filter none

# Gaussian filtering
python object_tracking_2.py --video content/highway.mp4 --output output.mp4 --filter gaussian

# Median filtering
python object_tracking_2.py --video content/highway.mp4 --output output.mp4 --filter median
```

### Batch Processing
To compare all filters across multiple videos:
1. **Load input videos in:** `content/noise/kaggle_mp4/` or `content/noise/custom_mp4/`
2. **Configure batch_runner.py:**
   - Set `DATASET = "kaggle_mp4"` or `DATASET = "custom_mp4"` (line 9)
   - For custom videos, define polygon coordinates in `CUSTOM_POLYGONS` dict (lines 15-20)
3. **Run batch processing:**
   ```bash
   python batch_runner.py
   ```

4. **Results saved to:** `content/batch_results/noise/<dataset>/`
   - `all_results.csv` - Per-video, per-filter metrics
   - `aggregate_statistics.csv` - Summary statistics across all videos
   - `videos/` - Processed output videos with annotations
   - `*_detailed.csv` - Per-frame, per-track detailed logs

### Visualizer
   - `visualize_results.csv` - Plots displaying the resulting metrics


## Scripts
- `extract_videos.py` - Extracts videos with "clear" weather from Kaggle dataset using metadata file
- `avi_to_mp4.py` - Converts AVI files to MP4 format (requires `pip install moviepy`)


## Notes
- Default confidence threshold: 0.50
- Output shows bounding boxes, track IDs, and speed estimates in km/h
- Blue polygon outline shows the region of interest