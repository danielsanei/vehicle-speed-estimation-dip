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

## Dehazing

### Overview
`dehazing.py` is a module containing several DCP-based dehazing methods, namely joint demosaicing and dehazing and Kirsh filterbank soft-matting.

### Files
- `dehazing.py` - Dehazing functions
- `count_detections.py` - Enumerates detections

### Usage

1. Download the fog dataset and load into ./input: https://drive.google.com/drive/u/0/folders/1_MD8IKrYOPNCUmPmUX6HDMi2uIGnVRDc

2. Generate output videos for all DCP dehazing variants:
```
python dehazing.py input/fog.mp4 output
```

3. Count detections for all dehazing methods:
```
python count_detections.py output
```

## Contrast
### Methodology: Stress Testing
We evaluated tracking robustness by generating a Synthetic Stress Dataset from the Kaggle highway videos. The dataset simulates adverse visual conditions through controlled degradations:
- **Darkness Levels:** 0% (original) to 80% (extreme low light)
- **Contrast Reduction:** Washed-out histograms simulating fog, haze, or glare

**Baselines:**
- None (control)
- Linear Stretch
- Naive Brightness Boost

**Histogram-Based:**
- Global Histogram Equalization (HE)
- CLAHE

**Mathematical Transforms:**
- Gamma Correction
- Sigmoid
- Logarithmic

**Frequency / Morphological:**
- Homomorphic Filtering
- Top-Hat Transform
- Retinex (Multi-Scale)

**Hybrid Pipelines:**
- Hybrid: Gamma + CLAHE
- Advanced Hybrid: Adaptive Gamma → CLAHE → Bilateral Denoising → Unsharp Masking

## Benchmark Results

| Method            | Stability Score | Improvement vs Baseline | 
|-------------------|-----------------|--------------------------|
| Advanced Hybrid   | 46.39           | +83.7%                   |
| CLAHE             | 38.62           | +52.9%                   |
| Retinex           | 34.45           | +36.4%                   |
| Global HE         | 26.83           | +6.2%                    |
| Baseline (None)   | 25.25           | 0%                       |

## Usage

1. Place your reference video inside the `content/` directory.
2. Open `contrastfinal.ipynb`.
3. Run the analysis cells to generate Efficiency Frontier and Robustness Curve plots.





# Cascaded Pipeline
## Synthetic Degradation System
Each frame is modified using deterministic mathematical transforms:

## Enhancement Pipelines
Each degraded frame is processed using one of several recovery pipelines:
### Basic Methods
- `none` — no enhancement  
- `clahe_only` — local contrast amplification in YUV  
- `dehaze_only` — dark channel‑based atmospheric recovery  

### Hybrid Methods
- `gamma_clahe` — adaptive gamma correction followed by CLAHE  
- `dehaze_clahe` — dehaze then local contrast enhancement  
- `denoise_clahe` — bilateral denoising then CLAHE  

### Full Recovery Pipeline
A cascaded multi‑stage enhancement:
```
Dehaze → Bilateral Denoising → Adaptive Gamma Correction → CLAHE
```
This pipeline aims to restore luminance, reduce noise, and sharpen edges before feeding the frame to the detector.

## Stress Test Scenarios
The suite includes 15 controlled scenarios:
- Darkness: mild, severe, extreme  
- Fog: mild, dense, extreme  
- Noise: mild, heavy  
- Motion blur: mild, severe  
- Combined composite environments  
- Pristine baseline  

## Analysis Engine
For each frame sequence:
- YOLOv8 performs detection  
- DeepSORT tracks object persistence  
- Brightness and detection confidence are recorded  

Metrics computed:
- **Unique IDs** detected  
- **Average tracking duration**  
- **Average detection confidence**  
- **Detections per frame / total detections**  
- **Average processed‑frame brightness**

These metrics quantify whether an enhancement pipeline restores sufficient structure for consistent detection.

1. Place source videos in the `content/` directory.  
2. Open `finalcascaded.ipynb` or `cascadedpipelinev3.ipynb`.  
3. Run the full stress test block to generate:
   - Scenario × Enhancement evaluations
   - Aggregated performance tables
   - Visualization dashboard  
4. Review results stored in `content/processed/stress_tests/`.

## Deployment Recommendation

- **Standard Conditions:** Use `clahe_only` for minimal latency.  
- **Challenging Conditions:** Use `gamma_clahe` or `denoise_clahe`.  
- **Extreme Low‑Light / Severe Fog:** Use `full_recovery` for maximum restoration.  

This module provides a comprehensive framework for evaluating and improving robustness of detection systems in degraded real‑world environments.


## Deployment Recommendations
- **Standard Deployment:** Use `method='clahe'` for strong improvements with negligible latency.
- **Extreme Low-Light Conditions:** Use `method='advanced_hybrid'` for nighttime or adverse surveillance scenarios. Approx. 5.5 ms per frame (~27 FPS).

## Notes
- Default confidence threshold: 0.50
- Output shows bounding boxes, track IDs, and speed estimates in km/h

- Blue polygon outline shows the region of interest


