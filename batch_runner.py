import os
import csv
from pathlib import Path
import numpy as np
from object_tracking_2 import process_video

# Configuration
# DATASET = "kaggle_mp4"
DATASET = "custom_mp4"
VIDEO_DIR = f"content/noise/{DATASET}"
OUTPUT_DIR = f"content/batch_results/noise/{DATASET}"
RESULTS_CSV = f"{OUTPUT_DIR}/all_results.csv"
AGGREGATE_CSV = f"{OUTPUT_DIR}/aggregate_statistics.csv"
FILTER_MODES = ["none", "gaussian", "median"]
CUSTOM_POLYGONS = {
    "left.mp4": np.array([[544, 1057], [954, 1057], [851, 557], [629, 557]], dtype=np.float32),
    "far_left.mp4": np.array([[544, 1057], [954, 1057], [851, 557], [629, 557]], dtype=np.float32),
    "right.mp4": np.array([[749, 1057], [1262, 1057], [1012, 557], [734, 557]], dtype=np.float32),
    "middle.mp4": np.array([[496, 1057], [1242, 1057], [1042, 408], [637, 408]], dtype=np.float32),
}

def get_video_files(directory):
    """Get all video files from directory."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(directory).glob(f"*{ext}"))
    return sorted(video_files)


def run_batch_processing():
    """Run object tracking on all videos with different filters."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "videos"), exist_ok=True)
    
    # Get all video files
    video_files = get_video_files(VIDEO_DIR)
    
    if not video_files:
        print(f"No video files found in {VIDEO_DIR}")
        return None
    
    print(f"\nFound {len(video_files)} video files")
    print(f"Processing with filters: {FILTER_MODES}")
    print("="*80)
    
    all_results = {}
    
    # Process each video with each filter
    total_tasks = len(video_files) * len(FILTER_MODES)
    current_task = 0
    
    for video_idx, video_file in enumerate(video_files, 1):
        video_name = video_file.stem
        all_results[video_name] = {}
        
        for filter_mode in FILTER_MODES:
            current_task += 1
            
            # Define output path
            output_filename = f"{video_name}_{filter_mode}.mp4"
            output_path = os.path.join(OUTPUT_DIR, "videos", output_filename)
            
            # Print progress
            print(f"[{current_task}/{total_tasks}] Processing: {video_name}")
            print(f"    Filter: {filter_mode}")
            
            # Process video
            try:
                import time
                start_time = time.time()

                # get polygon
                video_filename = video_file.name
                polygon = CUSTOM_POLYGONS.get(video_filename, None)
                
                metrics = process_video(
                    video_path=str(video_file),
                    output_path=output_path,
                    filter_mode=filter_mode,
                    conf_threshold=0.50,
                    class_id=None,
                    blur_id=None,
                    no_display=True,
                    polygon=polygon
                )
                
                elapsed = time.time() - start_time
                
                if metrics:
                    all_results[video_name][filter_mode] = metrics
                    print(f"    ✓ Completed in {elapsed:.1f}s")
                    print(f"    Detections: {metrics['total_detections']}, Avg confidence: {metrics['avg_yolo_confidence']:.3f}")
                else:
                    print(f"    ✗ Failed")
                    all_results[video_name][filter_mode] = None
                    
            except Exception as e:
                print(f"    ✗ Error: {str(e)}")
                all_results[video_name][filter_mode] = None
    
    print("="*80)
    print(f"Batch processing complete! Processed {current_task}/{total_tasks} tasks")
    print("="*80)
    
    return all_results


def save_results(all_results):
    """Save results to CSV file."""
    
    # Prepare CSV rows
    csv_rows = []
    for video_name, filters in all_results.items():
        for filter_mode, metrics in filters.items():
            if metrics:
                row = {
                    'video_name': video_name,
                    'filter_mode': filter_mode,
                    'total_frames': int(metrics['total_frames']),
                    'total_detections': int(metrics['total_detections']),
                    'avg_detections_per_frame': float(metrics['avg_detections_per_frame']),
                    'avg_yolo_confidence': float(metrics['avg_yolo_confidence']),
                    'frames_with_detections': int(metrics['frames_with_detections']),
                    'detection_rate': float(metrics['detection_rate']),
                    'avg_speed_variance': float(metrics['avg_speed_variance']),
                    'max_speed_variance': float(metrics['max_speed_variance']),
                }
                csv_rows.append(row)
    
    if csv_rows:
        with open(RESULTS_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n✓ Results saved to {RESULTS_CSV}")
    else:
        print("\n✗ No results to save")


def compute_aggregate_statistics(all_results):
    """Compute aggregate statistics across all videos."""
    
    stats = {}
    
    for filter_mode in FILTER_MODES:
        # Collect metrics for this filter
        detections_per_frame = []
        confidences = []
        detection_rates = []
        avg_speed_variances = []
        max_speed_variances = []
        
        for video_name, filters in all_results.items():
            if filter_mode in filters and filters[filter_mode]:
                metrics = filters[filter_mode]
                detections_per_frame.append(metrics['avg_detections_per_frame'])
                confidences.append(metrics['avg_yolo_confidence'])
                detection_rates.append(metrics['detection_rate'])
                avg_speed_variances.append(metrics['avg_speed_variance'])
                max_speed_variances.append(metrics['max_speed_variance'])
        
        # Calculate statistics
        stats[filter_mode] = {
            'num_videos': len(detections_per_frame),
            'avg_detections_per_frame': {
                'mean': np.mean(detections_per_frame) if detections_per_frame else 0,
                'std': np.std(detections_per_frame) if detections_per_frame else 0,
                'median': np.median(detections_per_frame) if detections_per_frame else 0,
            },
            'avg_yolo_confidence': {
                'mean': np.mean(confidences) if confidences else 0,
                'std': np.std(confidences) if confidences else 0,
                'median': np.median(confidences) if confidences else 0,
            },
            'detection_rate': {
                'mean': np.mean(detection_rates) if detection_rates else 0,
                'std': np.std(detection_rates) if detection_rates else 0,
                'median': np.median(detection_rates) if detection_rates else 0,
            },
            'avg_speed_variance': {
                'mean': np.mean(avg_speed_variances) if avg_speed_variances else 0,
                'std': np.std(avg_speed_variances) if avg_speed_variances else 0,
                'median': np.median(avg_speed_variances) if avg_speed_variances else 0,
            },
            'max_speed_variance': {
                'mean': np.mean(max_speed_variances) if max_speed_variances else 0,
                'std': np.std(max_speed_variances) if max_speed_variances else 0,
                'median': np.median(max_speed_variances) if max_speed_variances else 0,
            }
        }
    
    return stats


def print_aggregate_statistics(stats):
    """Print aggregate statistics in a readable format."""
    
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS ACROSS ALL VIDEOS")
    print("="*80)
    
    for filter_mode in FILTER_MODES:
        print(f"\n{'#'*80}")
        print(f"FILTER MODE: {filter_mode.upper()}")
        print(f"{'#'*80}")
        print(f"Number of videos processed: {stats[filter_mode]['num_videos']}")
        print()
        
        # Print each metric
        for metric_name, values in stats[filter_mode].items():
            if metric_name == 'num_videos':
                continue
            
            # Format metric name for display
            display_name = metric_name.replace('_', ' ').title()
            print(f"{display_name}:")
            print(f"  Mean:   {values['mean']:.4f}")
            print(f"  Median: {values['median']:.4f}")
            print(f"  Std:    {values['std']:.4f}")
            print()
    
    # Comparison between filters
    print("\n" + "="*80)
    print("FILTER COMPARISON (Higher values = Better detection, Lower variance = More stable)")
    print("="*80)
    
    if len(FILTER_MODES) == 2:
        filter1, filter2 = FILTER_MODES
        
        print(f"\nComparing {filter1.upper()} vs {filter2.upper()}:")
        print("-" * 80)
        
        # Compare key metrics
        metrics_to_compare = [
            ('avg_detections_per_frame', 'higher_better'),
            ('avg_yolo_confidence', 'higher_better'), 
            ('detection_rate', 'higher_better'),
            ('avg_speed_variance', 'lower_better')
        ]
        
        for metric, direction in metrics_to_compare:
            val1 = stats[filter1][metric]['mean']
            val2 = stats[filter2][metric]['mean']
            diff = val2 - val1
            pct_change = (diff / val1 * 100) if val1 != 0 else 0
            
            display_name = metric.replace('_', ' ').title()
            
            # Determine winner
            if direction == 'higher_better':
                winner = filter2 if val2 > val1 else filter1
                symbol = "↑" if val2 > val1 else "↓"
            else:  # lower_better
                winner = filter2 if val2 < val1 else filter1
                symbol = "↓" if val2 < val1 else "↑"
            
            print(f"\n{display_name}:")
            print(f"  {filter1}: {val1:.4f}")
            print(f"  {filter2}: {val2:.4f}")
            print(f"  Difference: {diff:+.4f} ({pct_change:+.2f}%) {symbol}")
            print(f"  Winner: {winner}")
    
    print("\n" + "="*80)


def main():
    print("="*80)
    print("BATCH VIDEO PROCESSING")
    print("="*80)
    print(f"Input directory: {VIDEO_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Run batch processing
    all_results = run_batch_processing()
    
    if all_results:
        # Save results to CSV
        save_results(all_results)
        
        # Compute and print statistics
        stats = compute_aggregate_statistics(all_results)
        print_aggregate_statistics(stats)
        
        # Save aggregate stats to CSV
        save_aggregate_stats(stats)
        
        print("\n" + "="*80)
        print("✓ BATCH PROCESSING COMPLETE!")
        print("="*80)
        print(f"Individual results: {RESULTS_CSV}")
        print(f"Aggregate stats: {AGGREGATE_CSV}")
        print(f"Output videos: {OUTPUT_DIR}/videos/")
    else:
        print("\n✗ Batch processing failed - no results generated")


def save_aggregate_stats(stats):
    """Save aggregate statistics to CSV."""
    csv_rows = []
    
    for filter_mode in FILTER_MODES:
        for metric_name, values in stats[filter_mode].items():
            if metric_name == 'num_videos':
                continue
            
            row = {
                'filter_mode': filter_mode,
                'metric': metric_name,
                'mean': float(values['mean']),
                'median': float(values['median']),
                'std': float(values['std'])
            }
            csv_rows.append(row)
    
    if csv_rows:
        with open(AGGREGATE_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filter_mode', 'metric', 'mean', 'median', 'std'])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"✓ Aggregate statistics saved to {AGGREGATE_CSV}")


if __name__ == "__main__":
    main()