# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# dataset + directory configuration
DATASET = "kaggle_mp4"  # "kaggle_mp4" or "custom_mp4"
RESULTS_DIR = f"content/batch_results/noise/{DATASET}"
OUTPUT_DIR = f"{RESULTS_DIR}/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# load data
all_results = pd.read_csv(f"{RESULTS_DIR}/all_results.csv")

# plot 1: tracking metrics comparison (bar chart)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
filter_colors = {'none': '#7f7f7f', 'gaussian': '#2ca02c', 'median': '#d62728'}
metrics = [
    ('unique_track_ids', 'Unique Track IDs\n(Lower = Better)', 'lower'),
    ('avg_track_duration', 'Avg Track Duration (frames)\n(Higher = Better)', 'higher'),
    ('track_stability_score', 'Track Stability Score\n(Higher = Better)', 'higher')
]
for idx, (metric, title, direction) in enumerate[tuple[str, str, str]](metrics):
    ax = axes[idx]
    # get mean values for each filter
    data = all_results.groupby('filter_mode')[metric].mean().reset_index()
    # create bars
    bars = ax.bar(data['filter_mode'], data[metric], 
                   color=[filter_colors[f] for f in data['filter_mode']],
                   edgecolor='black', linewidth=1.5)
    # add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    # highlight best performer with gold border
    best_idx = data[metric].argmin() if direction == 'lower' else data[metric].argmax()
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    # format axes
    ax.set_xlabel('Filter Mode', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_tracking_metrics_comparison.png", dpi=300, bbox_inches='tight')

# plot 2: detection quality vs tracking stability tradeoffs
fig, ax = plt.subplots(figsize=(10, 8))
for filter_mode in ['none', 'gaussian', 'median']:
    filter_data = all_results[all_results['filter_mode'] == filter_mode]
    x = filter_data['avg_yolo_confidence']
    y = filter_data['track_stability_score']
    # plot scatter points for each video
    ax.scatter(x, y, 
               s=100, 
               alpha=0.6, 
               color=filter_colors[filter_mode],
               label=filter_mode.upper(),
               edgecolors='black',
               linewidth=1)
    # add mean marker
    mean_x = x.mean()
    mean_y = y.mean()
    ax.scatter(mean_x, mean_y,
               s=400,
               marker='*',
               color=filter_colors[filter_mode],
               edgecolors='black',
               linewidth=2,
               zorder=5)
# format axes
ax.set_xlabel('Average YOLO Confidence (Detection Quality)', fontsize=12, fontweight='bold')
ax.set_ylabel('Track Stability Score (Higher = Better)', fontsize=12, fontweight='bold')
ax.set_title('Detection Quality vs Tracking Stability\n(★ = mean per filter)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_quality_vs_stability_scatter.png", dpi=300, bbox_inches='tight')

# plot 3: radar chart (compare overall metrics)
metrics_radar = {
    'Detection\nRate': ('detection_rate', False),  # False = no inversion
    'YOLO\nConfidence': ('avg_yolo_confidence', False),
    'Track\nDuration': ('avg_track_duration', False),
    'Stability\nScore': ('track_stability_score', False),
    'Low\nFragmentation': ('unique_track_ids', True),  # True = invert (lower is better)
    'Speed\nStability': ('avg_speed_variance', True)
}
categories = list(metrics_radar.keys())
num_vars = len(categories)
# compute angles for radar chart
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # complete the circle
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
for filter_mode in ['none', 'gaussian', 'median']:
    values = []
    filter_data = all_results[all_results['filter_mode'] == filter_mode]
    for col, invert in metrics_radar.values():
        val = filter_data[col].mean()
        # normalize to 0-100 scale
        all_vals = all_results[col]
        if invert:
            # invert scale for lower is better metrics
            normalized = 100 * (1 - (val - all_vals.min()) / (all_vals.max() - all_vals.min() + 1e-6))
        else:
            normalized = 100 * (val - all_vals.min()) / (all_vals.max() - all_vals.min() + 1e-6)
        values.append(normalized)
    values += values[:1]
    # plot and fill radar
    ax.plot(angles, values, 'o-', linewidth=2, label=filter_mode.upper(), 
            color=filter_colors[filter_mode])
    ax.fill(angles, values, alpha=0.15, color=filter_colors[filter_mode])
# format polar plot
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
ax.set_title('Overall Performance Comparison\n(Normalized scores: higher = better)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_performance_radar.png", dpi=300, bbox_inches='tight')

# summarize statistics
print("\n" + "="*70)
print("Visualization Summary")
print("="*70)
print(f"\nDataset: {DATASET}")
print(f"Total videos: {all_results['video_name'].nunique()}")
print(f"Output directory: {OUTPUT_DIR}")
print("\nGenerated plots:")
print("  1. 01_tracking_metrics_comparison.png - Bar charts of key metrics")
print("  2. 02_quality_vs_stability_scatter.png - Detection vs tracking trade-off")
print("  3. 03_performance_radar.png - Overall performance comparison")

# print key findings
print("\n" + "="*70)
print("Key Metrics")
print("="*70)
summary = all_results.groupby('filter_mode').agg({
    'unique_track_ids': 'mean',
    'avg_track_duration': 'mean',
    'track_stability_score': 'mean'
}).round(2)

print("\n" + summary.to_string())

best_filter = summary['track_stability_score'].idxmax()
print(f"\nWinner: {best_filter.upper()} filter")
print(f"   Best overall tracking stability across all metrics")
print("\n" + "="*70)