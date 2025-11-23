# imports
from moviepy import VideoFileClip    # read, edit, export video files
import os

# define folders
input_folder = "content/noise/kaggle_avi"
output_folder = "content/noise/kaggle_mp4"
os.makedirs(output_folder, exist_ok=True)   # create output folder if does not exist

# count total number of AVI files to convert
avi_files = [f for f in os.listdir(input_folder) if f.endswith(".avi")]
total_files = len(avi_files)

# convert AVI to MP4
print(f"Converting {total_files} .avi files... ")
i = 1
for file in os.listdir(input_folder):
    if file.endswith(".avi"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".mp4")           # change .avi extension to .mp4

        print(i, end=" ", flush=True)
        clip = VideoFileClip(input_path)                                                        # open video clip
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)      # write video in MP4 format
        clip.close()
        i += 1
print("\nConversions complete")