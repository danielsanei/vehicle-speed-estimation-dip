# imports
import os
import shutil

# define paths
INFO_PATH = "content/kaggle_metadata.txt"
VIDEO_DIR = "content/kaggle_videos_avi/"
OUTPUT_DIR = "content/noise/kaggle_avi"

def main():

    # create output file if does not exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # read every line in text file
    with open(INFO_PATH, "r") as f:
        lines = f.readlines()

    # skip header line
    i = 0
    clear = 0
    clear_list = []      # store clear filenames for progress counter
    for line in lines[1:]:
        i += 1
        parts = line.strip().split('\t')

        # expected columns: filename, date, timestamp, direction, day/night, weather, start_frame, num_frames, class, notes
        if len(parts) < 7:      # only need to extract weather
            continue

        filename = parts[0]
        weather = parts[5]     

        # extract footage with clear weather
        if weather.lower() == "clear":
            clear_list.append(filename)     # store it
            clear += 1
            avi_name = filename + ".avi"
            src_path = os.path.join(VIDEO_DIR, avi_name)
            dst_path = os.path.join(OUTPUT_DIR, avi_name)

            # copy file over to noise/ directory
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                print(f"Missing file: {avi_name}")

    # progress printout (like avi_to_mp4.py)
    print(f"\nExtracting {clear} .avi files...")

    for idx, _ in enumerate(clear_list, start=1):
        print(idx, end=" ")
    print("\nExtraction complete")

if __name__ == "__main__":
    main()
