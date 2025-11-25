import cv2
import numpy as np
import sys

def get_dark_channel(image, patch_size=15):
    return cv2.erode(np.min(image, axis=2), np.ones((patch_size, patch_size), np.uint8))

def estimate_atmospheric_light(image, dark_channel, p=0.001):
    flat_image = image.reshape((-1, 3))
    indices = np.argsort(dark_channel.reshape(-1))[::-1]
    brightest_pixels = flat_image[indices[:int(p * len(indices))]]
    return np.mean(brightest_pixels, axis=0)

def estimate_transmission(image, atmospheric_light, omega=0.95, patch_size=15):
    # return 1 - omega * get_dark_channel(image / atmospheric_light, patch_size)
    return 1 - get_dark_channel(image / atmospheric_light, patch_size)

def recover_image(image, atmospheric_light, transmission, eps=0.001):
    transmission = np.clip(transmission, eps, 1)
    return np.clip((image - atmospheric_light) / transmission[:, :, np.newaxis] + atmospheric_light, 0, 255).astype(np.uint8)

def dehaze(frame):
    frame = frame.astype(np.float64)
    dark_channel = get_dark_channel(frame)
    atmospheric_light = estimate_atmospheric_light(frame, dark_channel)
    return recover_image(frame, atmospheric_light, estimate_transmission(frame, atmospheric_light))

# driver code
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Not enough arguments provided.")
        exit()

    # open video file
    video = cv2.VideoCapture(sys.argv[1])
    if not video.isOpened():
        print("Failed to open video.")
        exit()

    # create VideoWriter object
    out = cv2.VideoWriter(sys.argv[2], cv2.VideoWriter_fourcc(*'mp4v'), 60, (1920, 1200))

    # read in all video frames
    frame_count = 0
    before_frame = None
    after_frame = None
    while True:
        ret, frame = video.read()
        if not ret:
            break   # reached the end of video
    
        # apply dehazing to current frame
        dehazed_frame = dehaze(frame)
        frame_count += 1

        # save before, after frames
        before_frame = frame.copy()
        after_frame = dehazed_frame.copy()
        out.write(after_frame)

    out.release()
    print(f"Output video created successfully.")