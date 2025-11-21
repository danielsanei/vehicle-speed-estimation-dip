# imports
import cv2
import numpy as np

# estimate noise magnitude
def estimate_noise(gray):
    blurred = cv2.GaussianBlur(gray, (7,7), 1.5)                        # create blurred image
    high_freq = gray.astype(np.float32) - blurred.astype(np.float32)    # get high frequency components
    return high_freq.std()

# convert noise magnitude to sigma heuristic
def choose_sigma(noise_std):
    return max(0.2, min(0.8, 0.2 * noise_std))  # take estimated noise, scale by 0.2, clamp lower/upper limits

# choose kernel size based on sigma heuristic
def choose_kernel_size(sigma):      # kernel should cover 3*std of Gaussian curve (99.7%)
    k = int(4 * sigma + 1)              # kernel width = 2*(3*std) + 1 (where +1 is for central pixels)
    return k+1 if k%2==0 else k     # ensure odd kernel dimensions

# remove noise using Gaussian filtering
def denoise_gaussian(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    noise_std = estimate_noise(gray)
    sigma = choose_sigma(noise_std)
    k = choose_kernel_size(sigma)
    print(f"noise_std={noise_std:.2f}, sigma={sigma:.2f}, kernel={k}")
    return cv2.GaussianBlur(frame, (k,k), sigma)

# driver code
if __name__ == "__main__":

    # open video file
    video = cv2.VideoCapture("content/highway.mp4")
    if not video.isOpened():
        print("Failed to open video.")
        exit()

    # read in all video frames
    frame_count = 0
    before_frame = None
    after_frame = None
    while True:
        ret, frame = video.read()
        if not ret:
            break   # reached the end of video
    
        # apply Gaussian filtering to current frame
        denoised_frame = denoise_gaussian(frame)
        frame_count += 1

        # save before, after frames
        before_frame = frame.copy()
        after_frame = denoised_frame.copy()

    # show results for final frame as sample display
    print(f"\Processed {frame_count} frames.")
    cv2.imshow("before", before_frame)
    cv2.imshow("after", after_frame)
    cv2.waitKey(10000)