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

# choose median filtering kernel size based on noise level
def choose_median_kernel_size(noise_std):
    return 3
    # if noise_std < 5:
    #     return 3            # 3x3
    # else:
    #     return 5            # 5x5
    # else:
    #     return 7            # 7x7
    # else:
    #     return 9            # 9x9

# remove noise using Gaussian filtering (mathematically driven)
def denoise_gaussian(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    noise_std = estimate_noise(gray)
    sigma = choose_sigma(noise_std)
    k = choose_kernel_size(sigma)
    print(f"noise_std={noise_std:.2f}, sigma={sigma:.2f}, kernel={k}")
    return cv2.GaussianBlur(frame, (k,k), sigma)

# remove noise using Median filtering (empirically driven)
def denoise_median(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    noise_std = estimate_noise(gray)
    k = choose_median_kernel_size(noise_std)
    print(f"noise_std={noise_std:.2f}, kernel={k}")
    return cv2.medianBlur(frame, k)

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
    
        # apply denoising to current frame
        denoised_gaussian = denoise_gaussian(frame)
        denoised_median = denoise_median(frame)
        frame_count += 1

        # save before, after frames
        before_frame = frame.copy()
        after_gaussian = denoised_gaussian.copy()
        after_median = denoised_median.copy()

    # show results for final frame as sample display
    print(f"\Processed {frame_count} frames.")
    cv2.imshow("before", before_frame)
    cv2.imshow("after gaussian", after_gaussian)
    cv2.imshow("after median", after_median)
    cv2.waitKey(10000)