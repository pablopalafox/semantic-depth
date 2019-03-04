import os
import cv2
import glob


input_path = "results/stuttgart_video/result_sequence_imgs/*.png"
output_path = "results/stuttgart_video/result.mp4"

test_frame = cv2.imread("media/videos/stuttgart_video/stuttgart_02_000000_005102_leftImg8bit.png")
height, width = test_frame.shape[0], test_frame.shape[1]

video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), 8, (width, height))


for frame_path in sorted(glob.glob(input_path)):
    frame = cv2.imread(frame_path)
    video.write(frame)