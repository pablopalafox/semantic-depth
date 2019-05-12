import os
import cv2
import glob

# input_paths = ["../results/stuttgart_video/result_sequence_imgs/*.png", 
# 				"../results/stuttgart_video/rendered_sequence_top/*.png",
# 				"../results/stuttgart_video/rendered_sequence_good_frontal/*.png"]

# output_paths = ["../results/stuttgart_video/result_imgs.mp4", 
# 				"../results/stuttgart_video/result_top_render.mp4", 
# 				"../results/stuttgart_video/result_frontal_render.mp4"]



input_paths = ["../results/stuttgart_video/result_sequence_imgs/*.png"]

output_paths = ["../results/stuttgart_video/result_imgs.mp4"]

for i in range(len(input_paths)):
	print("Reading from", input_paths[i])

	test_frame = cv2.imread("../results/stuttgart_video/result_sequence_imgs/stuttgart_02_000000_005100_leftImg8bit.png")
	height, width = test_frame.shape[0], test_frame.shape[1]
	print(height, width)

	video = cv2.VideoWriter(output_paths[i], cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))

	for frame_path in sorted(glob.glob(input_paths[i])):
	    frame = cv2.imread(frame_path)
	    video.write(frame)
	print("Done", input_paths[i])