import cv2

# Video paths
video1_path = 'video_pred.mp4'
video2_path = 'video_truth.mp4' 

video_row1 = ['/home/advit/aug30_exps/_videos_/vid_no_temps_count_1.mp4', 
              '/home/advit/aug30_exps/_videos_/vid_all_temps_count_1.mp4', 
              '/home/advit/aug30_exps/_videos_/vid_all_temps_count_2.mp4']

video_row2 = ['/home/advit/aug30_exps/_videos_/vid_all_temps_count_3.mp4', 
              '/home/advit/aug30_exps/_videos_/vid_all_temps_count_4.mp4', 
              '/home/advit/aug30_exps/_videos_/vid_all_temps_count_5.mp4']

video_row3 = ['/home/advit/aug30_exps/_videos_/vid_all_temps_count_6.mp4', 
              '/home/advit/aug30_exps/_videos_/vid_all_temps_count_7.mp4', 
               '/home/advit/aug30_exps/_videos_/vid_all_temps_count_7.mp4']

video_row4 = ['/home/advit/aug30_exps/_videos_/vid_no_temps_count_2.mp4', 
              '/home/advit/aug30_exps/_videos_/vid_no_temps_count_3.mp4', 
              '/home/advit/aug30_exps/_videos_/vid_no_temps_count_4.mp4']

video_paths = [video_row1, video_row2, video_row3, video_row4]

import numpy as np 

# Read video properties from the first video
cap = cv2.VideoCapture(video_paths[0][0])
print("Got the first video!")


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))
total_frames = int(cap.get(7))

# Calculate dimensions for the grid layout
grid_rows = 4
grid_cols = 3
output_frame_width = frame_width * grid_cols
output_frame_height = frame_height * grid_rows

# Create VideoWriter for the combined video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the video codec
output_path = '/home/advit/aug30_exps/_videos_/output_combined_video_dummy.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (output_frame_width, output_frame_height))


print(f"frame_width, {frame_width}")
print(f"frame_width, {frame_height}")
print(f"fps, {fps}")

print(f"total_frames, {total_frames}")


# # Loop through frames in the videos
# for i in range(total_frames):
#     # Create an empty frame for the grid
#     combined_frame = np.zeros((output_frame_height, output_frame_width, 3), dtype=np.uint8)
    
    
#     # Initialize video index
#     video_index = 0
#     print(f"Frame: {i}")
#     for row in range(grid_rows):
#         for col in range(grid_cols):
#             if video_index < 12:
#                 # Read the current video frame
#                 print(f" - Video {video_index} path is: {video_paths[row][col]}")

#                 if video_paths[row][col]: 

#                     cap = cv2.VideoCapture(video_paths[row][col])
#                     ret, frame = cap.read()

#                     if ret:
#                         # Resize the frame to match the grid cell dimensions
#                         resized_frame = cv2.resize(frame, (frame_width, frame_height))
#                         y_start = row * frame_height
#                         y_end = y_start + frame_height
#                         x_start = col * frame_width
#                         x_end = x_start + frame_width

#                         # Place the resized frame in the grid cell
#                         combined_frame[y_start:y_end, x_start:x_end] = resized_frame

#                     cap.release()

#                 video_index += 1

#     out.write(combined_frame)

# # Release VideoWriter
# out.release()


cap_arr_r1 = [cv2.VideoCapture(path) for path in video_row1]
cap_arr_r2 = [cv2.VideoCapture(path) for path in video_row2]
cap_arr_r3 = [cv2.VideoCapture(path) for path in video_row3]
cap_arr_r4 = [cv2.VideoCapture(path) for path in video_row4] 


# Loop through frames in the videos
for _ in range(total_frames):
    r_1_1, f_1_1 = cap_arr_r1[0].read()
    r_1_2, f_1_2 = cap_arr_r1[1].read()
    r_1_3, f_1_3 = cap_arr_r1[2].read()

    r_2_1, f_2_1 = cap_arr_r2[0].read()
    r_2_2, f_2_2 = cap_arr_r2[1].read()
    r_2_3, f_2_3 = cap_arr_r2[2].read()

    r_3_1, f_3_1 = cap_arr_r3[0].read()
    r_3_2, f_3_2 = cap_arr_r3[1].read()

    r_4_1, f_4_1 = cap_arr_r4[0].read()
    r_4_2, f_4_2 = cap_arr_r4[1].read()
    r_4_3, f_4_3 = cap_arr_r4[2].read()

  
    # Stack frames vertically
    stacked_frame_col1 = cv2.vconcat([f_1_1, f_2_1, f_3_1, f_4_1])
    stacked_frame_col2 = cv2.vconcat([f_1_2, f_2_2, f_3_2, f_4_2])
    stacked_frame_col3 = cv2.vconcat([f_1_3, f_2_3, f_1_1, f_4_3])

    stacked_frame = cv2.hconcat([stacked_frame_col1, stacked_frame_col2, stacked_frame_col3])

    out.write(stacked_frame)

out.release() 
for cap in cap_arr_r1: 
    cap.release() 
for cap in cap_arr_r2: 
    cap.release() 
for cap in cap_arr_r3: 
    cap.release() 
for cap in cap_arr_r4: 
    cap.release() 



# # Read video properties from the first video
# cap1 = cv2.VideoCapture(video1_path)
# frame_width = int(cap1.get(3))
# frame_height = int(cap1.get(4))
# fps = int(cap1.get(5))
# total_frames = int(cap1.get(7))

# # Create VideoWriter for the stitched video
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the video codec
# output_path = 'output_stitched_video.mp4'
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height * 2))  # Doubled height

# # Read video properties from the second video
# cap2 = cv2.VideoCapture(video2_path)

# # Loop through frames in the videos
# for _ in range(total_frames):
#     ret1, frame1 = cap1.read()
#     ret2, frame2 = cap2.read()

#     if not ret1 or not ret2:
#         break

#     # Stack frames vertically
#     stacked_frame = cv2.vconcat([frame1, frame2])

#     out.write(stacked_frame)

# # Release VideoWriter and VideoCapture
# out.release()
# cap1.release()
# cap2.release()