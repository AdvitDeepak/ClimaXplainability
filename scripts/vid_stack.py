import cv2

# Video paths
video1_path = 'video_pred.mp4'
video2_path = 'video_truth.mp4' 


# Read video properties from the first video
cap1 = cv2.VideoCapture(video1_path)
frame_width = int(cap1.get(3))
frame_height = int(cap1.get(4))
fps = int(cap1.get(5))
total_frames = int(cap1.get(7))

# Create VideoWriter for the stitched video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the video codec
output_path = 'output_stitched_video.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height * 2))  # Doubled height

# Read video properties from the second video
cap2 = cv2.VideoCapture(video2_path)

# Loop through frames in the videos
for _ in range(total_frames):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # Stack frames vertically
    stacked_frame = cv2.vconcat([frame1, frame2])

    out.write(stacked_frame)

# Release VideoWriter and VideoCapture
out.release()
cap1.release()
cap2.release()