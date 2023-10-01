import imageio

# Video path
video_path = 'output_stitched_video.mp4'

# Output GIF path
gif_path = 'output_animation.gif'

# Read the video using imageio
video = imageio.get_reader(video_path)

# Get video metadata
fps = video.get_meta_data()['fps']

# Create a writer for the GIF
with imageio.get_writer(gif_path, mode='I', fps=fps) as gif_writer:
    for frame in video:
        gif_writer.append_data(frame)
