import cv2
import os
import re 

STOP_AT = 133
pattern = r'_\d+_'

IN_DIR = "/home/advit/sep7_exps/all_vars/truth"
OUT_DIR ="/home/advit/sep7_exps/all_vars/video/truth.mp4"



def make_video(folder, out_path): 
    image_files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])  # Update the extension as needed
    print(f"Found {len(image_files)} files in {folder}")

    first_image_path = os.path.join(folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    frame_height, frame_width, channels = first_image.shape

    fps = 1  # Frames per second

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the video codec
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

    for image_file in image_files:

        # match = re.search(pattern, image_file)
        # if match:
        #     int_match = int(match.group(0)[1:-1]) 
        #     if int_match > STOP_AT: 
        #         print(f" STOPPING AT {int_match}. Creating video...")
        #         break 

        image_path = os.path.join(folder, image_file)
        image = cv2.imread(image_path)
        
        # Resize the image to match the video frame size
        image = cv2.resize(image, (frame_width, frame_height))
        
        out.write(image)  # Write the image to the video
        
    out.release()  # Release the video writer



make_video(IN_DIR, OUT_DIR)

# Walk through all subdirectories and print their names
# cnt = 1 
# for dirpath, dirnames, filenames in os.walk(IN_DIR):
#     for dirname in dirnames:
#         subdir = os.path.join(dirpath, dirname)


#         make_video(subdir, os.path.join(OUT_DIR, f"vid_no_temps_count_{cnt}.mp4"))
#         cnt += 1





# OUT_PRED = 'pred_imgs'
# OUT_TRUTH = 'truth_imgs'

# OUT_PRED_VID = 'video_pred.mp4'
# OUT_TRUTH_VID = 'video_truth.mp4'


# def make_video(folder, out_path): 
#     image_files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])  # Update the extension as needed
#     print(f"Found {len(image_files)} files in {folder}")

#     first_image_path = os.path.join(folder, image_files[0])
#     first_image = cv2.imread(first_image_path)
#     frame_height, frame_width, channels = first_image.shape

#     fps = 4  # Frames per second

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the video codec
#     out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

#     for image_file in image_files:

#         match = re.search(pattern, image_file)
#         if match:
#             int_match = int(match.group(0)[1:-1]) 
#             if int_match > STOP_AT: 
#                 print(f" STOPPING AT {int_match}. Creating video...")
#                 break 

#         image_path = os.path.join(folder, image_file)
#         image = cv2.imread(image_path)
        
#         # Resize the image to match the video frame size
#         image = cv2.resize(image, (frame_width, frame_height))
        
#         out.write(image)  # Write the image to the video
        
#     out.release()  # Release the video writer


# # # Actually making the videos 
# # make_video(OUT_PRED, OUT_PRED_VID) 
# # make_video(OUT_TRUTH, OUT_TRUTH_VID)