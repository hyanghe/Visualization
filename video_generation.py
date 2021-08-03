import cv2
import os
import numpy as np

# image_folder = '.\\figs\\'
# image_folder_fluent = '.\\figs_fluent\\'
# video_name = 'video_y_velocity.avi'
# colorbar_width = 150

image_folder = '.\\figs_mlsolver_temperature\\'
image_folder_fluent = '.\\figs_fluent_temperature\\'
video_name = 'video_temperature.avi'
colorbar_width = 180

# images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = [f"{i+1}.png" for i in range(50)]

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# scale_frame = cv2.imread(os.path.join(image_folder, images[7]))[:, :150, :] ## y_velocity
scale_frame = cv2.imread(os.path.join(image_folder, images[7]))[:, :colorbar_width, :] ## temperature
# cv2.imwrite('scale_image.jpg', scale_frame)

# video = cv2.VideoWriter(video_name, 0, 1, (width, height)) # Original
# height, width = 360, 720
height, width = 720+colorbar_width, 360*2 ## First is # col, second is # of row
video = cv2.VideoWriter(video_name, 0, 1, (height, width)) # Rot 90

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (500, 45)
org_fluent = (500, 408)
# fontScale
fontScale = 1
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

for image in images:
    img = cv2.imread(os.path.join(image_folder, image))
    img_rot = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    crop_img = img_rot[300:660, :, :]

    img_fluent = cv2.imread(os.path.join(image_folder_fluent, image))
    img_fluent_rot = cv2.rotate(img_fluent, cv2.cv2.ROTATE_90_CLOCKWISE)
    crop_img_fluent = img_fluent_rot[300:660, :, :]
    # cv2.imshow("cropped", crop_img)
    # image = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)

    img_combined = np.concatenate((crop_img, crop_img_fluent), axis=0)
    img_combined_colorbar = np.concatenate((scale_frame, img_combined), axis=1)

    # Using cv2.putText() method
    img_combined_colorbar = cv2.putText(img_combined_colorbar, 'ML solver',\
                                        org, font, fontScale, color, thickness, cv2.LINE_AA)
    img_combined_colorbar = cv2.putText(img_combined_colorbar, 'Fluent',\
                                        org_fluent, font, fontScale, color, thickness, cv2.LINE_AA)

    # cv2.imwrite('scale_image_colorbar.jpg', img_combined_colorbar)
    video.write(img_combined_colorbar)

cv2.destroyAllWindows()
video.release()