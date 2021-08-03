import cv2
import os
import numpy as np

image_folder = '.\\figs\\'
image_folder_fluent = '.\\figs_fluent\\'
video_name = 'video_y_velocity_mlsolver_zoom_in.avi'
colorbar_width = 150

# image_folder = '.\\figs_mlsolver_temperature\\'
# image_folder_fluent = '.\\figs_fluent_temperature\\'
# video_name = 'video_temperature.avi'
# colorbar_width = 180


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
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video = cv2.VideoWriter(video_name,fourcc, 20.0, (height, width))

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (500, 45)
# org_fluent = (430, 400) ## Temperature
org_fluent = (415, 400) ## y Velocity
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

    img_zoom = cv2.imread(os.path.join(image_folder, image))
    img_zoom_rot = cv2.rotate(img_zoom, cv2.cv2.ROTATE_90_CLOCKWISE)

    crop_img_zoom = img_zoom_rot[430:530, 200:400, :]
    zero_pad_width = 50
    crop_img_zoom_resized_h_pad = cv2.resize(crop_img_zoom, (720-2*zero_pad_width,360-2*zero_pad_width), interpolation=cv2.INTER_AREA)
    BLUE = [255, 255, 255]
    crop_img_zoom_resized_h_pad = cv2.copyMakeBorder(crop_img_zoom_resized_h_pad,
                                                     zero_pad_width, zero_pad_width, zero_pad_width, zero_pad_width, cv2.BORDER_CONSTANT, value=BLUE)

    # zero_pad_width = 0
    # crop_img_zoom = img_zoom_rot[430:530, 200:400, :]
    # dim = (720-2*zero_pad_width, 360-2*zero_pad_width)
    # crop_img_zoom_resized = cv2.resize(crop_img_zoom, dim, interpolation=cv2.INTER_AREA)
    # vertical_padding = np.ones((zero_pad_width, 720-2*zero_pad_width, 3))
    # horizontal_padding = np.ones((360, zero_pad_width, 3))
    # crop_img_zoom_resized_v_pad = np.concatenate((vertical_padding, crop_img_zoom_resized, vertical_padding), axis=0)
    # crop_img_zoom_resized_h_pad = np.concatenate((horizontal_padding, crop_img_zoom_resized_v_pad, horizontal_padding), axis=1)


    img_combined = np.concatenate((crop_img, crop_img_zoom_resized_h_pad), axis=0)
    # img_combined = np.concatenate((crop_img, crop_img), axis=0)
    img_combined_colorbar = np.concatenate((scale_frame, img_combined), axis=1)

    # Using cv2.putText() method
    img_combined_colorbar = cv2.putText(img_combined_colorbar, 'ML solver',\
                                        org, font, fontScale, color, thickness, cv2.LINE_AA)
    img_combined_colorbar = cv2.putText(img_combined_colorbar, 'Zoom-in view',\
                                        org_fluent, font, fontScale, color, thickness, cv2.LINE_AA)

    # cv2.imwrite('scale_image_colorbar.jpg'+image, img_combined_colorbar)
    video.write(img_combined_colorbar)

cv2.destroyAllWindows()
video.release()