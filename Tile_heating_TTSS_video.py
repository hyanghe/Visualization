import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio

num_pixel = 500
num_step = 201
height, width = num_pixel, num_pixel
img = np.zeros((height, width, 3), np.uint8)

# img[:,:,0] = 255 # (B, G, R)
img[:,:,2] = 255
font = cv2.FONT_HERSHEY_SIMPLEX
org = (200, num_pixel - 75)
org_time = (50, 75)
org_temp = (150, 75)
fontScale = 1
# color = (255, 0, 0) # Blue color in BGR
color = (0, 255, 0)
thickness = 2 # Line thickness of 2 px
video_name = 'Tile_heating_TTSS.avi'
zero_pad_width = 10
# video = cv2.VideoWriter(video_name, 0, 1, (height+zero_pad_width*2, (width+zero_pad_width*2)*2))
Time = np.random.uniform(0, 1, 201)
Temperature = np.load('./data_ttss/Final_temp_ttss.npy')

# Make empty black image
image_temp = np.zeros((num_pixel, num_pixel, 3), np.uint8)

with imageio.get_writer('./movie.gif', mode='I') as writer:
    x_ranges = []
    y_ranges = []
    for i in range(1, num_step+1):
        globals()[f'new_img_{i}'] = np.zeros((height, width, 3), np.uint8)
        globals()[f'new_img_{i}'][:, :, 2] = 255


        step_size =  i / (num_step) * 255
        globals()[f'new_img_{i}'][int(0.45*num_pixel):int(0.55*num_pixel), int(0.45*num_pixel):int(0.55*num_pixel), 0] = step_size
        globals()[f'new_img_{i}'][int(0.45*num_pixel):int(0.55*num_pixel), int(0.45*num_pixel):int(0.55*num_pixel), 1] = 0
        globals()[f'new_img_{i}'][int(0.45*num_pixel):int(0.55*num_pixel), int(0.45*num_pixel):int(0.55*num_pixel), 2] = 0
        # globals()[f'new_img_{i}'] = img
        globals()[f'new_img_{i}'] = cv2.putText(globals()[f'new_img_{i}'], f'Time: {10/num_step*i:.3f}', org_time, font, fontScale, color, thickness, cv2.LINE_AA)

        img_title = cv2.putText(globals()[f'new_img_{i}'], 'Tile heating', org, font, fontScale, color, thickness, cv2.LINE_AA)


        # crop_img_zoom_resized_h_pad = cv2.resize(img_title, (720 - 2 * zero_pad_width, 360 - 2 * zero_pad_width),
        #                                          interpolation=cv2.INTER_AREA)
        BLUE = [255, 255, 255]
        img_title_boarder = cv2.copyMakeBorder(img_title,
                                                         zero_pad_width, zero_pad_width, zero_pad_width, zero_pad_width,
                                                         cv2.BORDER_CONSTANT, value=BLUE)




        globals()[f'new_image_temp_{i}'] = np.zeros((num_pixel, num_pixel, 3), np.uint8)
        # Make one pixel red
        x = 500 // num_step * i
        y = int(Temperature[i-1]/Temperature.max()*500)
        x_ranges.append(num_pixel - y )
        y_ranges.append(x)
        for x_range, y_range in zip(x_ranges, y_ranges):
            globals()[f'new_image_temp_{i}'][x_range:x_range+10, y_range:y_range+10] = [255, 0, 0]
        # image_temp[num_pixel - y :num_pixel - y+ 10, x:x + 10] = [255, 0, 0]
        globals()[f'new_image_temp_{i}'] = cv2.putText(globals()[f'new_image_temp_{i}'], 'Predicted curve', org, font, fontScale, color, thickness, cv2.LINE_AA)
        globals()[f'new_image_temp_{i}'] = cv2.putText(globals()[f'new_image_temp_{i}'], f'Temperature: {Temperature[i-1]:.3f}', org_temp, font, fontScale, color, thickness, cv2.LINE_AA)

        image_temp_boarder = cv2.copyMakeBorder(globals()[f'new_image_temp_{i}'],
                                                         zero_pad_width, zero_pad_width, zero_pad_width, zero_pad_width,
                                                         cv2.BORDER_CONSTANT, value=BLUE)
        # cv2.imwrite('image_temp.jpg', image_temp_boarder)
        # Save
        # cv2.imwrite("result.png", img_title_boarder)
        img_combined = np.concatenate((img_title_boarder, image_temp_boarder), axis=1)
        # cv2.imwrite('top view.png', img_title)
        cv2.waitKey(1)


        # video.write(img_combined)
        writer.append_data(img_combined)
# cv2.destroyAllWindows()
# video.release()

plt.show()


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