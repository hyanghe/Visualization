import matplotlib.pyplot as plt
import numpy as np
import pickle
import imageio
import cv2
# There are 1000 cases in total, there are 601 points in each case
# So in total there are 601000 points
## The input parameters are stored in u, which are:
## Power, HTC, Die_xy, Die_z, t_Diel, t_Insulator, K_Diel

## The coordiates of the points are in y

## The corresponding solutions (Temperatures) are in s

u_original = np.load("u.npy")
y_original = np.load("y.npy")
s = np.load("s.npy")


unit_size = 601

system_param_org = u_original[::unit_size]
def get_DeepONet_data(u_original, y_original):
    Power, HTC, Die_xy, Die_z, t_Diel, t_Insulator, K_Diel, y_train =\
    u_original[:, 0:1],u_original[:, 1:2],u_original[:, 2:3],\
    u_original[:, 3:4],u_original[:, 4:5],u_original[:, 5:6],\
    u_original[:, 6:7],y_original[:, 0:1]

    power_mag_n = np.log(Power*10**2) * 0.1
    HTC_n = np.log(HTC*10**9) * 0.1
    Die_xy_n = (Die_xy - 4000.0) / (30000.0 - 4000.0)
    Die_z_n = (Die_z - 10.0) / (200.0 - 10.0)
    t_Diel_n = (t_Diel - 1.0) / (10.0 - 1.0)
    t_Insulator_n = (t_Insulator - 0.01) / (0.05 - 0.01)
    K_Diel_n = np.log(K_Diel*10**3) * 0.1

    u_train = np.hstack((power_mag_n, HTC_n, Die_xy_n,\
     Die_z_n, t_Diel_n, t_Insulator_n, K_Diel_n))
    
    for i in range(y_train.shape[0]//unit_size):
        y_train[i*unit_size:(i+1)*unit_size, 0] =\
         y_train[i*unit_size:(i+1)*unit_size, 0] /\
          (Die_xy[i*unit_size:(i+1)*unit_size, 0][0] / 2.0)
    return u_train, y_train

u, y = get_DeepONet_data(u_original, y_original)


Temp_scaling_maxT = 150
Temp_scaling_minT = 120

## The input parameters for each of the 1000 cases are stored in params_test
params_test = u[::unit_size, :]


# Load the sklearn models.
filename_max = 'model_max.sav'
interpreter_maxT = pickle.load(open(filename_max, 'rb'))

filename_min = 'model_min.sav'
interpreter_minT = pickle.load(open(filename_min, 'rb'))

file_name_DeepONet = 'model.sav'
interpreter_DeepONet = pickle.load(open(file_name_DeepONet, 'rb'))

num_pixel = 500
font = cv2.FONT_HERSHEY_SIMPLEX
# org = (40, num_pixel - 75)
legend_space = 50
left_space = 50
org_Power = (left_space, 75)
org_HTC = (left_space, 75 + legend_space)
org_Die_xy = (left_space, 75 + legend_space*2)
org_Die_z = (left_space, 75 + legend_space*3)
org_t_Diel = (left_space, 75 + legend_space*4)
org_t_Insulator = (left_space, 75 + legend_space*5)
org_K_Diel = (left_space, 75 + legend_space*6)

# org_temp = (150, 75)
fontScale = 1
# color = (255, 0, 0) # Blue color in BGR
color = (0, 255, 0)
thickness = 2 # Line thickness of 2 px


test_num = 1000
with imageio.get_writer('./movie.gif', mode='I') as writer:
    for k in range(test_num):
        # case_num = np.random.choice(u.shape[0] // unit_size)
        img = np.zeros((num_pixel, num_pixel, 3), np.uint8)

        i = k
        globals()[f'img{i}'] = np.zeros((num_pixel, num_pixel, 3), np.uint8)
        case_num = k
        u_tr = params_test[i:i + 1, :]

        output_data_maxT = interpreter_maxT.predict(u_tr)
        pred_maxT = output_data_maxT

        output_data_minT = interpreter_minT.predict(u_tr)
        pred_minT = output_data_minT

        u_DeepONet = u[case_num * unit_size:(case_num + 1) * unit_size, :]
        y_DeepONet = y[case_num * unit_size:(case_num + 1) * unit_size, :]

        X_DeepONet = np.concatenate((u_DeepONet, y_DeepONet), axis=1)
        pred = interpreter_DeepONet.predict(X_DeepONet)

        Final_temp = pred * (pred_maxT * Temp_scaling_maxT - pred_minT * Temp_scaling_minT) + pred_minT * Temp_scaling_minT
        Final_temp = Final_temp[:, None]
        # Final_temp = (Final_temp - Final_temp.min()) / (Final_temp.max() - Final_temp.min())
        y_DeepONet = y_DeepONet*50*num_pixel
        Final_temp = Final_temp - 100
        y_DeepONet = y_DeepONet[:400, :] + 10
        Final_temp = num_pixel - Final_temp[:400, :]*15
        curve = np.column_stack((y_DeepONet.astype(np.int32), Final_temp.astype(np.int32)))
        curve_1 = cv2.polylines(globals()[f'img{i}'], [curve], False, (0, 255, 255), thickness=2)

        params_test
        globals()[f'img{i}'] = cv2.putText(globals()[f'img{i}'], f'Power: {system_param_org[i][0]:.3f} mW',\
                                           org_Power, font, fontScale, color, thickness, cv2.LINE_AA)
        # t_Diel, t_Insulator, K_Diel
        globals()[f'img{i}'] = cv2.putText(globals()[f'img{i}'], f'HTC: {system_param_org[i][1]:.2e} mW/um^2K', \
                                           org_HTC, font, fontScale, color, thickness, cv2.LINE_AA)
        globals()[f'img{i}'] = cv2.putText(globals()[f'img{i}'], f'Die_xy: {system_param_org[i][2]:.1f} um', \
                                           org_Die_xy, font, fontScale, color, thickness, cv2.LINE_AA)
        globals()[f'img{i}'] = cv2.putText(globals()[f'img{i}'], f'Die_z: {system_param_org[i][3]:.1f} um', \
                                           org_Die_z, font, fontScale, color, thickness, cv2.LINE_AA)
        globals()[f'img{i}'] = cv2.putText(globals()[f'img{i}'], f't_Diel: {system_param_org[i][4]:.2f} um', \
                                           org_t_Diel, font, fontScale, color, thickness, cv2.LINE_AA)
        globals()[f'img{i}'] = cv2.putText(globals()[f'img{i}'], f't_Insulator: {system_param_org[i][5]:.2f} um', \
                                           org_t_Insulator, font, fontScale, color, thickness, cv2.LINE_AA)
        globals()[f'img{i}'] = cv2.putText(globals()[f'img{i}'], f'K_Diel: {system_param_org[i][6]:.2e} mW/umK', \
                                           org_K_Diel, font, fontScale, color, thickness, cv2.LINE_AA)
        # draw_points = (np.asarray([y_DeepONet, Final_temp]).T).astype(np.int32)  # needs to be int32 and transposed
        # draw_points = draw_points.reshape((unit_size, 2))
        # curve = cv2.polylines(img, [draw_points], False, (255, 255, 255))  #
        cv2.imwrite('curve.png', curve_1)
        print('The max temperature is: ', Final_temp.max())

        writer.append_data(curve_1)