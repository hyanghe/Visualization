import numpy as np 


x_train_2000 = np.load('x_train_fix_scale2_5_5000cases.npy')
x_train_3000 = np.load('x_train_random_scale3_5_5000cases.npy')
y_train_2000 = np.load('y_train_fix_scale2_5_5000cases.npy')
y_train_3000 = np.load('y_train_random_scale3_5_5000cases.npy')

x_train = np.concatenate((x_train_2000, x_train_3000), axis=0)
y_train = np.concatenate((y_train_2000, y_train_3000), axis=0)
print('x_train, y_train: ', x_train.shape, y_train.shape)

np.save('x_train_total', x_train)
np.save('y_train_total', y_train)