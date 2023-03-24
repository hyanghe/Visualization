import numpy as np

cons = np.load('x_train_total.npy')
sols = np.load('y_train_total.npy')

total_num = cons.shape[0]

train_idx = np.random.choice(np.arange(total_num), int(total_num*0.8), replace=False)
test_idx = np.setxor1d(np.arange(total_num), train_idx)

x_train = cons[train_idx]
x_test = cons[test_idx]

y_train = sols[train_idx]
y_test = sols[test_idx]

np.save('x_train.npy', x_train)
np.save('x_test.npy', x_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
