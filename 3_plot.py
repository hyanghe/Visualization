import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# cons = np.load('x_train.npy')
# sols = np.load('y_train.npy')

cons = np.load('x_test.npy')
sols = np.load('y_test.npy')

idx = np.random.choice(sols.shape[0])
con = cons[idx]
sol = sols[idx]

fig = plt.figure()
ax = fig.add_subplot(121)
im1 = ax.imshow(sol, cmap='jet')
ax.set_title('solution')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')


ax = fig.add_subplot(122)
im1 = ax.imshow(con, cmap='jet')
ax.set_title('source')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')
plt.show()
# plt.savefig(f'{j}.jpg')
plt.close()
raise