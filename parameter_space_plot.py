from collections import OrderedDict
from collections import namedtuple
from itertools import product
import sys
import numpy as np
import matplotlib.pyplot as plt

# class RunBuilder():
#     @staticmethod
#     def get_runs(params):
#         Run = namedtuple('Run', params.keys())
#         runs = []
#         for v in product(*params.values()):
#             runs.append(Run(*v))
#         return runs


params = OrderedDict(
    Power = [0.01, 1, 10, 50],
    HTC = [1e-9, 1e-7, 1e-6, 2e-5],
    Die_xy = [4000, 10000, 20000, 30000],
    Die_z = [10, 100, 200],
    t_Diel = [1.0, 6.0, 10.0],
    t_Insulator = [0.01, 0.02, 0.05],
    K_Diel = [0.00138, 0.0138, 0.138]
)



num_cases = 1000
Power = np.random.randint(low = 1, high=5000, size=(num_cases, 1), dtype=int) * 10**-2
HTC = np.random.randint(low = 1, high=20000, size=(num_cases, 1), dtype=int) * 10**-9
Die_xy = np.random.randint(low = 4000, high=30000, size=(num_cases, 1), dtype=int)
Die_z = np.random.randint(low = 10, high=200, size=(num_cases, 1), dtype=int)
t_Diel = np.random.uniform(low=1.0, high=10.0, size=(num_cases, 1))
t_Insulator = np.random.uniform(low=1.0, high=5.0, size=(num_cases, 1)) * 10**-2
K_Diel = np.random.uniform(low=1.38, high=138.0, size=(num_cases, 1)) * 10**-3
params = np.hstack((Power, HTC, Die_xy, Die_z, t_Diel, t_Insulator, K_Diel))


Power = np.log(Power*10**2) * 0.1
HTC = np.log(HTC*10**9) * 0.1
Die_xy = (Die_xy - 4000.0) / (30000.0 - 4000.0)
Die_z = (Die_z - 10.0) / (200.0 - 10.0)
t_Diel = (t_Diel - 1.0) / (10.0 - 1.0)
t_Insulator = (t_Insulator - 0.01) / (0.05 - 0.01)
K_Diel = np.log(K_Diel*10**3) * 0.1

print("Power max is: ", params[:, 0].max())
print("Power min is: ", params[:, 0].min())
print("HTC max is: ", params[:, 1].max())
print("HTC min is: ", params[:, 1].min())
print("Die_xy max is: ", params[:, 2].max())
print("Die_xy min is: ", params[:, 2].min())
print("Die_z max is: ", params[:, 3].max())
print("Die_z min is: ", params[:, 3].min())
print("t_Diel max is: ", params[:, 4].max())
print("t_Diel min is: ", params[:, 4].min())
print("t_Insulator max is: ", params[:, 5].max())
print("t_Insulator min is: ", params[:, 5].min())
print("K_Diel max is: ", params[:, 6].max())
print("K_Diel min is: ", params[:, 6].min())
print("Parameter generation done")







# Create figure and subplot manually
# fig = plt.figure()
# host = fig.add_subplot(111)

# More versatile wrapper
fig, host = plt.subplots(figsize=(12, 5))  # (width, height) in inches
# (see https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.subplots.html)
# plt.subplots_adjust(left=0.05, bottom=0.1, right=0.9, top=0.9, wspace=None, hspace=None)
par1 = host.twinx()
par2 = host.twinx()
par3 = host.twinx()
par4 = host.twinx()
par5 = host.twinx()
par6 = host.twinx()
# par2.spines['right'].set_position(("axes", 1.05))

host.set_xlim(0, num_cases)
# host.set_ylim(0, 50) # Power
# par1.set_ylim(1e-9, 2e-5) # HTC
# par2.set_ylim(4000, 30000) # Die xy
# par3.set_ylim(10, 200) # Die z
# par4.set_ylim(1, 10) # t_diel
# par5.set_ylim(0.01, 0.05) # t_insl
# par6.set_ylim(0.00138, 0.138) # K_diel

# host.set_ylim(0, 1) # Power
# par1.set_ylim(0, 1) # HTC
# par2.set_ylim(0, 1) # Die xy
# par3.set_ylim(0, 1) # Die z
# par4.set_ylim(0, 1) # t_diel
# par5.set_ylim(0, 1) # t_insl
# par6.set_ylim(0, 1) # K_diel

# host.set_xlabel("Distance")
host.set_ylabel("Power")
par1.set_ylabel("HTC")
par2.set_ylabel("Die_xy")
par3.set_ylabel("Die_z")
par4.set_ylabel("t_diel")
par5.set_ylabel("t_insl")
par6.set_ylabel("K_diel")


color1 = plt.cm.jet(0)
color2 = plt.cm.jet(0.15)
color3 = plt.cm.jet(.3)
color4 = plt.cm.jet(.45)
color5 = plt.cm.jet(.6)
color6 = plt.cm.jet(.75)
color6 = plt.cm.jet(.9)

x_coord = np.arange(num_cases)
# Power, HTC, Die_xy, Die_z, t_Diel, t_Insulator, K_Diel
p1 = host.scatter(x_coord, Power, color=color1, label="Power")
p2 = par1.scatter(x_coord, HTC, color=color2, label="HTC")
p3 = par2.scatter(x_coord, Die_xy, color=color3, label="Die_xy")
p4 = par3.scatter(x_coord, Die_z, color=color4, label="Die_z")
p5 = par4.scatter(x_coord, t_Diel, color=color5, label="t_diel")
p6 = par5.scatter(x_coord, t_Insulator, color=color6, label="t_insl")
p7 = par6.scatter(x_coord, K_Diel, color=color6, label="K_diel")
# p1 = host.plot([0, 1, 2], [0, 1, 2], color=color1, label="Power")
# p2 = par1.plot([0, 1, 2], [0, 3, 2], color=color2, label="HTC")
# p3, = par2.plot([0, 1, 2], [50, 30, 15], color=color3, label="Die_xy")

lns = [p1, p2, p3, p4, p5, p6, p7]
host.legend(handles=lns, loc='best', bbox_to_anchor=(-0.6, 0.2, 0.5, 0.5))

# right, left, top, bottom
distance = 60
par2.spines['right'].set_position(('outward', 60))
par3.spines['right'].set_position(('outward', 60 + distance))
par4.spines['right'].set_position(('outward', 60 + distance*2))
par5.spines['right'].set_position(('outward', 60 + distance*3))
par6.spines['right'].set_position(('outward', 60 + distance*4))

# no x-ticks
par2.xaxis.set_ticks([])

# Sometimes handy, same for xaxis
# par2.yaxis.set_ticks_position('right')

# Move "Velocity"-axis to the left
# par2.spines['left'].set_position(('outward', 60))
# par2.spines['left'].set_visible(True)
# par2.yaxis.set_label_position('left')
# par2.yaxis.set_ticks_position('left')

# Adjust spacings w.r.t. figsize
fig.tight_layout()
# Alternatively: bbox_inches='tight' within the plt.savefig function
#                (overwrites figsize)

# Best for professional typesetting, e.g. LaTeX
plt.savefig("pyplot_multiple_y-axis.jpg")
# For raster graphics use the dpi argument. E.g. '[...].png", dpi=200)'


print('Done generating the parameter space')
raise




fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(c_scale_lst, TTSS_lst, color='b', marker='*')
plt.legend(['TTSS distribution'])
plt.xlabel('C_scale')
plt.ylabel('TTSS (s)')
plt.savefig(save_data_path +  f'TTSS distribution')
print('done generating cscale distribution')
header = 'power, HTC, Die_xy, Die_z, t_Diel, t_Insulator, K_Diel, C_scale, TTSS'

np.savetxt(save_data_path + 'extreme cases.txt', extreme_TTSS, fmt='%.3e', header = header)

import matplotlib.cm as cm
# colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y', 'purple']
markers = ['.', 'x', '^', '*', 'o', 'v', '<', '>', ',']
legend = ['Total power: 5e2--2e4', 'HTC (log scale): 1e-9--2e-5', 'Die_xy: 4000--30000',\
          'Die_z: 10--200', 't_Diel: 1--10', 't_Insulator: 1e-2--5e-2', 'K_Diel: 138e-5--138e-3',\
          'C_scale: 1e6--30e6']
colors = cm.rainbow(np.linspace(0, 1, len(legend)))
val = 0.  # this is the value where you want the data to appear on the y-axis.
for j in range(params_lst.shape[1]):
    # val += 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ar = params_lst[:, j] # just as an example array
    if j == 1:
        ax.plot(np.log10(ar), np.zeros_like(ar) + val, color = colors[j, :], marker =  markers[j])
    else:
        ax.plot(ar, np.zeros_like(ar) + val, color = colors[j, :], marker =  markers[j])
    plt.legend([legend[j]])
    plt.savefig(save_data_path + f'{j}')
    # plt.show()
print('done')

