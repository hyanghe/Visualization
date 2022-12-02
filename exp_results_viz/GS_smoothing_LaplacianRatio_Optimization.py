import numpy as np
import matplotlib.pyplot as plt
# sigma = [0, 5, 10, 15]

# sigma = [0, 5, 10, 15]
# relative_l2 = [0.071, 0.07, 0.066, 0.061]
# mae = [0.012, 0.013, 0.012, 0.01]

sigma = [0, 5, 10, 15, 25, 35, 50, 75, 100]
ratio = [1, 0.49993374, 0.267227534, 0.18039394,
 0.1079054630303578, 0.07613491935487737, 0.05204766750648096,
  0.03312164757608464, 0.02356428712055511]
ratio = [100*i for i in ratio]
relative_l2 = [0.071, 0.07, 0.066, 0.061, 0.0618, 0.0546, 0.0573, 0.074, 0.283]
mae = [0.012, 0.013, 0.012, 0.010, 0.010, 0.0096, 0.010, 0.014, 0.043]

# relative_l2 = [0.071, 0.07, 0.066, 0.061]
# mae = [0.012, 0.013, 0.012, 0.01]
# y1 = [1, 0.5, 0.27, 0.18]
# y2 = []
font_size = 15

plt.rcParams.update({'font.size': font_size})

fig, ax1 = plt.subplots(figsize=(5, 3))

# matplotlib.rc('xtick', labelsize=20) 
# matplotlib.rc('ytick', labelsize=20) 
plt.suptitle('Error plot')
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.9, wspace=None, hspace=None)

ax2 = ax1.twinx()
ax1.plot(ratio, relative_l2, 'g*-')
ax1.plot(ratio[-4], relative_l2[-4], 'ro', markersize=6)
ax1.annotate(f'Optimum\nRatio: {ratio[-4]:.2f}%', xy=(ratio[-4], relative_l2[-4]*1.05), xytext=(25, 0.1), fontsize=12,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax2.plot(ratio, mae, 'b.-')


ax1.set_xlabel('Ratio of Laplacian (%)')
ax1.set_ylabel('relative l2', color='g')
ax2.set_ylabel('mae', color='b')
ax1.invert_xaxis()
plt.show()