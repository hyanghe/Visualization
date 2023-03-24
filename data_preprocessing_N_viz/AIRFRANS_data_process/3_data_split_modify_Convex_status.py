import numpy as np 
import os

# nut_l = np.load('nut_l.npy')
# Ux_l = np.load('Ux_l.npy')

condition = np.load('airfoil_interpolated.npy').astype(np.float32)
solution = np.load('Ux_l.npy').astype(np.float32)
params = []
with open('params.txt', 'r') as file:
	lines = file.readlines()
	for line in lines:
		line = line.strip("][\n").split(',')
		# import pdb; pdb.set_trace()
		# for s in line:
		# 	print(s)
		params.append([float(s) for s in line][:2])
		# print(params)
		# import pdb; pdb.set_trace()
params = np.asarray(params)
print(params.shape)
reynolds_max, reynolds_min = params[:, 0].max(), params[:, 0].min()
print('reynolds_max, reynolds_min:\n', reynolds_max, reynolds_min )

# print(params[:, 1].max(), params[:, 1].min())
aoa_max, aoa_min = params[:, 1].max(), params[:, 1].min()
print('aoa_max, aoa_min:\n', aoa_max, aoa_min)


params[:,0] = (params[:,0]-reynolds_min)/(reynolds_max - reynolds_min)
params[:,1] = (params[:,1]-aoa_min)/(aoa_max - aoa_min)
print('After scaling:\n')
print('reynolds_max, reynolds_min:\n', params[:,0].max(), params[:,0].min())
print('aoa_max, aoa_min:\n', params[:,1].max(), params[:,1].min())

# raise
# idx = np.load('ConvexStatus.npy')
# idx_exclude = [8, 10, 11, 15, 21, 29, 37, 40, 42, 49, 54, 65, 69, 72, 81, 83, 87, 89, 99, 106,
# 120, 121, 141, 142, 143, 145, 151, 166, 169, 182, 188, 210, 212, 217, 227, 235, 249, 251, 252,
# 256, 258, 269, 296, 300, 312, 313, 325, 330, 340, 357, 358, 366, 367, 371, 378, 379, 386, 395,
# 403, 405, 407, 408, 413, 472, 474, 477, 482, 489, 535, 536, 562, 570, 574, 583, 590, 623, 625,
# 626, 638, 652, 653, 694, 703, 708, 726, 727, 728, 729, 730, 732, 741, 747, 750, 752, 764, 769,
# 773, 778, 785, 791, 800, 801, 804, 809, 811, 813, 817, 821, 833, 844, 847, 848, 855, 856, 869,
# 878, 879, 883, 899, 906, 909, 916, 925, 926, 932, 933, 938, 944, 945, 949, 950, 956, 958, 960,
# 963, 965, 966, 972, 975, 977, 982, 984, 988, 990, 992]
# idx[idx_exclude] = False
# condition = condition[idx]
# solution = solution[idx]

train_num = int(condition.shape[0] * 0.9)
train_idx = np.random.choice(condition.shape[0], train_num, replace=False)

test_idx = np.setxor1d(np.arange(condition.shape[0]), train_idx)

idxs = [train_idx, test_idx]
folders = ['train', 'test']
for folder, idx in zip(folders, idxs):
	f = os.getcwd() + '/'+folder
	os.makedirs(f, mode=0o777, exist_ok=True)
	cond = condition[idx]
	sol = solution[idx]
	param = params[idx]

	np.save(f + '/' + 'condition', cond)
	np.save(f + '/' + 'solution', sol)
	np.save(f + '/' + 'params', param)
	print(f'{folder} condition shape: ', cond.shape)
	print(f'{folder} solution shape: ', sol.shape)
	print(f'{folder} params shape: ', param.shape)
	# raise