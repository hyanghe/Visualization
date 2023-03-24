import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import imageio
import cv2
import os
import numpy as np
from collections import defaultdict
from scipy.interpolate import griddata as gd

# video_name = image_folder + 'video_10um.avi'
colorbar_width = 180
height, width = 1000, 1200 ## First is # col, second is # of row
# video = cv2.VideoWriter(video_name, 0, 1, (height, width)) # Rot 90

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (3000, 150) # coordiates start from top left corner, first is x, second is y
# fontScale
fontScale = 2
# Red color in BGR
color = (0, 0, 255)
# Line thickness of 2 px
thickness = 4


raw_data_f = './meshgraphnets/data/deforming_plate/raw_data.pkl'

# mlsolver_input = np.load('./meshgraphnets/data/deforming_plate/mlsolver_input.npy')
# print('mlsolver_input shape: ', mlsolver_input.shape)
# for num_time_step in range(mlsolver_input.shape[0]):
# 	for z_idx in range(8):
# 		plt.imshow(mlsolver_input[num_time_step, :,:,z_idx])
# 		plt.savefig(f'./meshgraphnets/figs/cross_section_{num_time_step}_{z_idx}.jpg')
# raise




with open(raw_data_f, 'rb') as file:
    myvar = pickle.load(file)
    print(myvar.keys())

out = {}
key = 'world_pos'
val = myvar[key]
print('val len is: ', len(val))
#98x400x1341x3
# node_nums = []
# for case in val:
# 	for time_step in case:
# 		node_nums.append(len(time_step))
# 		# print('node num is: ', len(time_step))
# 		# raise
# print('avg node is: ', sum(node_nums)/len(node_nums))
# raise
world_pos = []
prev_world_pos = []
target_world_pos = []
cells = []
mesh_pos = []
node_type = []
stress = []
for case, cell, mesh_p, node_t, stre in zip(val, myvar['cells'], myvar['mesh_pos'], myvar['node_type'], myvar['stress']):
	world_pos.append(case[1:-1])
	prev_world_pos.append(case[0:-2])
	target_world_pos.append(case[2:])
	cells.append(cell[1:-1])
	mesh_pos.append(mesh_p[1:-1])
	node_type.append(node_t[1:-1])
	stress.append(stre[1:-1])



count = 0
interpolated_node_type_3D_v = []
for i, (c, m, n, w, p, t, s) in enumerate(zip(cells, mesh_pos, node_type, world_pos, prev_world_pos, target_world_pos, stress)):
	print('length of m,n,w,p,t,s: ', len(m),len(n),len(w),len(p),len(t),len(s))

	with imageio.get_writer('./meshgraphnets/figs/' + f'movie_input_{i}.gif', mode='I') as writer:

		# for time_step in range(50, len(m), 100):
		for time_step in range(0, len(m), 1):
			# Separate the plate and the obstacle
			# Unique values in node types are: 0, 1 and 3, based on visualization,\
			#                                  0 is plate, 1 is obstacle, 3 is bc
			# n_type_uniq_v = np.unique(np.asarray(n))
			# print('unique node type: ', n_type_uniq_v)
			w_np = np.asarray(w) # w_np shape is:  (398, 1305, 3)
			t_np = np.asarray(t) # t_np shape is:  (398, 1305, 3)
			n_np = np.asarray(n) # n_np shape is:  (398, 1305, 1)
			m_np = np.asarray(m) # m_np shape is:  (398, 1305, 3)
			# print('w_np shape is: ', w_np.shape)
			# print('w_np[:, :, 0] min max: ', w_np[:, :, 0].min(), w_np[:, :, 0].max())
			# print('w_np[:, :, 1] min max: ', w_np[:, :, 1].min(), w_np[:, :, 1].max())
			# print('w_np[:, :, 2] min max: ', w_np[:, :, 2].min(), w_np[:, :, 2].max())
			# raise

			w_np_step = w_np[time_step]
			t_np_step = t_np[time_step]
			n_np_step = n_np[time_step]

			m_np_step = m_np[time_step] # m_np_step shape is:  (1305, 3)

			plate_type_value = 2
			p_idx_step = (n_np_step == 0).flatten()
			n_np_step[p_idx_step] = plate_type_value
			bc_idx_step = (n_np_step == 3).flatten()
			n_np_step[bc_idx_step] = plate_type_value
			obs_idx_step = (n_np_step == 1).flatten()
			print('obs_idx_step shape is: ', obs_idx_step.shape)
			plate_idx_step = ~obs_idx_step
			plate_tgt_pos_step = t_np_step[plate_idx_step]
			plate_w_pos_step = w_np_step[plate_idx_step]
			plate_m_pos_step = m_np_step[plate_idx_step]

############################  Data generation for ML Solver input ##############################
############################  Data generation for ML Solver input ##############################
############################  Data generation for ML Solver input ##############################
############################  Data generation for ML Solver input ##############################

			# build the map idx --> coord for the world pos (both plate and obs)
			print('n_np_step shape is: ', n_np_step.shape)
			n_np_step_dict = {}
			for k in range(n_np_step.shape[0]):
				n_np_step_dict[k] = n_np_step[k, :]

			m_np_step_dict = {}
			for k in range(m_np_step.shape[0]):
				m_np_step_dict[k] = m_np_step[k, :]
			# Establish the grid
			print('m_np_step shape is: ', m_np_step.shape)



			x_min_all, x_max_all = w_np[:, :, 0].min(), w_np[:, :, 0].max()
			y_min_all, y_max_all = w_np[:, :, 1].min(), w_np[:, :, 1].max()
			z_min_all, z_max_all = w_np[:, :, 2].min(), w_np[:, :, 2].max()
			print('x_min_all, x_max_all is: ', x_min_all, x_max_all)
			print('y_min_all, y_max_all is: ', y_min_all, y_max_all)
			print('z_min_all, z_max_all is: ', z_min_all, z_max_all)
			# x_pixels = 15*2
			# y_pixels = 30*2

			
			x_pixels_all = 16*1
			y_pixels_all = 32*1
			# z_pixels = int(y_pixels*(z_max-z_min)/(y_max-y_min))
			z_pixels_all = 2*1*4

			# x_pixels_all = 16*2
			# y_pixels_all = 32*2
			# z_pixels_all = 2*2*4

			xx_all = np.linspace(x_min_all, x_max_all, x_pixels_all)
			yy_all = np.linspace(y_min_all, y_max_all, y_pixels_all)
			print('z_pixels_all: ', z_pixels_all)
			zz_all = np.linspace(z_min_all, z_max_all, z_pixels_all) #z_pixels = 16
			xxx_all, yyy_all, zzz_all = np.meshgrid(xx_all, yy_all, zz_all)
			grid_coords_all = np.hstack((xxx_all.reshape(-1, 1), yyy_all.reshape(-1, 1), zzz_all.reshape(-1,1)))
			print('grid_coords_all shape is: ', grid_coords_all.shape)

			# build the map idx --> coord for the grids
			grid_coords_all_dict = {}
			for j in range(grid_coords_all.shape[0]):
				# print('grid_coords[i, :] shape is: ', grid_coords[i, :].shape)
				grid_coords_all_dict[j] = grid_coords_all[j, :]

			# build the map graph_idx --> grid_idx
			graph_grid_dist_all = defaultdict(lambda:1000)
			graph_all_2_grid_all = {}
			grid_all_2_graph_all = {}
			print('m_np_step shape is: ', m_np_step.shape)
			print('grid_coords_all shape is: ', grid_coords_all.shape)
			# for graph_idx, graph_corrd in plate_np_step_dict.items(): # Use plate
			for graph_idx_, graph_corrd_ in m_np_step_dict.items(): # Use mesh instead of plate
				for grid_idx_, grid_coord_ in grid_coords_all_dict.items():
					dist = np.linalg.norm(graph_corrd_ - grid_coord_)
					if dist < graph_grid_dist_all[graph_idx_]:
						graph_grid_dist_all[graph_idx_] = dist
						graph_all_2_grid_all[graph_idx_] = grid_idx_
						grid_all_2_graph_all[grid_idx_] = graph_idx_

			mappable_grid_idx_all = list(graph_all_2_grid_all.values())
			mappable_grid_coords_all = grid_coords_all[mappable_grid_idx_all]

			graph_nodes_idx_all = list(graph_all_2_grid_all.keys())
			grid_nodes_idx_all = list(graph_all_2_grid_all.values())
			print('graph_nodes_idx_all shape is: ', len(graph_nodes_idx_all))
			print('grid_nodes_idx_all shape is: ', len(grid_nodes_idx_all))
			# raise
			grid_all_node_type_step = np.zeros_like(grid_coords_all[:,0:1])
			print('grid_all_node_type_step shape is: ', grid_all_node_type_step.shape)


			missing_idx = []
			for grid_idx in range(grid_all_node_type_step.shape[0]):
				if grid_idx in grid_all_2_graph_all:
					# print('grid_idx: ', grid_idx)
					# print('grid_all_2_graph_all[grid_idx]: ', grid_all_2_graph_all[grid_idx])
					grid_all_node_type_step[grid_idx] = n_np_step_dict[grid_all_2_graph_all[grid_idx]]
				else:
					missing_idx.append(grid_idx)
					# grid_moving_delta_step[grid_idx] = 0
			# print('missing_idx is: ', missing_idx)
			
			print('m_np_step shape is: ', m_np_step.shape)
			print('m_np_step[:, 0] min max: ', m_np_step[:, 0].min(), m_np_step[:, 0].max())
			print('m_np_step[:, 1] min max: ', m_np_step[:, 1].min(), m_np_step[:, 1].max())
			print('m_np_step[:, 2] min max: ', m_np_step[:, 2].min(), m_np_step[:, 2].max())

			print('grid_coords_all shape is: ', grid_coords_all.shape)
			print('grid_coords_all[:, 0] min max: ', grid_coords_all[:, 0].min(), grid_coords_all[:, 0].max())
			print('grid_coords_all[:, 1] min max: ', grid_coords_all[:, 1].min(), grid_coords_all[:, 1].max())
			print('grid_coords_all[:, 2] min max: ', grid_coords_all[:, 2].min(), grid_coords_all[:, 2].max())

			void_x = np.random.uniform(grid_coords_all[:, 0].min(),\
									   grid_coords_all[:, 0].max(),
									   size=(1000,1))
			void_y = np.random.uniform(grid_coords_all[:, 1].min(),\
									   grid_coords_all[:, 1].max(),
									   size=(1000,1))
			void_z = np.random.uniform(m_np_step[:, 2].max(),\
									   grid_coords_all[:, 1].max(),
									   size=(1000,1))
			void_coords = np.concatenate((void_x, void_y, void_z), axis=1)
			void_values = np.zeros_like(void_z)
			
			w_np_step_N_void = np.concatenate((w_np_step, void_coords), axis=0)
			n_np_step_N_void = np.concatenate((n_np_step, void_values), axis=0)


########################## Grid node type interpolation ##########################################3
			interpolated_node_type = gd(w_np_step_N_void, n_np_step_N_void, grid_coords_all,\
			 method='nearest', fill_value=0)
			print('interpolated_node_type shape is: ', interpolated_node_type.shape)
			interpolated_node_type_3D = interpolated_node_type.reshape(y_pixels_all, x_pixels_all,\
				 z_pixels_all)
			interpolated_node_type_3D_v.append(interpolated_node_type_3D)
			# for z_idx in range(8):
			# 	plt.imshow(interpolated_node_type.reshape(y_pixels_all, x_pixels_all,\
			# 	 z_pixels_all)[:,:,z_idx])
			# 	plt.savefig(f'./meshgraphnets/figs/cross_section_{z_idx}_{time_step}.jpg')
			# raise
			# interpolated_node_type = gd(m_np_step, n_np_step, grid_coords_all,\
			#  method='nearest', fill_value=0)

			fig = plt.figure(figsize = (8, 10), dpi=100)
			plt.rcParams.update({'font.size': 20})
			plt.subplots_adjust(wspace=0.35, hspace = 0.25, left = 0.05, right=0.85, top=0.85, bottom=0.05)
			plt.suptitle(f'case: {i}, time step: {time_step}')
			ax = fig.add_subplot(1, 1, 1, projection='3d')
			im = ax.scatter(grid_coords_all[:,0], grid_coords_all[:,1],\
			 grid_coords_all[:,2], c=interpolated_node_type, cmap='jet', alpha=0.5)
			plt.colorbar(im)
			ax.title.set_text('node type\n (plate + obs + void)')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords_all[:,0],\
			 grid_coords_all[:,1], grid_coords_all[:,2])])  # equal aspect ratio
			plt.savefig(f'./meshgraphnets/figs/node_type_grids_{time_step}.jpg')
			# raise
########################## Grid node type interpolation ###########################################

########################## Graph node type ########################################################
			# fig = plt.figure(figsize = (8, 10), dpi=100)
			# plt.rcParams.update({'font.size': 20})
			# plt.subplots_adjust(wspace=0.35, hspace = 0.25, left = 0.05, right=0.85, top=0.85, bottom=0.05)
			# plt.suptitle(f'case: {i}, time step: {time_step}')
			# ax = fig.add_subplot(1, 1, 1, projection='3d')
			# im = ax.scatter(w_np_step[:,0], w_np_step[:,1],\
			#  w_np_step[:,2], c=n_np_step, cmap='jet', alpha=0.5)
			# plt.colorbar(im)
			# ax.title.set_text('node type\n (plate + obs)')
			# ax.set_box_aspect([np.ptp(i) for i in (w_np_step[:,0],\
			#  w_np_step[:,1], w_np_step[:,2])])  # equal aspect ratio
			# plt.savefig(f'./meshgraphnets/figs/node_type_{time_step}.jpg')
			# # plt.close()
########################## Graph node type ########################################################

############################  Data generation for ML Solver input ##############################
############################  Data generation for ML Solver input ##############################
############################  Data generation for ML Solver input ##############################
############################  Data generation for ML Solver input ##############################




			# fig_test_name = f'./meshgraphnets/figs/node_type(filter void)_{time_step}.jpg'
			# fig_test_name = f'./meshgraphnets/figs/node_type_{time_step}.jpg'
			fig_test_name = f'./meshgraphnets/figs/node_type_grids_{time_step}.jpg'
			image = fig_test_name
			img = cv2.imread(os.path.join(image))
			# converting BGR to RGB
			# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			writer.append_data(img)
		#             video.write(img)
			plt.close()

#######################################################################################################3
#######################################################################################################3
#######################################################################################################3
#######################################################################################################3
#######################################################################################################3
#######################################################################################################3
#######################################################################################################3
		interpolated_node_type_3D_v = np.asarray(interpolated_node_type_3D_v)
		np.save('./meshgraphnets/data/deforming_plate/mlsolver_input', interpolated_node_type_3D_v)
		count += 1
		if count >= 1:
			raise
raise


for i, (c, m, n, w, p, t, s) in enumerate(zip(cells, mesh_pos, node_type, world_pos, prev_world_pos, target_world_pos, stress)):
	
	# print('length of m,n,w,p,t,s: ', len(m),len(n),len(w),len(p),len(t),len(s))
	fig = plt.figure(figsize = (12, 10))
	plt.subplots_adjust(wspace=0.15, hspace = 0.1, left = 0.05, right=0.95, top=0.9, bottom=0.05)
	# plt.suptitle(f'bc is: {bc}, thickness is: {*thickness,},\n obs is: {*obstacle,},\n SSIM (solution, smoothed): {score:.2f} --> {new_ssim:.2f}')
	plt.suptitle(f'raw data')
	ax = fig.add_subplot(2, 2, 1, projection='3d')
	im = ax.scatter(m[:,0], m[:,1], m[:,2], c=m[:,2], cmap='jet')
	ax.title.set_text('mesh position')
	ax.set_box_aspect([np.ptp(i) for i in (m[:,0], m[:,1], m[:,2])])  # equal aspect ratio
	# cbaxes = fig.add_axes([0.22, 0.55, 0.02, 0.15])
	# plt.colorbar(im, cax = cbaxes)

	ax = fig.add_subplot(2, 2, 2, projection='3d')
	im = ax.scatter(w[:,0], w[:,1], w[:,2], c='r', cmap='jet')
	ax.title.set_text('world position')
	ax.set_box_aspect([np.ptp(i) for i in (w[:,0], w[:,1], w[:,2])])  # equal aspect ratio
	# cbaxes = fig.add_axes([0.65, 0.55, 0.02, 0.15])
	# plt.colorbar(im, cax = cbaxes)
	# plt.colorbar(im)
	ax = fig.add_subplot(2, 2, 3, projection='3d')
	im = ax.scatter(p[:,0], p[:,1], p[:,2], c='g', cmap='jet')
	ax.set_box_aspect([np.ptp(i) for i in (p[:,0], p[:,1], p[:,2])])  # equal aspect ratio
	ax.title.set_text('previous position')

	ax = fig.add_subplot(2, 2, 4, projection='3d')
	im = ax.scatter(t[:,0], t[:,1], t[:,2], c='b', cmap='jet')
	ax.set_box_aspect([np.ptp(i) for i in (t[:,0], t[:,1], t[:,2])])  # equal aspect ratio
	ax.title.set_text('target position')

	plt.savefig(f'./meshgraphnets/figs/{i}')
	plt.close()
	# raise