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


# cells = myvar['cells'][:][1:-1]
# mesh_pos = myvar['mesh_pos'][:][1:-1]
# node_type = myvar['node_type'][:][1:-1]
# world_pos = myvar['world_pos']
# prev_world_pos = 
# target_world_pos = 
# stress = myvar['stress'][:][1:-1]

count = 0
for i, (c, m, n, w, p, t, s) in enumerate(zip(cells, mesh_pos, node_type, world_pos, prev_world_pos, target_world_pos, stress)):
	
	print('length of m,n,w,p,t,s: ', len(m),len(n),len(w),len(p),len(t),len(s))
	# obs_idx_mesh = (np.asarray(n)[i] == 1).flatten()
	# # print('obs_idx_mesh shape is: ', obs_idx_mesh.shape)
	# plate_idx_mesh = ~obs_idx_mesh
	# # print('m[0][plate_idx_mesh] shape is: ', m[0][plate_idx_mesh].shape)
	# x_mn, x_mx = m[i][plate_idx_mesh][:,0].min(), m[i][plate_idx_mesh][:,0].max()
	# y_mn, y_mx = m[i][plate_idx_mesh][:,1].min(), m[i][plate_idx_mesh][:,1].max()
	# z_mn, z_mx = m[i][plate_idx_mesh][:,2].min(), m[i][plate_idx_mesh][:,2].max()
	# # print('z_mn, z_mx: ', z_mn, z_mx)
	# # print('m[i] len is: ', len(m[i][plate_idx_mesh]))
	# plate_idx_mesh_top = m[i][plate_idx_mesh][:,-1] == z_mn
	# # print('len(plate_idx_mesh_top) is: ', sum(plate_idx_mesh_top))
	# # raise
	# x_g, y_g = np.linspace(x_mn, x_mx, 15), np.linspace(y_mn, y_mx, 30)
	# xx_g, yy_g = np.meshgrid(x_g, y_g)
	# m_interp = gd(m[i][plate_idx_mesh][plate_idx_mesh_top][:,:2], np.ones((m[i][plate_idx_mesh][plate_idx_mesh_top].shape[0])),\
	# 											np.hstack((xx_g.reshape(-1, 1),\
	# 													yy_g.reshape(-1, 1))), 
	# 											method='nearest', fill_value=0)
	# print('np.var(m_interp) is: ', np.var(m_interp))
	# if np.var(m_interp) != 0.0:
	# 	continue
	# else:
	# 	print(f'case {i} included')
	# 	continue
		


	with imageio.get_writer('./meshgraphnets/figs/' + f'movie_{i}.gif', mode='I') as writer:

		for time_step in range(50, len(m), 50):
			# Separate the plate and the obstacle
			# Unique values in node types are: 0, 1 and 3, based on visualization,\
			#                                  0 is plate, 1 is obstacle, 3 is bc
			# n_type_uniq_v = np.unique(np.asarray(n))
			# print('unique node type: ', n_type_uniq_v)
			w_np = np.asarray(w) # w_np shape is:  (398, 1305, 3)
			t_np = np.asarray(t) # t_np shape is:  (398, 1305, 3)
			n_np = np.asarray(n) # n_np shape is:  (398, 1305, 1)
			m_np = np.asarray(m) # m_np shape is:  (398, 1305, 3)

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
			x_min_all, x_max_all = m_np_step[:,0].min(), m_np_step[:,0].max()
			y_min_all, y_max_all = m_np_step[:,1].min(), m_np_step[:,1].max()
			z_min_all, z_max_all = m_np_step[:,2].min(), m_np_step[:,2].max()
			print('x_min_all, x_max_all is: ', x_min_all, x_max_all)
			print('y_min_all, y_max_all is: ', y_min_all, y_max_all)
			print('z_min_all, z_max_all is: ', z_min_all, z_max_all)
			# x_pixels = 15*2
			# y_pixels = 30*2
			x_pixels_all = 16*1
			y_pixels_all = 32*1
			# z_pixels = int(y_pixels*(z_max-z_min)/(y_max-y_min))
			z_pixels_all = 2*1*4

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
			
			# raise
			interpolated_node_type = gd(m_np_step, n_np_step, grid_coords_all,\
			 method='nearest', fill_value=0)

			# fig = plt.figure(figsize = (8, 10), dpi=100)
			# plt.rcParams.update({'font.size': 20})
			# plt.subplots_adjust(wspace=0.35, hspace = 0.25, left = 0.05, right=0.85, top=0.85, bottom=0.05)
			# plt.suptitle(f'case: {i}, time step: {time_step}')
			# ax = fig.add_subplot(1, 1, 1, projection='3d')
			# im = ax.scatter(m_np_step[:,0], m_np_step[:,1],\
			#  m_np_step[:,2], c=n_np_step, cmap='jet', alpha=0.5)
			# plt.colorbar(im)
			# ax.title.set_text('node type\n (plate + obs)')
			# ax.set_box_aspect([np.ptp(i) for i in (grid_coords_all[:,0],\
			#  grid_coords_all[:,1], grid_coords_all[:,2])])  # equal aspect ratio
			# plt.savefig(f'./meshgraphnets/figs/node_type')
			# raise
			fig = plt.figure(figsize = (8, 10), dpi=100)
			plt.rcParams.update({'font.size': 20})
			plt.subplots_adjust(wspace=0.35, hspace = 0.25, left = 0.05, right=0.85, top=0.85, bottom=0.05)
			plt.suptitle(f'case: {i}, time step: {time_step}')
			ax = fig.add_subplot(1, 1, 1, projection='3d')
			im = ax.scatter(grid_coords_all[:,0], grid_coords_all[:,1],\
			 grid_coords_all[:,2], c=interpolated_node_type, cmap='jet', alpha=0.5)
			plt.colorbar(im)
			ax.title.set_text('node type\n (plate + obs)')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords_all[:,0],\
			 grid_coords_all[:,1], grid_coords_all[:,2])])  # equal aspect ratio
			plt.savefig(f'./meshgraphnets/figs/node_type')
			plt.close()

			# print('grid_coords_all[:,2] < 0 is: ', grid_coords_all[:,2] < 0, len(grid_coords_all[:,2]<0))
			# print('interpolated_node_type != 1 is: ',\
			#  interpolated_node_type.flatten() != 1, len(interpolated_node_type.flatten() != 1))
			# raise
			criterion_1 = grid_coords_all[:,2] < 0
			criterion_2 = interpolated_node_type.flatten() != 1
			void_idx = criterion_1 & criterion_2
			# print('void_idx: ', void_idx)
			# raise
			interpolated_node_type[void_idx] = 0
			fig = plt.figure(figsize = (8, 10), dpi=100)
			plt.rcParams.update({'font.size': 20})
			plt.subplots_adjust(wspace=0.35, hspace = 0.25, left = 0.05, right=0.85, top=0.85, bottom=0.05)
			plt.suptitle(f'case: {i}, time step: {time_step}')
			ax = fig.add_subplot(1, 1, 1, projection='3d')
			im = ax.scatter(grid_coords_all[:,0], grid_coords_all[:,1],\
			 grid_coords_all[:,2], c=interpolated_node_type, cmap='jet', alpha=0.5)
			# plt.colorbar(im)
			ax.title.set_text('node type\n (plate + obs)')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords_all[:,0],\
			 grid_coords_all[:,1], grid_coords_all[:,2])])  # equal aspect ratio
			cbaxes = fig.add_axes([0.85, 0.25, 0.02, 0.15])
			plt.colorbar(im, cax = cbaxes)
			plt.savefig(f'./meshgraphnets/figs/node_type(filter void)')
			plt.close()	
			raise


			interpolated_grid_all_node_type_step = grid_all_node_type_step.copy()
			if len(missing_idx) is not 0:
				missing_idx = np.asarray(missing_idx)
				missing_pts = grid_coords_all[missing_idx]

				interpolated_node_type = gd(m_np_step, n_np_step, missing_pts, method='nearest')
				interpolated_grid_all_node_type_step[missing_idx] = interpolated_node_type
				print('interpolated_node_type: ', interpolated_node_type.shape)

			# print('mappable_grid_idx: ', mappable_grid_idx)
			# raise
			## Plot the graph and grids
			fig = plt.figure(figsize = (8, 10), dpi=100)
			plt.rcParams.update({'font.size': 20})
			plt.subplots_adjust(wspace=0.35, hspace = 0.25, left = 0.05, right=0.85, top=0.85, bottom=0.05)
			# plt.suptitle(f'bc is: {bc}, thickness is: {*thickness,},\n obs is: {*obstacle,},\n SSIM (solution, smoothed): {score:.2f} --> {new_ssim:.2f}')
			plt.suptitle(f'case: {i}, time step: {time_step}')
			ax = fig.add_subplot(1, 1, 1, projection='3d')
			im = ax.scatter(grid_coords_all[:,0], grid_coords_all[:,1],\
			 grid_coords_all[:,2], c=interpolated_grid_all_node_type_step, cmap='jet', alpha=0.5)
			ax.title.set_text('node type\n (plate + obs)')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords_all[:,0],\
			 grid_coords_all[:,1], grid_coords_all[:,2])])  # equal aspect ratio
			# cbaxes = fig.add_axes([0.22, 0.55, 0.02, 0.15])
			# plt.colorbar(im, cax = cbaxes)

			plt.savefig(f'./meshgraphnets/figs/case_{i}_time step_{time_step}_data_interpolation_node_type')
			plt.close()
			raise
############################  Data generation for ML Solver input ##############################
############################  Data generation for ML Solver input ##############################
############################  Data generation for ML Solver input ##############################
############################  Data generation for ML Solver input ##############################



############################  Data generation for ML Solver output ##############################
############################  Data generation for ML Solver output ##############################
############################  Data generation for ML Solver output ##############################
############################  Data generation for ML Solver output ##############################
			plate_moving_delta_step = plate_tgt_pos_step - plate_w_pos_step
			print('plate_tgt_pos_step is: ', plate_tgt_pos_step.shape)

			# build the map idx --> coord for the world pos (only plate, use mesh)
			plate_mesh_np_step_dict = {}
			for k in range(plate_m_pos_step.shape[0]):
				plate_mesh_np_step_dict[k] = plate_m_pos_step[k, :]
			print('plate_m_pos_step shape is: ', plate_m_pos_step.shape)


			# Establish the grid
			print('plate_m_pos_step shape is: ', plate_m_pos_step.shape)
			x_min, x_max = plate_m_pos_step[:,0].min(), plate_m_pos_step[:,0].max()
			y_min, y_max = plate_m_pos_step[:,1].min(), plate_m_pos_step[:,1].max()
			z_min, z_max = plate_m_pos_step[:,2].min(), plate_m_pos_step[:,2].max()
			print('x_min, x_max is: ', x_min, x_max)
			print('y_min, y_max is: ', y_min, y_max)
			print('z_min, z_max is: ', z_min, z_max)
			# raise
			# x_pixels = 15*2
			# y_pixels = 30*2
			x_pixels = 16*2
			y_pixels = 32*2
			# z_pixels = int(y_pixels*(z_max-z_min)/(y_max-y_min))
			z_pixels = 2*2

			xx = np.linspace(x_min, x_max, x_pixels)
			yy = np.linspace(y_min, y_max, y_pixels)
			print('z_pixels: ', z_pixels)
			zz = np.linspace(z_min, z_max, z_pixels) #z_pixels = 16
			xxx, yyy, zzz = np.meshgrid(xx, yy, zz)
			grid_coords = np.hstack((xxx.reshape(-1, 1), yyy.reshape(-1, 1), zzz.reshape(-1,1)))
			print('grid_coords shape is: ', grid_coords.shape)
			# raise

			# build the map idx --> coord for the grids
			grid_coords_dict = {}
			for j in range(grid_coords.shape[0]):
				# print('grid_coords[i, :] shape is: ', grid_coords[i, :].shape)
				grid_coords_dict[j] = grid_coords[j, :]

			# build the map graph_idx --> grid_idx
			graph_grid_dist = defaultdict(lambda:1000)
			graph_2_grid = {}
			grid_2_graph = {}
			# raise
			print('plate_w_pos_step shape is: ', plate_w_pos_step.shape)
			print('grid_coords shape is: ', grid_coords.shape)
			# raise
			# for graph_idx, graph_corrd in plate_np_step_dict.items(): # Use plate
			for graph_idx, graph_corrd in plate_mesh_np_step_dict.items(): # Use mesh instead of plate
				for grid_idx, grid_coord in grid_coords_dict.items():
					dist = np.linalg.norm(graph_corrd - grid_coord)
					if dist < graph_grid_dist[graph_idx]:
						graph_grid_dist[graph_idx] = dist
						graph_2_grid[graph_idx] = grid_idx
						grid_2_graph[grid_idx] = graph_idx

			mappable_grid_idx = list(graph_2_grid.values())
			mappable_grid_coords = grid_coords[mappable_grid_idx]

			print('plate_moving_delta_step shape is: ', plate_moving_delta_step.shape)
			graph_nodes_idx = list(graph_2_grid.keys())
			grid_nodes_idx = list(graph_2_grid.values())
			print('graph_nodes_idx shape is: ', len(graph_nodes_idx))
			print('grid_nodes_idx shape is: ', len(grid_nodes_idx))
			grid_moving_delta_step = np.zeros_like(grid_coords)
			print('grid_moving_delta_step shape is: ', grid_moving_delta_step.shape)
			# print('plate_moving_delta_step[grid_2_graph[598]]: ', plate_moving_delta_step[grid_2_graph[598]])


			missing_idx = []
			for grid_idx in range(grid_moving_delta_step.shape[0]):
				if grid_idx in grid_2_graph:
					grid_moving_delta_step[grid_idx] = plate_moving_delta_step[grid_2_graph[grid_idx]]
				else:
					missing_idx.append(grid_idx)
					# grid_moving_delta_step[grid_idx] = 0
			print('missing_idx is: ', missing_idx)
			

			moving_delta_min = (t_np - w_np).min()
			moving_delta_max = (t_np - w_np).max()

###################### Scale the delta by all delta min and max ###################### 
			# grid_moving_delta_step = (grid_moving_delta_step - moving_delta_min)/(moving_delta_max - moving_delta_min)
###################### Scale the delta by all delta min and max ###################### 

			# raise
			interpolated_grid_moving_delta_step = grid_moving_delta_step.copy()
			if len(missing_idx) is not 0:
				missing_idx = np.asarray(missing_idx)
				missing_pts = grid_coords[missing_idx]

				interpolated_x_translation = gd(plate_w_pos_step, plate_moving_delta_step[:,0], missing_pts, method='linear')
				print('interpolated_x_translation shape is: ', interpolated_x_translation.shape)
				interpolated_grid_moving_delta_step[missing_idx, 0] = interpolated_x_translation

				interpolated_y_translation = gd(plate_w_pos_step, plate_moving_delta_step[:,1], missing_pts, method='nearest')
				interpolated_grid_moving_delta_step[missing_idx, 1] = interpolated_y_translation

				interpolated_z_translation = gd(plate_w_pos_step, plate_moving_delta_step[:,2], missing_pts, method='nearest')
				interpolated_grid_moving_delta_step[missing_idx, 2] = interpolated_z_translation
				# raise

			# print('mappable_grid_idx: ', mappable_grid_idx)
			# raise
			## Plot the graph and grids
			fig = plt.figure(figsize = (18, 20), dpi=100)
			plt.rcParams.update({'font.size': 20})
			plt.subplots_adjust(wspace=0.35, hspace = 0.25, left = 0.05, right=0.85, top=0.85, bottom=0.05)
			# plt.suptitle(f'bc is: {bc}, thickness is: {*thickness,},\n obs is: {*obstacle,},\n SSIM (solution, smoothed): {score:.2f} --> {new_ssim:.2f}')
			plt.suptitle(f'case: {i}, time step: {time_step}')
			ax = fig.add_subplot(4, 3, 1, projection='3d')
			im = ax.scatter(plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2], c='r', cmap='jet')
			ax.title.set_text('graph nodes\n (plate only)')
			ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
			# cbaxes = fig.add_axes([0.22, 0.55, 0.02, 0.15])
			# plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 3, 2, projection='3d')
			im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c='g', cmap='jet')
			ax.title.set_text('grid nodes')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			# cbaxes = fig.add_axes([0.65, 0.55, 0.02, 0.15])
			# plt.colorbar(im, cax = cbaxes)
			# plt.colorbar(im)

			ax = fig.add_subplot(4, 3, 3, projection='3d')
			im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c='g', cmap='jet')
			im = ax.scatter(mappable_grid_coords[:,0], mappable_grid_coords[:,1],\
			 mappable_grid_coords[:,2], c='r', cmap='jet')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			ax.legend(['Grids', 'Mapping'], loc = 'lower right', bbox_to_anchor=(1.7, 0.1))
			ax.title.set_text('grid nodes \n and mapping')

			ax = fig.add_subplot(4, 3, 4, projection='3d')
			im = ax.scatter(plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2], c=plate_moving_delta_step[:, 0], cmap='jet')
			ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
			ax.title.set_text('graph node \n x translation')
			cbaxes = fig.add_axes([0.28, 0.49, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 3, 5, projection='3d')
			im = ax.scatter(plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2], c=plate_moving_delta_step[:, 1], cmap='jet')
			ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
			ax.title.set_text('graph node \n y translation')
			cbaxes = fig.add_axes([0.56, 0.49, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 3, 6, projection='3d')
			im = ax.scatter(plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2], c=plate_moving_delta_step[:, 2], cmap='jet')
			ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
			ax.title.set_text('graph node \n z translation')
			cbaxes = fig.add_axes([0.84, 0.49, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 3, 7, projection='3d')
			im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c=grid_moving_delta_step[:, 0], cmap='jet')
			ax.title.set_text('grid nodes \n x translation')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			cbaxes = fig.add_axes([0.28, 0.275, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 3, 8, projection='3d')
			im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c=grid_moving_delta_step[:, 1], cmap='jet')
			ax.title.set_text('grid nodes \n y translation')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			cbaxes = fig.add_axes([0.56, 0.275, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)


			ax = fig.add_subplot(4, 3, 9, projection='3d')
			im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c=grid_moving_delta_step[:, 2], cmap='jet')
			ax.title.set_text('grid nodes \n z translation')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			cbaxes = fig.add_axes([0.84, 0.275, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 3, 10, projection='3d')
			im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c=interpolated_grid_moving_delta_step[:, 0], cmap='jet')
			ax.title.set_text('interpolated \n x translation')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			cbaxes = fig.add_axes([0.28, 0.05, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 3, 11, projection='3d')
			im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c=interpolated_grid_moving_delta_step[:, 1], cmap='jet')
			ax.title.set_text('interpolated \n y translation')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			cbaxes = fig.add_axes([0.56, 0.05, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 3, 12, projection='3d')
			im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c=interpolated_grid_moving_delta_step[:, 2], cmap='jet')
			ax.title.set_text('interpolated \n z translation')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			cbaxes = fig.add_axes([0.84, 0.05, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			plt.savefig(f'./meshgraphnets/figs/case_{i}_time step_{time_step}_data_interpolation')
			plt.close()
			# raise
			# print('w_np shape is: ', w_np.shape)
			# print('t_np shape is: ', t_np.shape)

############################  Data generation for ML Solver output ##############################
############################  Data generation for ML Solver output ##############################
############################  Data generation for ML Solver output ##############################
############################  Data generation for ML Solver output ##############################			




#######################################################################################################3
#######################################################################################################3
#######################################################################################################3
#######################################################################################################3
#######################################################################################################3
#######################################################################################################3
		# 	## Plot the world pos, prev pos and tgt pos
		# 	fig = plt.figure(figsize = (12, 10))
		# 	plt.rcParams.update({'font.size': 20})
		# 	plt.subplots_adjust(wspace=0.15, hspace = 0.1, left = 0.05, right=0.95, top=0.9, bottom=0.05)
		# 	# plt.suptitle(f'bc is: {bc}, thickness is: {*thickness,},\n obs is: {*obstacle,},\n SSIM (solution, smoothed): {score:.2f} --> {new_ssim:.2f}')
		# 	plt.suptitle(f'raw data: case: {i}, time step: {time_step}')
		# 	ax = fig.add_subplot(2, 2, 1, projection='3d')
		# 	im = ax.scatter(m[time_step, :,0], m[time_step, :,1], m[time_step, :,2], c=m[time_step, :,2], cmap='jet')
		# 	ax.title.set_text('mesh position')
		# 	ax.set_box_aspect([np.ptp(i) for i in (m[time_step, :,0], m[time_step, :,1], m[time_step, :,2])])  # equal aspect ratio
		# 	# cbaxes = fig.add_axes([0.22, 0.55, 0.02, 0.15])
		# 	# plt.colorbar(im, cax = cbaxes)

		# 	ax = fig.add_subplot(2, 2, 2, projection='3d')
		# 	im = ax.scatter(w[time_step, :,0], w[time_step, :,1], w[time_step, :,2], c='r', cmap='jet')
		# 	ax.title.set_text('world position')
		# 	ax.set_box_aspect([np.ptp(i) for i in (w[time_step, :,0], w[time_step, :,1], w[time_step, :,2])])  # equal aspect ratio
		# 	# cbaxes = fig.add_axes([0.65, 0.55, 0.02, 0.15])
		# 	# plt.colorbar(im, cax = cbaxes)
		# 	# plt.colorbar(im)
		# 	ax = fig.add_subplot(2, 2, 3, projection='3d')
		# 	im = ax.scatter(p[time_step, :,0], p[time_step, :,1], p[time_step, :,2], c='g', cmap='jet')
		# 	ax.set_box_aspect([np.ptp(i) for i in (p[time_step, :,0], p[time_step, :,1], p[time_step, :,2])])  # equal aspect ratio
		# 	ax.title.set_text('previous position')

		# 	ax = fig.add_subplot(2, 2, 4, projection='3d')
		# 	im = ax.scatter(t[time_step, :,0], t[time_step, :,1], t[time_step, :,2], c='b', cmap='jet')
		# 	ax.set_box_aspect([np.ptp(i) for i in (t[time_step, :,0], t[time_step, :,1], t[time_step, :,2])])  # equal aspect ratio
		# 	ax.title.set_text('target position')

		# 	plt.savefig(f'./meshgraphnets/figs/case_{i}_time step_{time_step}')
		# 	plt.close()


		# 	# Plot the mesh pos, node type and stress
		# 	fig = plt.figure(figsize = (12, 10))
		# 	plt.rcParams.update({'font.size': 20})
		# 	plt.subplots_adjust(wspace=0.15, hspace = 0.1, left = 0.05, right=0.9, top=0.8, bottom=0.05)
		# 	# plt.suptitle(f'bc is: {bc}, thickness is: {*thickness,},\n obs is: {*obstacle,},\n SSIM (solution, smoothed): {score:.2f} --> {new_ssim:.2f}')
			
		# 	plt.suptitle(f'raw data: case: {i}, time step: {time_step},\n mesh pos == world pos?:  {sum(m[time_step, :,0]==w[time_step, :,0])==len(w[time_step, :,0])}')
		# 	ax = fig.add_subplot(2, 2, 1, projection='3d')
		# 	im = ax.scatter(m[time_step, :,0], m[time_step, :,1], m[time_step, :,2], c=m[time_step, :,2], cmap='jet')
		# 	ax.title.set_text('mesh position')
		# 	ax.set_box_aspect([np.ptp(i) for i in (m[time_step, :,0], m[time_step, :,1], m[time_step, :,2])])  # equal aspect ratio
		# 	cbaxes = fig.add_axes([0.47, 0.45, 0.02, 0.15])
		# 	plt.colorbar(im, cax = cbaxes)

		# 	ax = fig.add_subplot(2, 2, 2, projection='3d')
		# 	im = ax.scatter(w[time_step, :,0], w[time_step, :,1], w[time_step, :,2], c='g', cmap='jet')
		# 	ax.set_box_aspect([np.ptp(i) for i in (w[time_step, :,0], w[time_step, :,1], w[time_step, :,2])])  # equal aspect ratio
		# 	ax.title.set_text('world position')
		# 	cbaxes = fig.add_axes([0.9, 0.45, 0.02, 0.15])
		# 	plt.colorbar(im, cax = cbaxes)

		# 	ax = fig.add_subplot(2, 2, 3, projection='3d')
		# 	im = ax.scatter(w[time_step, :,0], w[time_step, :,1], w[time_step, :,2], c=n[time_step,:,0], cmap='jet')
		# 	ax.title.set_text('node type')
		# 	ax.set_box_aspect([np.ptp(i) for i in (w[time_step, :,0], w[time_step, :,1], w[time_step, :,2])])  # equal aspect ratio
		# 	cbaxes = fig.add_axes([0.47, 0.1, 0.02, 0.15])
		# 	plt.colorbar(im, cax = cbaxes)
		# 	# plt.colorbar(im)

		# 	ax = fig.add_subplot(2, 2, 4, projection='3d')
		# 	im = ax.scatter(w[time_step, :,0], w[time_step, :,1], w[time_step, :,2], c=s[time_step,:,0], cmap='jet')
		# 	ax.set_box_aspect([np.ptp(i) for i in (w[time_step, :,0], w[time_step, :,1], w[time_step, :,2])])  # equal aspect ratio
		# 	ax.title.set_text('stress')
		# 	cbaxes = fig.add_axes([0.9, 0.1, 0.02, 0.15])
		# 	plt.colorbar(im, cax = cbaxes)

		# 	plt.savefig(f'./meshgraphnets/figs/node_type_stress_case_{i}_time_step_{time_step}')
		# 	plt.close()

		# 	# print('mesh pos: ', m[time_step, :,0][:100])
		# 	# print('world pos: ', w[time_step, :,0][:100])


		# 	# Plot the stress change along with time
		# 	fig = plt.figure(figsize = (12, 10))
		# 	plt.rcParams.update({'font.size': 20})
		# 	plt.subplots_adjust(wspace=0.15, hspace = 0.1, left = 0.05, right=0.95, top=0.9, bottom=0.05)
		# 	# plt.suptitle(f'bc is: {bc}, thickness is: {*thickness,},\n obs is: {*obstacle,},\n SSIM (solution, smoothed): {score:.2f} --> {new_ssim:.2f}')
		# 	ax = fig.add_subplot(1, 1, 1, projection='3d')
		# 	im = ax.scatter(w[time_step, :,0], w[time_step, :,1], w[time_step, :,2], c=s[time_step,:,0], cmap='jet')
		# 	# ax.title.set_text('world position')
		# 	ax.set_box_aspect([np.ptp(i) for i in (w[time_step, :,0], w[time_step, :,1], w[time_step, :,2])])  # equal aspect ratio
		# 	cbaxes = fig.add_axes([0.875, 0.35, 0.02, 0.15])
		# 	plt.colorbar(im, cax = cbaxes)
		# 	fig_test_name = f'./meshgraphnets/figs/word_pos_case_{i}_time step_{time_step}.jpg'
		# 	plt.savefig(fig_test_name)
		# 	# raise
		# 	# cbaxes = fig.add_axes([0.22, 0.55, 0.02, 0.15])
		# 	# plt.colorbar(im, cax = cbaxes)

		# 	image = fig_test_name
		# 	img = cv2.imread(os.path.join(image))
		# 	# print('img shape is: ', img.shape)
		# 	# raise
		# 	# converting BGR to RGB
		# 	# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# 	writer.append_data(img)
		# #             video.write(img)
		# 	plt.close()

#######################################################################################################3
#######################################################################################################3
#######################################################################################################3
#######################################################################################################3
#######################################################################################################3
#######################################################################################################3
#######################################################################################################3

		count += 1
		if count >= 3:
			raise
raise







# def add_targets(ds, fields, add_history):
#   """Adds target and optionally history fields to dataframe."""
#   def fn(trajectory):
#     out = {}
#     for key, val in trajectory.items():
#       out[key] = val[1:-1]
#       if key in fields:
#         if add_history:
#           out['prev|'+key] = val[0:-2]
#         out['target|'+key] = val[2:]
#     return out
#   return ds.map(fn, num_parallel_calls=8)


# cells = np.load('./meshgraphnets/cells.npz', allow_pickle=True)['arr_0']
# mesh_pos = np.load('./meshgraphnets/mesh_pos.npz', allow_pickle=True)['arr_0']
# node_type = np.load('./meshgraphnets/node_type.npz', allow_pickle=True)['arr_0']
# world_pos = np.load('./meshgraphnets/world_pos.npz', allow_pickle=True)['arr_0']
# prev_world_pos = np.load('./meshgraphnets/prev_world_pos.npz', allow_pickle=True)['arr_0']
# target_world_pos = np.load('./meshgraphnets/target_world_pos.npz', allow_pickle=True)['arr_0']
# stress = np.load('./meshgraphnets/stress.npz', allow_pickle=True)['arr_0']

# print('cells: ', cells)
# print('mesh_pos: ', mesh_pos)
# print('node_type: ', node_type)
# print('world_pos: ', world_pos)
# print('prev_world_pos: ', prev_world_pos)
# print('target_world_pos: ', target_world_pos)
# print('stress: ', stress)
# i = 0
# print('cells: ', len(cells[i]))
# print('mesh_pos: ', len(mesh_pos[i]))
# print('node_type: ', len(node_type[i]))
# print('world_pos: ', len(world_pos[i]))
# print('prev_world_pos: ', len(prev_world_pos[i]))
# print('target_world_pos: ', len(target_world_pos[i]))
# print('stress: ', len(stress[i]))
# raise
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