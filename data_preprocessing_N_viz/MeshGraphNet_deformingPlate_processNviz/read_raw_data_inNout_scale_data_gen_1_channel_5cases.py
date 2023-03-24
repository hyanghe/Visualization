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
from scipy.ndimage import gaussian_filter
import meshio
import trimesh


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


# mlsolver_output_x = np.load('./meshgraphnets/data/deforming_plate/mlsolver_x_case_39.npy')
# print('mlsolver_output_x shape: ', mlsolver_output_x.shape)
# for num_time_step in range(mlsolver_output_x.shape[0]):
# 	for z_idx in [0, 7, 15]:
# 	# for z_idx in [0, 1, 2]:
# 		plt.imshow(mlsolver_output_x[num_time_step, :,:,z_idx], cmap='jet')
# 		plt.savefig(f'./meshgraphnets/figs/mlsolver_output_x_{num_time_step}_{z_idx}.jpg')
# raise



raw_data_f = './meshgraphnets/data/deforming_plate/raw_data.pkl'


with open(raw_data_f, 'rb') as file:
    myvar = pickle.load(file)
    print(myvar.keys())
    all_data = myvar

out = {}
key = 'world_pos'
val = myvar[key]
print('val len is: ', len(val))
#98x400x1341x3



######################################## Cherry pick full plates ###################
######################################## Cherry pick full plates ###################
######################################## Cherry pick full plates ###################
def get_surface_mesh(mesh):
    points = mesh.points
    tetra = mesh.cells[0].data
    tri = np.concatenate((tetra[:, :3], tetra[:, 1:], tetra[:, [0, 1, 3]], tetra[:, [0, 2, 3]]))
    tri = np.sort(tri, axis=1)
    tri, n_counts = np.unique(tri, axis=0, return_counts=True)
    surface_cells = tri[n_counts == 1]
    surface_mesh = meshio.Mesh(points, [("triangle", surface_cells)])
    return surface_mesh

def get_mesh(dict_data, data_idx=0, t_idx=0, keys=None, file_name=None, allowed_node_types=None):
    points = dict_data["mesh_pos"][data_idx][t_idx]
    cells = dict_data["cells"][data_idx][t_idx]

    points_data = {}
    if keys is not None:
        for key in keys:
            points_data[key] = dict_data[key][data_idx][t_idx, :, 0]

    if allowed_node_types is not None:
        assert isinstance(allowed_node_types, list)
        node_types = dict_data["node_type"][data_idx][t_idx, :, 0]
        # print('node_types: ', node_types)
        node_mask = np.isin(node_types, allowed_node_types)
        points = points[node_mask]
        mask_cells = np.all(np.isin(cells, np.where(node_mask)[0]), axis=1)
        cells = cells[mask_cells]
        # renumber cells, since nodes id's have changed.
        cells_renumbering_dict = {k: v for (v, k) in enumerate(np.where(node_mask)[0])}
        cells = np.vectorize(cells_renumbering_dict.get)(cells)
        for key in points_data.keys():
            points_data[key] = points_data[key][node_mask]

    cells = [("tetra", cells)]
    mesh = meshio.Mesh(points, cells, point_data=points_data)
    if file_name is not None:
        mesh.write(file_name)
    return mesh

def get_plate_mesh(dict_data, **kwargs):
        return get_mesh(dict_data, allowed_node_types=[0, 3],**kwargs)

def get_raster_points(z, eps=5e-3):
	IMG_MIN_X = 0
	IMG_MAX_X = 0.25
	IMG_MIN_Y = 0 
	IMG_MAX_Y = 0.5
	xmin, xmax = IMG_MIN_X+eps, IMG_MAX_X-eps
	ymin, ymax = IMG_MIN_Y+eps, IMG_MAX_Y-eps
	nx, ny = (32, 64)
	points = np.meshgrid(
	    np.linspace(xmin, xmax, nx),
	    np.linspace(ymin, ymax, ny),
	    np.linspace(z, z, 1)
	)
	points = np.stack(points)
	points = np.swapaxes(points, 1, 2)
	points = points.reshape(3, -1).transpose().astype(np.float32)
	return points

def get_occupancy(mesh, points):
    new_mesh = trimesh.Trimesh(vertices=mesh.points, faces=mesh.cells[0].data)
    occ = new_mesh.contains(points).astype(float)
    return occ

def process_one_geometry(mesh):
	mid_z = mesh.points.mean(axis=0)[-1]
	plate_raster_points = get_raster_points(mid_z)
	occ = get_occupancy(mesh, plate_raster_points)
	occ = occ.reshape((32, 64))
	return occ

plate_data = []
n_data = len(all_data["mesh_pos"])
for data_idx in range(n_data):
	plate_mesh = get_plate_mesh(all_data, data_idx=data_idx)
	plate_surface_mesh = get_surface_mesh(plate_mesh)
	plate_data.append(plate_surface_mesh)


surface_mesh_data = plate_data
full_plate_idx = []
for i in range(len(surface_mesh_data)):
# for i in range(10):
	mesh = surface_mesh_data[i]
	# data_processed = [process_one_geometry(mesh) for mesh in surface_mesh_data[i:i+1]]
	data_processed = process_one_geometry(mesh)
	if np.var(data_processed) == 0:
		full_plate_idx.append(i)
		# fig = plt.figure()
		# ax = fig.add_subplot(1, 1, 1)
		# im = ax.imshow(data_processed, cmap='jet')
		# ax.title.set_text(f'plate')
		# plt.colorbar(im)
		# fig_test_name = f'./meshgraphnets/figs/plate_{i}.jpg'
		# plt.savefig(fig_test_name)
######################################## Cherry pick full plates ###################
######################################## Cherry pick full plates ###################
######################################## Cherry pick full plates ###################


######################################## Select a few cases from full plates #######
# selected_cases = [0, 3, 39, 79, 85, 89]
# selected_cases = [3, 39, 79, 85, 89]
selected_cases = [85, 89]
######################################## Select a few cases from full plates #######


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


GS_sigma = 3

count = 0

# case_id = 2
for i, (c, m, n, w, p, t, s) in enumerate(zip(cells, mesh_pos, node_type, world_pos, prev_world_pos, target_world_pos, stress)):
# for i, (c, m, n, w, p, t, s) in enumerate(zip(cells[case_id:], mesh_pos[case_id:], node_type[case_id:],\
# 								world_pos[case_id:], prev_world_pos[case_id:],\
# 								 target_world_pos[case_id:], stress[case_id:])):
	if i not in selected_cases:
		continue
	case_idx = i

	interpolated_grid_moving_delta_step_x_v = []
	interpolated_grid_moving_delta_step_y_v = []
	interpolated_grid_moving_delta_step_z_v = []
	interpolated_node_type_3D_v = []

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
		


	with imageio.get_writer('./meshgraphnets/figs/' + f'movie_output_{case_idx}.gif', mode='I') as writer:

		# for time_step in range(150, len(m), 50):
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

			w_np_step = w_np[time_step]
			t_np_step = t_np[time_step]
			n_np_step = n_np[time_step]

			m_np_step = m_np[time_step] # m_np_step shape is:  (1305, 3)

			plate_type_value = 2
			p_idx_step = (n_np_step == 0).flatten()
			n_np_step[p_idx_step] = plate_type_value
###########################   Scale m_np_step to make z thickness to be the same as width ##########
			print('m_np_step[:, 2] min, max: ', m_np_step[p_idx_step, 2].min(), m_np_step[p_idx_step, 2].max())

			m_np_step[p_idx_step, 2] = m_np_step[p_idx_step, 2] / m_np_step[p_idx_step, 2].max() * \
										m_np_step[:, 0].max()
			print('After scaling, m_np_step[:, 2] min, max: ', m_np_step[p_idx_step, 2].min(),\
										m_np_step[p_idx_step, 2].max())
###########################   Scale m_np_step to make z thickness to be the same as width ##########
			

			bc_idx_step = (n_np_step == 3).flatten()
			n_np_step[bc_idx_step] = plate_type_value

			####################### Include the obstacle ########################
			# obs_idx_step = (n_np_step == 1).flatten()
			obs_idx_step = (n_np_step == 100).flatten()
			####################### Include the obstacle ########################

			####################### Find obs bounding box ########################
			obs_idx_BoundingBox = (n_np_step == 1).flatten()
			obs_bbox_x_mn = m_np_step[obs_idx_BoundingBox, 0].min()
			obs_bbox_x_mx = m_np_step[obs_idx_BoundingBox, 0].max()
			obs_bbox_y_mn = m_np_step[obs_idx_BoundingBox, 1].min()
			obs_bbox_y_mx = m_np_step[obs_idx_BoundingBox, 1].max()
			obs_bbox_z_mn = m_np_step[obs_idx_BoundingBox, 2].min()
			obs_bbox_z_mx = m_np_step[obs_idx_BoundingBox, 2].max()
			# print('obs_bbox_x_mn, obs_bbox_x_mx:', obs_bbox_x_mn, obs_bbox_x_mx)
			# print('obs_bbox_y_mn, obs_bbox_y_mx:', obs_bbox_y_mn, obs_bbox_y_mx)
			# print('obs_bbox_z_mn, obs_bbox_z_mx:', obs_bbox_z_mn, obs_bbox_z_mx)
			# raise
			####################### Find obs bounding box ########################


			print('obs_idx_step shape is: ', obs_idx_step.shape)
			plate_idx_step = ~obs_idx_step
			plate_tgt_pos_step = t_np_step[plate_idx_step]
			plate_w_pos_step = w_np_step[plate_idx_step]
			plate_m_pos_step = m_np_step[plate_idx_step]


############################  Data generation for ML Solver output ##############################
############################  Data generation for ML Solver output ##############################
############################  Data generation for ML Solver output ##############################
############################  Data generation for ML Solver output ##############################
			plate_moving_delta_step = plate_tgt_pos_step - plate_w_pos_step
			print('plate_moving_delta_step is: ', plate_moving_delta_step.shape)

############################  Add void points ##############################################
############################  Add void points ##############################################
############################  Add void points ##############################################
			print('obs_bbox_x_mn, obs_bbox_x_mx:', obs_bbox_x_mn, obs_bbox_x_mx)
			print('obs_bbox_y_mn, obs_bbox_y_mx:', obs_bbox_y_mn, obs_bbox_y_mx)
			print('obs_bbox_z_mn, obs_bbox_z_mx:', obs_bbox_z_mn, obs_bbox_z_mx)
			print('plate_m_pos_step shape: ', plate_m_pos_step.shape)
			print('plate_m_pos_step[:, 0] min, max',plate_m_pos_step[:, 0].min(), plate_m_pos_step[:, 0].max())
			print('plate_m_pos_step[:, 1] min, max',plate_m_pos_step[:, 1].min(), plate_m_pos_step[:, 1].max())
			print('plate_m_pos_step[:, 2] min, max',plate_m_pos_step[:, 2].min(), plate_m_pos_step[:, 2].max())
			# print('plate_w_pos_step[0, 2] min, max',plate_w_pos_step[0, 2].min(), plate_w_pos_step[:, 2].max())

			# raise

			num_void = 1000
			void_x = np.random.uniform(plate_m_pos_step[:, 0].min(),\
									   plate_m_pos_step[:, 0].max(),\
									   size=(num_void,1))
			void_y = np.random.uniform(plate_m_pos_step[:, 1].min(),\
									   plate_m_pos_step[:, 1].max(),\
									   size=(num_void,1))
			void_z = np.random.uniform(plate_m_pos_step[:, 2].min(),\
									   0,\
									   size=(num_void,1))
			void_coords = np.concatenate((void_x, void_y, void_z), axis=1)
			# void_values = np.zeros_like(void_z)

			idx_exclude_obs = ((obs_bbox_x_mn <= void_x) & (void_x <= obs_bbox_x_mx)) & \
							  ((obs_bbox_y_mn <= void_y) & (void_y <= obs_bbox_y_mx))
			# print('idx_exclude_obs: ', idx_exclude_obs)
			# raise
			void_coords = void_coords[~idx_exclude_obs.flatten()]
			void_values = np.zeros_like(void_coords)

			# Plot the void points
			# fig = plt.figure(figsize = (12, 10))
			# plt.rcParams.update({'font.size': 20})
			# plt.subplots_adjust(wspace=0.15, hspace = 0.1, left = 0.05, right=0.95, top=0.9, bottom=0.05)
			# ax = fig.add_subplot(1, 1, 1, projection='3d')
			# im = ax.scatter(void_coords[:,0],\
			# 				void_coords[:,1],\
			# 				void_coords[:,2],\
			# 				c=void_values[:,0], cmap='jet')
			# ax.set_box_aspect([np.ptp(i) for i in (w[time_step, :,0], w[time_step, :,1], w[time_step, :,2])])  # equal aspect ratio
			# cbaxes = fig.add_axes([0.875, 0.35, 0.02, 0.15])
			# plt.colorbar(im, cax = cbaxes)
			# fig_test_name = f'./meshgraphnets/figs/void_pts_{i}_time step_{time_step}.jpg'
			# plt.savefig(fig_test_name)
			# # raise

			# print('plate_w_pos_step, plate_moving_delta_step', plate_w_pos_step.shape, plate_moving_delta_step.shape)
			# print('void_coords, void_values: ', void_coords.shape, void_values.shape)
			plate_w_pos_step = np.concatenate((plate_w_pos_step, void_coords), axis=0)
			plate_moving_delta_step = np.concatenate((plate_moving_delta_step, void_values), axis=0)
			# raise
############################  Add void points ##############################################
############################  Add void points ##############################################
############################  Add void points ##############################################

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
			# x_pixels = 16*2
			# y_pixels = 32*2
			# # z_pixels = int(y_pixels*(z_max-z_min)/(y_max-y_min))
			# z_pixels = 2*2

			x_pixels = 16*1
			y_pixels = 32*1
			# z_pixels = int(y_pixels*(z_max-z_min)/(y_max-y_min))
			# z_pixels = 2*2
			# z_pixels = x_pixels + 4
			z_pixels = x_pixels


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
			
############################## Interpolate node type #####################################
############################## Interpolate node type #####################################
############################## Interpolate node type #####################################
			# print('m_np_step shape is: ', m_np_step.shape)
			# print('m_np_step[:, 0] min max: ', m_np_step[:, 0].min(), m_np_step[:, 0].max())
			# print('m_np_step[:, 1] min max: ', m_np_step[:, 1].min(), m_np_step[:, 1].max())
			# print('m_np_step[:, 2] min max: ', m_np_step[:, 2].min(), m_np_step[:, 2].max())

			# print('grid_coords shape is: ', grid_coords.shape)
			# print('grid_coords[:, 0] min max: ', grid_coords[:, 0].min(), grid_coords[:, 0].max())
			# print('grid_coords[:, 1] min max: ', grid_coords[:, 1].min(), grid_coords[:, 1].max())
			# print('grid_coords[:, 2] min max: ', grid_coords[:, 2].min(), grid_coords[:, 2].max())
			# raise
			# void_x = np.random.uniform(grid_coords[:, 0].min(),\
			# 						   grid_coords[:, 0].max(),
			# 						   size=(1000,1))
			# void_y = np.random.uniform(grid_coords_all[:, 1].min(),\
			# 						   grid_coords_all[:, 1].max(),
			# 						   size=(1000,1))
			# void_z = np.random.uniform(m_np_step[:, 2].max(),\
			# 						   grid_coords_all[:, 1].max(),
			# 						   size=(1000,1))
			# void_coords = np.concatenate((void_x, void_y, void_z), axis=1)
			# void_values = np.zeros_like(void_z)
############# Use mesh position instead of world position #################################
			# w_np_step_N_void = np.concatenate((w_np_step, void_coords), axis=0)
			m_np_step_N_void = np.concatenate((m_np_step, void_coords), axis=0)
############# Use mesh position instead of world position #################################

			n_np_step_N_void = np.concatenate((n_np_step, void_values[:, 0:1]), axis=0)

			interpolated_node_type = gd(m_np_step_N_void, n_np_step_N_void, grid_coords,\
			 method='nearest', fill_value=0)
			print('interpolated_node_type shape is: ', interpolated_node_type.shape)
			interpolated_node_type_3D = interpolated_node_type.reshape(y_pixels, x_pixels,\
				 z_pixels)
			interpolated_node_type_3D_v.append(interpolated_node_type_3D)

			# fig = plt.figure(figsize = (8, 10), dpi=100)
			# plt.rcParams.update({'font.size': 20})
			# plt.subplots_adjust(wspace=0.35, hspace = 0.25, left = 0.05, right=0.85, top=0.85, bottom=0.05)
			# plt.suptitle(f'case: {i}, time step: {time_step}')
			# ax = fig.add_subplot(1, 1, 1, projection='3d')
			# im = ax.scatter(grid_coords[:,0], grid_coords[:,1],\
			#  grid_coords[:,2], c=interpolated_node_type, cmap='jet', alpha=0.5)
			# # im = ax.scatter(m_np_step_N_void[:,0], m_np_step_N_void[:,1],\
			# #  m_np_step_N_void[:,2], c=n_np_step_N_void, cmap='jet', alpha=0.5)
			# plt.colorbar(im)
			# ax.title.set_text('node type\n (plate + obs + void)')
			# ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0],\
			#  grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			# plt.savefig(f'./meshgraphnets/figs/node_type_grids_{time_step}.jpg')
			# raise



############################## Interpolate node type #####################################
############################## Interpolate node type #####################################
############################## Interpolate node type #####################################
			# raise
			interpolated_grid_moving_delta_step = grid_moving_delta_step.copy()
			if len(missing_idx) is not 0:
				missing_idx = np.asarray(missing_idx)
				missing_pts = grid_coords[missing_idx]

				interpolated_x_translation = gd(plate_w_pos_step, plate_moving_delta_step[:,0], missing_pts,\
				 method='linear', fill_value=0.0)
				print('interpolated_x_translation shape is: ', interpolated_x_translation.shape)
				interpolated_grid_moving_delta_step[missing_idx, 0] = interpolated_x_translation

				interpolated_y_translation = gd(plate_w_pos_step, plate_moving_delta_step[:,1], missing_pts,\
				 method='linear', fill_value=0.0)
				interpolated_grid_moving_delta_step[missing_idx, 1] = interpolated_y_translation

				interpolated_z_translation = gd(plate_w_pos_step, plate_moving_delta_step[:,2], missing_pts,\
				 method='linear', fill_value=0.0)
				interpolated_grid_moving_delta_step[missing_idx, 2] = interpolated_z_translation
				# raise
			
			print('interpolated_grid_moving_delta_step shape: ', interpolated_grid_moving_delta_step.shape)

#######################  Apply smoothing on solution ########################3
			grid_sln_x = interpolated_grid_moving_delta_step.reshape((y_pixels, x_pixels, z_pixels, 3))[:,:,:,0]
			grid_sln_x_GS = gaussian_filter(grid_sln_x, sigma=GS_sigma)
			grid_sln_y = interpolated_grid_moving_delta_step.reshape((y_pixels, x_pixels, z_pixels, 3))[:,:,:,1]
			grid_sln_y_GS = gaussian_filter(grid_sln_y, sigma=GS_sigma)
			grid_sln_z = interpolated_grid_moving_delta_step.reshape((y_pixels, x_pixels, z_pixels, 3))[:,:,:,2]
			grid_sln_z_GS = gaussian_filter(grid_sln_z, sigma=GS_sigma)

			interpolated_grid_moving_delta_step = np.concatenate((grid_sln_x_GS.reshape(-1, 1),\
																  grid_sln_y_GS.reshape(-1, 1),\
																  grid_sln_z_GS.reshape(-1, 1),\
																 ), axis=-1)
			print('After smoothing, interpolated_grid_moving_delta_step shape: ', interpolated_grid_moving_delta_step.shape)

			# fig = plt.figure(figsize = (18, 20), dpi=100)
			# plt.rcParams.update({'font.size': 20})
			# plt.subplots_adjust(wspace=1, hspace = 0.5, left = 0.05, right=0.85, top=0.85, bottom=0.05)
			# ax = fig.add_subplot(1, 1, 1, projection='3d')
			# im = ax.scatter(grid_coords[:,0],\
			# 				grid_coords[:,1],\
			# 				grid_coords[:,2],\
			# 				c=grid_sln_z_GS.flatten(), cmap='jet')
			# ax.title.set_text('graph nodes\n (plate + obs + void points)')
			# plt.colorbar(im)
			# plt.savefig(f'./meshgraphnets/figs/case_{i}_time step_{time_step}_GS_sln')

			# raise
#######################  Apply smoothing on solution ########################3


#######################  Apply smoothing on node type ########################
			print('interpolated_node_type_3D shape: ', interpolated_node_type_3D.shape)
			# interpolated_node_type_3D_GS = gaussian_filter(interpolated_node_type_3D, sigma=GS_sigma)
			interpolated_node_type_3D_GS = gaussian_filter(interpolated_node_type_3D, sigma=2)

			print('After smoothing, interpolated_node_type_3D_GS shape: ', interpolated_node_type_3D_GS.shape)

			# fig = plt.figure(figsize = (18, 20), dpi=100)
			# plt.rcParams.update({'font.size': 20})
			# plt.subplots_adjust(wspace=1, hspace = 0.5, left = 0.05, right=0.85, top=0.85, bottom=0.05)
			# ax = fig.add_subplot(1, 1, 1, projection='3d')
			# im = ax.scatter(grid_coords[:,0],\
			# 				grid_coords[:,1],\
			# 				grid_coords[:,2],\
			# 				c=interpolated_node_type_3D_GS.flatten(), cmap='jet')
			# ax.title.set_text('graph nodes\n (plate + obs + void points)')
			# plt.colorbar(im)
			# plt.savefig(f'./meshgraphnets/figs/case_{i}_time step_{time_step}_GS_node_type')
#######################  Apply smoothing on solution ########################3

# ################################ Visualizing the cross section  node type ################################
# ################################ Visualizing the cross section  node type ################################
# ################################ Visualizing the cross section  node type ################################
# 			fig = plt.figure(figsize = (18, 20), dpi=100)
# 			plt.rcParams.update({'font.size': 20})
# 			plt.subplots_adjust(wspace=1, hspace = 0.5, left = 0.05, right=0.85, top=0.85, bottom=0.05)

# 			ax = fig.add_subplot(4, 3, 1, projection='3d')
# 			im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2],\
# 			 c=interpolated_node_type_3D.flatten(), cmap='jet')
# 			ax.title.set_text('interpolated \n node type')
# 			ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1],\
# 			 grid_coords[:,2])])  # equal aspect ratio
# 			cbaxes = fig.add_axes([0.28, 0.75, 0.01, 0.05])
# 			plt.colorbar(im, cax = cbaxes)

# 			grid_coords_3D = grid_coords.reshape((y_pixels, x_pixels, z_pixels, 3))

# 			ax = fig.add_subplot(4, 3, 4)
# 			im = ax.imshow(interpolated_node_type_3D_GS[y_pixels//2,:,:].T, origin='lower', cmap='jet')
# 			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
# 			ax.title.set_text('xz center plane')
# 			cbaxes = fig.add_axes([0.285, 0.52, 0.01, 0.05])
# 			plt.colorbar(im, cax = cbaxes)

# 			ax = fig.add_subplot(4, 3, 5)
# 			im = ax.imshow(interpolated_node_type_3D_GS[:,x_pixels//2,:].T, origin='lower', cmap='jet')
# 			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
# 			ax.title.set_text('yz center plane')
# 			cbaxes = fig.add_axes([0.6, 0.52, 0.01, 0.05])
# 			plt.colorbar(im, cax = cbaxes)

# 			ax = fig.add_subplot(4, 3, 6)
# 			im = ax.imshow(interpolated_node_type_3D_GS[:,:,z_pixels//2].T, origin='lower', cmap='jet')
# 			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
# 			ax.title.set_text('xy center plane')
# 			cbaxes = fig.add_axes([0.9, 0.52, 0.01, 0.05])
# 			plt.colorbar(im, cax = cbaxes)

# 			ax = fig.add_subplot(4, 3, 7)
# 			im = ax.imshow(interpolated_node_type_3D_GS[0,:,:].T, origin='lower', cmap='jet')
# 			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
# 			ax.title.set_text('xz bottom plane')
# 			cbaxes = fig.add_axes([0.285, 0.29, 0.01, 0.05])
# 			plt.colorbar(im, cax = cbaxes)

# 			ax = fig.add_subplot(4, 3, 8)
# 			im = ax.imshow(interpolated_node_type_3D_GS[:,0,:].T, origin='lower', cmap='jet')
# 			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
# 			ax.title.set_text('yz bottom plane')
# 			cbaxes = fig.add_axes([0.6, 0.29, 0.01, 0.05])
# 			plt.colorbar(im, cax = cbaxes)

# 			ax = fig.add_subplot(4, 3, 9)
# 			im = ax.imshow(interpolated_node_type_3D_GS[:,:,0].T, origin='lower', cmap='jet')
# 			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
# 			ax.title.set_text('xy bottom plane')
# 			cbaxes = fig.add_axes([0.9, 0.29, 0.01, 0.05])
# 			plt.colorbar(im, cax = cbaxes)

# 			ax = fig.add_subplot(4, 3, 10)
# 			im = ax.imshow(interpolated_node_type_3D_GS[-1,:,:].T, origin='lower', cmap='jet')
# 			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
# 			ax.title.set_text('xz top plane')
# 			cbaxes = fig.add_axes([0.285, 0.1, 0.01, 0.05])
# 			plt.colorbar(im, cax = cbaxes)

# 			ax = fig.add_subplot(4, 3, 11)
# 			im = ax.imshow(interpolated_node_type_3D_GS[:,-1,:].T, origin='lower', cmap='jet')
# 			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
# 			ax.title.set_text('yz top plane')
# 			cbaxes = fig.add_axes([0.6, 0.1, 0.01, 0.05])
# 			plt.colorbar(im, cax = cbaxes)

# 			ax = fig.add_subplot(4, 3, 12)
# 			im = ax.imshow(interpolated_node_type_3D_GS[:,:,-1].T, origin='lower', cmap='jet')
# 			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
# 			ax.title.set_text('xy top plane')
# 			cbaxes = fig.add_axes([0.9, 0.1, 0.01, 0.05])
# 			plt.colorbar(im, cax = cbaxes)

# 			plt.savefig(f'./meshgraphnets/figs/case_{i}_time step_{time_step}_node_interpolation_GS_{GS_sigma}')
# 			plt.close()
# 			raise

# ################################ Visualizing the cross section node type ################################
# ################################ Visualizing the cross section node type  ################################
# ################################ Visualizing the cross section node type  ################################


# ################################ Visualizing the cross section ################################
# ################################ Visualizing the cross section ################################
# ################################ Visualizing the cross section ################################
			fig = plt.figure(figsize = (18, 20), dpi=100)
			plt.rcParams.update({'font.size': 20})
			plt.subplots_adjust(wspace=1, hspace = 0.5, left = 0.05, right=0.85, top=0.85, bottom=0.05)

			ax = fig.add_subplot(4, 3, 1, projection='3d')
			im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c=interpolated_grid_moving_delta_step[:, 0], cmap='jet')
			ax.title.set_text('interpolated \n x translation')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			cbaxes = fig.add_axes([0.28, 0.75, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 3, 2, projection='3d')
			im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c=interpolated_grid_moving_delta_step[:, 1], cmap='jet')
			ax.title.set_text('interpolated \n y translation')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			cbaxes = fig.add_axes([0.56, 0.75, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 3, 3, projection='3d')
			im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c=interpolated_grid_moving_delta_step[:, 2], cmap='jet')
			ax.title.set_text('interpolated \n z translation')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			cbaxes = fig.add_axes([0.84, 0.75, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			grid_coords_3D = grid_coords.reshape((y_pixels, x_pixels, z_pixels, 3))
			interpoldated_solution_3D = interpolated_grid_moving_delta_step.reshape((y_pixels,\
																					 x_pixels,\
																					 z_pixels,\
																					 3))
			ax = fig.add_subplot(4, 3, 4)
			im = ax.imshow(interpoldated_solution_3D[y_pixels//2,:,:,0].T, origin='lower', cmap='jet')
			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
			ax.title.set_text('xz center plane \n (x trans.)')
			cbaxes = fig.add_axes([0.285, 0.52, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 3, 5)
			im = ax.imshow(interpoldated_solution_3D[:,x_pixels//2,:,0].T, origin='lower', cmap='jet')
			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
			ax.title.set_text('yz center plane \n (x trans.)')
			cbaxes = fig.add_axes([0.6, 0.52, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 3, 6)
			im = ax.imshow(interpoldated_solution_3D[:,:,z_pixels//2,0].T, origin='lower', cmap='jet')
			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
			ax.title.set_text('xy center plane \n (x trans.)')
			cbaxes = fig.add_axes([0.9, 0.52, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 3, 7)
			im = ax.imshow(interpoldated_solution_3D[0,:,:,0].T, origin='lower', cmap='jet')
			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
			ax.title.set_text('xz bottom plane \n (x trans.)')
			cbaxes = fig.add_axes([0.285, 0.29, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 3, 8)
			im = ax.imshow(interpoldated_solution_3D[:,0,:,0].T, origin='lower', cmap='jet')
			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
			ax.title.set_text('yz bottom plane \n (x trans.)')
			cbaxes = fig.add_axes([0.6, 0.29, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 3, 9)
			im = ax.imshow(interpoldated_solution_3D[:,:,0,0].T, origin='lower', cmap='jet')
			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
			ax.title.set_text('xy bottom plane \n (x trans.)')
			cbaxes = fig.add_axes([0.9, 0.29, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 3, 10)
			im = ax.imshow(interpoldated_solution_3D[-1,:,:,0].T, origin='lower', cmap='jet')
			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
			ax.title.set_text('xz top plane \n (x trans.)')
			cbaxes = fig.add_axes([0.285, 0.1, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 3, 11)
			im = ax.imshow(interpoldated_solution_3D[:,-1,:,0].T, origin='lower', cmap='jet')
			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
			ax.title.set_text('yz top plane \n (x trans.)')
			cbaxes = fig.add_axes([0.6, 0.1, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 3, 12)
			im = ax.imshow(interpoldated_solution_3D[:,:,-1,0].T, origin='lower', cmap='jet')
			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
			ax.title.set_text('xy top plane \n (x trans.)')
			cbaxes = fig.add_axes([0.9, 0.1, 0.01, 0.05])
			plt.colorbar(im, cax = cbaxes)

			plt.savefig(f'./meshgraphnets/figs/case_{case_idx}_time step_{time_step}_data_interpolation_GS_{GS_sigma}')
			plt.close()
			# raise

# ################################ Visualizing the cross section ################################
# ################################ Visualizing the cross section ################################
# ################################ Visualizing the cross section ################################




################################ Visualizing the data processing steps ################################
################################ Visualizing the data processing steps ################################
################################ Visualizing the data processing steps ################################
			# fig = plt.figure(figsize = (18, 20), dpi=100)
			# plt.rcParams.update({'font.size': 20})
			# plt.subplots_adjust(wspace=0.35, hspace = 0.25, left = 0.05, right=0.85, top=0.85, bottom=0.05)
			# # plt.suptitle(f'bc is: {bc}, thickness is: {*thickness,},\n obs is: {*obstacle,},\n SSIM (solution, smoothed): {score:.2f} --> {new_ssim:.2f}')
			# plt.suptitle(f'case: {i}, time step: {time_step}')
			# ax = fig.add_subplot(4, 3, 1, projection='3d')
			# im = ax.scatter(plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2], c='r', cmap='jet')
			# ax.title.set_text('graph nodes\n (plate + obs + void points)')
			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
			# # cbaxes = fig.add_axes([0.22, 0.55, 0.02, 0.15])
			# # plt.colorbar(im, cax = cbaxes)

			# ax = fig.add_subplot(4, 3, 2, projection='3d')
			# im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c='g', cmap='jet')
			# ax.title.set_text('grid nodes')
			# ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			# # cbaxes = fig.add_axes([0.65, 0.55, 0.02, 0.15])
			# # plt.colorbar(im, cax = cbaxes)
			# # plt.colorbar(im)

			# ax = fig.add_subplot(4, 3, 3, projection='3d')
			# im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c='g', cmap='jet')
			# im = ax.scatter(mappable_grid_coords[:,0], mappable_grid_coords[:,1],\
			#  mappable_grid_coords[:,2], c='r', cmap='jet')
			# ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			# ax.legend(['Grids', 'Mapping'], loc = 'lower right', bbox_to_anchor=(1.7, 0.1))
			# ax.title.set_text('grid nodes \n and mapping')

			# ax = fig.add_subplot(4, 3, 4, projection='3d')
			# im = ax.scatter(plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2], c=plate_moving_delta_step[:, 0], cmap='jet')
			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
			# ax.title.set_text('graph node \n x translation')
			# cbaxes = fig.add_axes([0.28, 0.49, 0.01, 0.05])
			# plt.colorbar(im, cax = cbaxes)

			# ax = fig.add_subplot(4, 3, 5, projection='3d')
			# im = ax.scatter(plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2], c=plate_moving_delta_step[:, 1], cmap='jet')
			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
			# ax.title.set_text('graph node \n y translation')
			# cbaxes = fig.add_axes([0.56, 0.49, 0.01, 0.05])
			# plt.colorbar(im, cax = cbaxes)

			# ax = fig.add_subplot(4, 3, 6, projection='3d')
			# im = ax.scatter(plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2], c=plate_moving_delta_step[:, 2], cmap='jet')
			# ax.set_box_aspect([np.ptp(i) for i in (plate_w_pos_step[:,0], plate_w_pos_step[:,1], plate_w_pos_step[:,2])])  # equal aspect ratio
			# ax.title.set_text('graph node \n z translation')
			# cbaxes = fig.add_axes([0.84, 0.49, 0.01, 0.05])
			# plt.colorbar(im, cax = cbaxes)

			# ax = fig.add_subplot(4, 3, 7, projection='3d')
			# im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c=grid_moving_delta_step[:, 0], cmap='jet')
			# ax.title.set_text('grid nodes \n x translation')
			# ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			# cbaxes = fig.add_axes([0.28, 0.275, 0.01, 0.05])
			# plt.colorbar(im, cax = cbaxes)

			# ax = fig.add_subplot(4, 3, 8, projection='3d')
			# im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c=grid_moving_delta_step[:, 1], cmap='jet')
			# ax.title.set_text('grid nodes \n y translation')
			# ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			# cbaxes = fig.add_axes([0.56, 0.275, 0.01, 0.05])
			# plt.colorbar(im, cax = cbaxes)


			# ax = fig.add_subplot(4, 3, 9, projection='3d')
			# im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c=grid_moving_delta_step[:, 2], cmap='jet')
			# ax.title.set_text('grid nodes \n z translation')
			# ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			# cbaxes = fig.add_axes([0.84, 0.275, 0.01, 0.05])
			# plt.colorbar(im, cax = cbaxes)

			# ax = fig.add_subplot(4, 3, 10, projection='3d')
			# im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c=interpolated_grid_moving_delta_step[:, 0], cmap='jet')
			# ax.title.set_text('interpolated \n x translation')
			# ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			# cbaxes = fig.add_axes([0.28, 0.05, 0.01, 0.05])
			# plt.colorbar(im, cax = cbaxes)

			# ax = fig.add_subplot(4, 3, 11, projection='3d')
			# im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c=interpolated_grid_moving_delta_step[:, 1], cmap='jet')
			# ax.title.set_text('interpolated \n y translation')
			# ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			# cbaxes = fig.add_axes([0.56, 0.05, 0.01, 0.05])
			# plt.colorbar(im, cax = cbaxes)

			# ax = fig.add_subplot(4, 3, 12, projection='3d')
			# im = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2], c=interpolated_grid_moving_delta_step[:, 2], cmap='jet')
			# ax.title.set_text('interpolated \n z translation')
			# ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0], grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			# cbaxes = fig.add_axes([0.84, 0.05, 0.01, 0.05])
			# plt.colorbar(im, cax = cbaxes)

			# plt.savefig(f'./meshgraphnets/figs/case_{i}_time step_{time_step}_data_interpolation_process')
			# plt.close()
			# # raise

# ################################ Visualizing the data processing steps ################################
# ################################ Visualizing the data processing steps ################################
# ################################ Visualizing the data processing steps ################################

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


			# # Plot the stress change along with time
			# fig = plt.figure(figsize = (12, 10))
			# plt.rcParams.update({'font.size': 20})
			# plt.subplots_adjust(wspace=0.15, hspace = 0.1, left = 0.05, right=0.95, top=0.9, bottom=0.05)
			# # plt.suptitle(f'bc is: {bc}, thickness is: {*thickness,},\n obs is: {*obstacle,},\n SSIM (solution, smoothed): {score:.2f} --> {new_ssim:.2f}')
			# ax = fig.add_subplot(1, 1, 1, projection='3d')
			# im = ax.scatter(w[time_step, :,0], w[time_step, :,1], w[time_step, :,2], c=s[time_step,:,0], cmap='jet')
			# # ax.title.set_text('world position')
			# ax.set_box_aspect([np.ptp(i) for i in (w[time_step, :,0], w[time_step, :,1], w[time_step, :,2])])  # equal aspect ratio
			# cbaxes = fig.add_axes([0.875, 0.35, 0.02, 0.15])
			# plt.colorbar(im, cax = cbaxes)
			# fig_test_name = f'./meshgraphnets/figs/word_pos_case_{i}_time step_{time_step}.jpg'
			# plt.savefig(fig_test_name)
			# # raise
			# # cbaxes = fig.add_axes([0.22, 0.55, 0.02, 0.15])
			# # plt.colorbar(im, cax = cbaxes)

			# Plot the stress change along with time
			# im = ax.scatter(grid_coords[:,0], grid_coords[:,1],\
			#  grid_coords[:,2], c=interpolated_node_type, cmap='jet', alpha=0.5)
			# # im = ax.scatter(m_np_step_N_void[:,0], m_np_step_N_void[:,1],\
			# #  m_np_step_N_void[:,2], c=n_np_step_N_void, cmap='jet', alpha=0.5)
			fig = plt.figure(figsize = (13, 20))
			plt.rcParams.update({'font.size': 20})
			plt.subplots_adjust(wspace=0.2, hspace = 0.15, left = 0.0, right=0.85, top=0.9, bottom=0.05)
			# plt.suptitle(f'Channel 1: translation along 1 axis (row 1, 2, 3) \n Channel 2: node type (row 4)')
			plt.suptitle(f'1 channel: translation along 1 axis (row 1, 2, 3)')

			ax = fig.add_subplot(4, 2, 1, projection='3d')
			im = ax.scatter(m_np_step_N_void[:,0], m_np_step_N_void[:,1],\
			 m_np_step_N_void[:,2], c=plate_moving_delta_step[:,0], cmap='jet', alpha=0.5)
			ax.title.set_text('Graph nodes x translation')
			ax.set_box_aspect([np.ptp(i) for i in (m_np_step_N_void[:,0],\
			 m_np_step_N_void[:,1], m_np_step_N_void[:,2])])  # equal aspect ratio
			cbaxes = fig.add_axes([0.37, 0.73, 0.02, 0.15])
			plt.colorbar(im, cax = cbaxes)

			ax = fig.add_subplot(4, 2, 2, projection='3d')
			im = ax.scatter(grid_coords[:,0], grid_coords[:,1],\
			 grid_coords[:,2], c=interpolated_grid_moving_delta_step[:, 0], cmap='jet', alpha=0.5)
			ax.title.set_text('Grid nodes x translation')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0],\
			 grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			cbaxes = fig.add_axes([0.85, 0.73, 0.02, 0.15])
			plt.colorbar(im, cax = cbaxes)		

			ax = fig.add_subplot(4, 2, 3, projection='3d')
			im = ax.scatter(m_np_step_N_void[:,0], m_np_step_N_void[:,1],\
			 m_np_step_N_void[:,2], c=plate_moving_delta_step[:,1], cmap='jet', alpha=0.5)
			ax.title.set_text('Graph nodes y translation')
			ax.set_box_aspect([np.ptp(i) for i in (m_np_step_N_void[:,0],\
			 m_np_step_N_void[:,1], m_np_step_N_void[:,2])])  # equal aspect ratio
			cbaxes = fig.add_axes([0.37, 0.5, 0.02, 0.15])
			plt.colorbar(im, cax = cbaxes)	

			ax = fig.add_subplot(4, 2, 4, projection='3d')
			im = ax.scatter(grid_coords[:,0], grid_coords[:,1],\
			 grid_coords[:,2], c=interpolated_grid_moving_delta_step[:, 1], cmap='jet', alpha=0.5)
			ax.title.set_text('Grid nodes y translation')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0],\
			 grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			cbaxes = fig.add_axes([0.85, 0.5, 0.02, 0.15])
			plt.colorbar(im, cax = cbaxes)	

			ax = fig.add_subplot(4, 2, 5, projection='3d')
			im = ax.scatter(m_np_step_N_void[:,0], m_np_step_N_void[:,1],\
			 m_np_step_N_void[:,2], c=plate_moving_delta_step[:,2], cmap='jet', alpha=0.5)
			ax.title.set_text('Graph nodes z translation')
			ax.set_box_aspect([np.ptp(i) for i in (m_np_step_N_void[:,0],\
			 m_np_step_N_void[:,1], m_np_step_N_void[:,2])])  # equal aspect ratio
			cbaxes = fig.add_axes([0.37, 0.3, 0.02, 0.15])
			plt.colorbar(im, cax = cbaxes)	

			ax = fig.add_subplot(4, 2, 6, projection='3d')
			im = ax.scatter(grid_coords[:,0], grid_coords[:,1],\
			 grid_coords[:,2], c=interpolated_grid_moving_delta_step[:, 2], cmap='jet', alpha=0.5)
			ax.title.set_text('Grid nodes z translation')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0],\
			 grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			cbaxes = fig.add_axes([0.85, 0.3, 0.02, 0.15])
			plt.colorbar(im, cax = cbaxes)	

			ax = fig.add_subplot(4, 2, 7, projection='3d')
			im = ax.scatter(m_np_step_N_void[:,0], m_np_step_N_void[:,1],\
			 m_np_step_N_void[:,2], c=n_np_step_N_void, cmap='jet', alpha=0.5)
			ax.title.set_text('Graph node type')
			ax.set_box_aspect([np.ptp(i) for i in (m_np_step_N_void[:,0],\
			 m_np_step_N_void[:,1], m_np_step_N_void[:,2])])  # equal aspect ratio
			cbaxes = fig.add_axes([0.37, 0.05, 0.02, 0.15])
			plt.colorbar(im, cax = cbaxes)	

			ax = fig.add_subplot(4, 2, 8, projection='3d')
			im = ax.scatter(grid_coords[:,0], grid_coords[:,1],\
			 grid_coords[:,2], c=interpolated_node_type_3D_GS.flatten(), cmap='jet', alpha=0.5)
			ax.title.set_text('Grid node type')
			ax.set_box_aspect([np.ptp(i) for i in (grid_coords[:,0],\
			 grid_coords[:,1], grid_coords[:,2])])  # equal aspect ratio
			cbaxes = fig.add_axes([0.85, 0.05, 0.02, 0.15])
			plt.colorbar(im, cax = cbaxes)	

			fig_test_name = f'./meshgraphnets/figs/case_{case_idx}_time step_{time_step}.jpg'
			plt.savefig(fig_test_name)
			# raise
			# raise
			# cbaxes = fig.add_axes([0.22, 0.55, 0.02, 0.15])
			# plt.colorbar(im, cax = cbaxes)

			image = fig_test_name
			img = cv2.imread(os.path.join(image))
			# print('img shape is: ', img.shape)
			# raise
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
		
			print('interpolated_grid_moving_delta_step shape is: ', interpolated_grid_moving_delta_step.shape)
			interpolated_grid_moving_delta_step = interpolated_grid_moving_delta_step.reshape(y_pixels,\
			 x_pixels, z_pixels, 3)
			print('interpolated_grid_moving_delta_step shape: ', interpolated_grid_moving_delta_step.shape)

			first_channel_x_translation = interpolated_grid_moving_delta_step[:,:,:,0]
			ml_solver_x_translation = first_channel_x_translation

			first_channel_y_translation = interpolated_grid_moving_delta_step[:,:,:,1]
			ml_solver_y_translation = first_channel_y_translation

			first_channel_z_translation = interpolated_grid_moving_delta_step[:,:,:,2]
			ml_solver_z_translation = first_channel_z_translation
			interpolated_grid_moving_delta_step_x_v.append(ml_solver_x_translation)
			interpolated_grid_moving_delta_step_y_v.append(ml_solver_y_translation)
			interpolated_grid_moving_delta_step_z_v.append(ml_solver_z_translation)
			# interpolated_node_type_3D_v.append(interpolated_node_type_3D)
			# fig = plt.figure(figsize = (12, 4))
			# plt.rcParams.update({'font.size': 15})
			# plt.subplots_adjust(wspace=0.15, hspace = 0.15, left = 0.05, right=0.95,\
			#  top=0.8, bottom=0.1)
			# plt.suptitle(f'bottom surface translation (time_step: {time_step})')
			# ax = fig.add_subplot(1, 3, 1)
			# ax.title.set_text('x')
			# im = ax.imshow(interpolated_grid_moving_delta_step[:,:,0,0], cmap='jet')
			# ax_divider = make_axes_locatable(ax)
			# cax = ax_divider.append_axes("right", size="7%", pad="2%")
			# plt.colorbar(im, cax=cax)

			# ax = fig.add_subplot(1, 3, 2)
			# ax.title.set_text('y')
			# im = ax.imshow(interpolated_grid_moving_delta_step[:,:,0,1], cmap='jet')
			# ax_divider = make_axes_locatable(ax)
			# cax = ax_divider.append_axes("right", size="7%", pad="2%")
			# plt.colorbar(im, cax=cax)

			# ax = fig.add_subplot(1, 3, 3)
			# ax.title.set_text('z')
			# im = ax.imshow(interpolated_grid_moving_delta_step[:,:,0,2], cmap='jet')
			# ax_divider = make_axes_locatable(ax)
			# cax = ax_divider.append_axes("right", size="7%", pad="2%")
			# plt.colorbar(im, cax=cax)

			# fig_test_name = f'./meshgraphnets/figs/interpolated_grid_moving_delta_step_case_{i}_time step_{time_step}.jpg'
			# plt.savefig(fig_test_name)

			# image = fig_test_name
			# img = cv2.imread(os.path.join(image))
			# writer.append_data(img)
			# plt.close()

		count += 1
		interpolated_grid_moving_delta_step_x_v = np.asarray(interpolated_grid_moving_delta_step_x_v)
		interpolated_grid_moving_delta_step_y_v = np.asarray(interpolated_grid_moving_delta_step_y_v)
		interpolated_grid_moving_delta_step_z_v = np.asarray(interpolated_grid_moving_delta_step_z_v)
		print('interpolated_grid_moving_delta_step_x_v shape: ', interpolated_grid_moving_delta_step_x_v.shape)
		print('interpolated_grid_moving_delta_step_y_v shape: ', interpolated_grid_moving_delta_step_y_v.shape)
		print('interpolated_grid_moving_delta_step_z_v shape: ', interpolated_grid_moving_delta_step_z_v.shape)

		np.save(f'./meshgraphnets/data/deforming_plate/mlsolver_x_case_{i}',\
		 interpolated_grid_moving_delta_step_x_v)
		np.save(f'./meshgraphnets/data/deforming_plate/mlsolver_y_case_{i}',\
		 interpolated_grid_moving_delta_step_y_v)
		np.save(f'./meshgraphnets/data/deforming_plate/mlsolver_z_case_{i}',\
		 interpolated_grid_moving_delta_step_z_v)

		if count >= 6:
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