import glob
import meshio
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pyvista
import cv2
from PIL import Image
import shapely.geometry as geometry

rootdir = './Dataset'
subdirs = []

def interp(pts, resolution):
    # x_min, x_max = -1, 2
    # y_min, y_max = -0.5, 0.5
    x_min, x_max = -0.25, 1
    y_min, y_max = -0.2, 0.2
    concave = False
    threshold = np.mean(pts[:,1])
    # pts_cvx_idx = (pts[:,0]>=0.1) & (pts[:,0]<=0.9) & (pts[:,1]<=0)
    pts_cvx_idx = (pts[:,0]>=0.1) & (pts[:,0]<=0.9) & (pts[:,1]<=threshold)

    pts_cvx = pts[pts_cvx_idx]
    pts_cvx = sorted(pts_cvx , key=lambda k: k[0])
    pts_cvx = np.asarray(pts_cvx)
    slope = (pts_cvx[-1, 1] - pts_cvx[0, 1]) / (pts_cvx[-1, 0] - pts_cvx[0, 0])
    test_y = pts_cvx[:,0] * slope - pts_cvx[0,0] * slope + pts_cvx[0, 1]
    if sum(test_y) < sum(pts_cvx[:,1]):
        concave = True
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.scatter(pts_cvx[:,0], pts_cvx[:,1], c = 'r')
    # ax.scatter(pts_cvx[:,0], test_y, c = 'b')
    # ax.legend(['original', 'Benchmark'])
    # plt.show()
    # x_min, x_max = pts[:,0].min(), pts[:,0].max()
    # y_min, y_max = pts[:,1].min(), pts[:,1].max()
    xx, yy = np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    xxx, yyy = np.meshgrid(xx, yy)
    grids = np.concatenate((xxx.reshape(-1, 1), yyy.reshape(-1, 1)), axis=-1)
    
    interpolated = griddata(pts[:,:2], np.ones_like(pts[:,:1])*1.0, grids, fill_value=0.0,method='linear').reshape(resolution, resolution)

    # interpolated = griddata(pts[:,:2], np.ones_like(pts[:,:1])*0.0, grids, fill_value=1.0,method='linear').reshape(resolution, resolution)

    return interpolated, concave


def area(p):
    return 0.5 * abs(sum(x0*y1 - x1*y0 for ((x0, y0), (x1, y1)) in segments(p)))

def segments(p):
    return zip(p, p[1:] + [p[0]])

resolution = 1024
# resolution = 2048

# resolution = 4096

cnt = 0
status = []
interpolated_data = []
for rootdir, dirs, files in os.walk(rootdir):
    if not dirs:
        break
    # ConvexStatus_FINAL = np.load('ConvexStatus_FINAL.npy')
    # dirs_filtered = [dirs[i] for i in range(len(ConvexStatus_FINAL)) if ConvexStatus_FINAL[i]]

    dirs_filtered = []
    with open('./dirs_filtered.txt', 'r') as fp:
        for line in fp:
            dirs_filtered.append(line)

    for subdir in dirs_filtered:
        # print('subdir: ', subdir)
        # raise
        cur_dir = os.path.join(rootdir, subdir).strip()
        
        file_airfoil = glob.glob(cur_dir + '/*aerofoil.vtp')
        file_freestream = glob.glob(cur_dir + '/*freestream.vtp')

        mesh_airfoil = pyvista.read(file_airfoil[0])
        mesh_freestream = pyvista.read(file_freestream[0])

        airfoil_pts = np.asarray(mesh_airfoil.points)
        freestream_pts = np.asarray(mesh_freestream.points)

        # pts = np.concatenate((airfoil_pts, freestream_pts), axis=0)
        # # idx = ((pts[:,0]>=0) & (pts[:,0]<=8)) & ((pts[:,1]>=-3) & (pts[:,1]<=3))
        # # airfoil_pts = pts[idx]
        # airfoil_pts = pts
        airfoil_interpolated, concave = interp(airfoil_pts, resolution)
        flag = 'Convex'
        if concave:
            flag = 'Concave'

        fig = plt.figure(figsize = (7, 3), dpi=100)
        plt.rcParams.update({'font.size': 15})
        plt.subplots_adjust(wspace=0.5, hspace = 0.2, left = 0.1, right=0.95, top=0.7, bottom=0.1)
        plt.suptitle(f'Mesh ({flag})')
        ax = fig.add_subplot(1, 2, 1)
        im = ax.scatter(airfoil_pts[:,0],airfoil_pts[:,1], cmap='jet')
        ax.title.set_text(f'air foil')

        ax = fig.add_subplot(1, 2, 2)
        im = ax.imshow(airfoil_interpolated, cmap='jet', origin='lower')
        ax.title.set_text(f'Interpolated\nairfoil')   


        # ax = fig.add_subplot(1, 4, 3)
        # im = ax.scatter(freestream_pts[:,0],freestream_pts[:,1], cmap='jet')
        # ax.title.set_text(f'freestream')    

        # # freestream_interpolated = interp(freestream_pts, resolution)
        # # ax = fig.add_subplot(1, 4, 4)
        # # im = ax.imshow(freestream_interpolated, cmap='jet', origin='lower')
        # # ax.title.set_text(f'Interpolated\nfreestream') 
        if concave:
            status.append(False)
        else:
            status.append(True)
            # plt.savefig(f'.\\ConvexFilter_2_FILTER\\case_{cnt}_{flag}')
            file_name = file_airfoil[0].split("/")[-1][:-4]
            plt.savefig(f'./ConvexFilter_2_FILTER/case_{cnt}_{flag}_{file_name}.jpg')

        # file_name = file_airfoil[0].split("/")[-1][:-4]
        # plt.savefig(f'./ConvexFilter_2_FILTER/case_{cnt}_{flag}_{file_name}.jpg')

        airfoil_interpolated = airfoil_interpolated.astype(np.float32)
        interpolated_data.append(airfoil_interpolated)

        plt.close()
        cnt += 1
        # if cnt >= 2:
        #     break
        # plt.show()
status = np.array(status)
interpolated_data = np.asarray(interpolated_data)
# np.save('ConvexStatus', status)
np.save('airfoil_interpolated', interpolated_data)
raise

        # TimeValue = np.asarray(mesh_airfoil['TimeValue'])
        # nut = np.asarray(mesh_airfoil['nut'])
        # TimeValue = np.asarray(mesh_airfoil['nut'])
        
        # 'nut', 'p', 'U', 'Normals', 'nut', 'p', 'U', 'Normals', 'Length']
        # print(f"All arrays: {mesh_airfoil.array_names}")



        # pts = mesh.points
        # nut= mesh.point_data['nut']
        # p = mesh.point_data['p']
        # U = mesh.point_data['U']


# params = np.asarray(params)
with open('params.txt', 'w') as f:
    for line in params:
        f.write(f"{line}\n")
        # fig = plt.figure(figsize = (15, 3), dpi=100)
        # plt.rcParams.update({'font.size': 15})
        # plt.subplots_adjust(wspace=1.2, hspace = 0.5, left = 0.05, right=0.95, top=0.95, bottom=0.05)

        # ax = fig.add_subplot(1, 5, 1)
        # im = ax.imshow(nut_interp, cmap='jet')
        # ax.title.set_text(f'nut')
        # ax_divider = make_axes_locatable(ax)
        # cax = ax_divider.append_axes("right", size="7%", pad="2%")
        # cb = fig.colorbar(im, cax=cax)

        # ax = fig.add_subplot(1, 5, 2)
        # im = ax.imshow(p_interp, cmap='jet')
        # ax.title.set_text(f'p')
        # ax_divider = make_axes_locatable(ax)
        # cax = ax_divider.append_axes("right", size="7%", pad="2%")
        # cb = fig.colorbar(im, cax=cax)

        # ax = fig.add_subplot(1, 5, 3)
        # im = ax.imshow(Ux_interp, cmap='jet')
        # ax.title.set_text(f'Ux')
        # ax_divider = make_axes_locatable(ax)
        # cax = ax_divider.append_axes("right", size="7%", pad="2%")
        # cb = fig.colorbar(im, cax=cax)

        # ax = fig.add_subplot(1, 5, 4)
        # im = ax.imshow(Uy_interp, cmap='jet')
        # ax.title.set_text(f'Uy')
        # ax_divider = make_axes_locatable(ax)
        # cax = ax_divider.append_axes("right", size="7%", pad="2%")
        # cb = fig.colorbar(im, cax=cax)

        # ax = fig.add_subplot(1, 5, 5)
        # im = ax.imshow(Uz_interp, cmap='jet')
        # ax.title.set_text(f'Uz')
        # ax_divider = make_axes_locatable(ax)
        # cax = ax_divider.append_axes("right", size="7%", pad="2%")
        # cb = fig.colorbar(im, cax=cax)

        # f_name = file[0].split("\\")[-1][:-4]
        # fig_test_name = f'./{f_name}.jpg'
        # plt.savefig(fig_test_name)



