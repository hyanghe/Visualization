import glob
import meshio
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


rootdir = './Dataset'
subdirs = []

nut_l = []
p_l = []
Ux_l = []
Uy_l = []
Uz_l = []
params = []
cnt = 0
# resolution = 512
resolution = 1024
# resolution = 2048
# resolution = 4096

def plot_sol(nut_interp, p_interp, Ux_interp, Uy_interp, Uz_interp, file):
    fig = plt.figure(figsize = (15, 3), dpi=100)
    plt.rcParams.update({'font.size': 15})
    plt.subplots_adjust(wspace=1.2, hspace = 0.5, left = 0.05, right=0.95, top=0.95, bottom=0.05)

    ax = fig.add_subplot(1, 5, 1)
    im = ax.imshow(nut_interp, cmap='jet')
    ax.title.set_text(f'nut')
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    cb = fig.colorbar(im, cax=cax)

    ax = fig.add_subplot(1, 5, 2)
    im = ax.imshow(p_interp, cmap='jet')
    ax.title.set_text(f'p')
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    cb = fig.colorbar(im, cax=cax)

    ax = fig.add_subplot(1, 5, 3)
    im = ax.imshow(Ux_interp, cmap='jet')
    ax.title.set_text(f'Ux')
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    cb = fig.colorbar(im, cax=cax)

    ax = fig.add_subplot(1, 5, 4)
    im = ax.imshow(Uy_interp, cmap='jet')
    ax.title.set_text(f'Uy')
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    cb = fig.colorbar(im, cax=cax)

    ax = fig.add_subplot(1, 5, 5)
    im = ax.imshow(Uz_interp, cmap='jet')
    ax.title.set_text(f'Uz')
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    cb = fig.colorbar(im, cax=cax)

    f_name = file[0].split("\\")[-1][:-4]
    fig_test_name = f'./interpolated_solution/{f_name}.jpg'
    plt.savefig(fig_test_name)
    plt.close()

for rootdir, dirs, files in os.walk(rootdir):
    if not dirs:
        break
    # ConvexStatus_FINAL = np.load('ConvexStatus_FINAL.npy')
    # dirs_filtered = [dirs[i] for i in range(len(ConvexStatus_FINAL)) if ConvexStatus_FINAL[i]]
    
    dirs_filtered = []
    with open('./dirs_filtered.txt', 'r') as fp:
        for line in fp:
            dirs_filtered.append(line)
    # for subdir in dirs:
    for subdir in dirs_filtered:
        dir = os.path.join(rootdir, subdir).strip()
        file = glob.glob(dir + '/*.vtu')
        mesh = meshio.read(file[0])
        pts = mesh.points
        nut= mesh.point_data['nut']
        p = mesh.point_data['p']
        U = mesh.point_data['U']
        Ux = U[:, 0]
        Uy = U[:, 1]
        Uz = U[:, 2]

        # idx = ((pts[:,0]>=-2) & (pts[:,0]<=4)) & ((pts[:,1]>=-1) & (pts[:,1]<=1))
        # idx = ((pts[:,0]>=-1) & (pts[:,0]<=2)) & ((pts[:,1]>=-0.5) & (pts[:,1]<=0.5))
        # idx = ((pts[:,0]>=-0.5) & (pts[:,0]<=1)) & ((pts[:,1]>=-0.25) & (pts[:,1]<=0.25))
        idx = ((pts[:,0]>=-0.25) & (pts[:,0]<=1)) & ((pts[:,1]>=-0.2) & (pts[:,1]<=0.2))
        pts, nut, p, Ux, Uy, Uz = pts[idx], nut[idx], p[idx], Ux[idx], Uy[idx], Uz[idx]


        
        x_min, x_max = pts[:,0].min(), pts[:,0].max()
        y_min, y_max = pts[:,1].min(), pts[:,1].max()
        xx, yy = np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        xxx, yyy = np.meshgrid(xx, yy)
        grids = np.concatenate((xxx.reshape(-1, 1), yyy.reshape(-1, 1)), axis=-1)


        # nut_interp = griddata(pts[:,:2], nut, grids, fill_value=0.0).reshape(resolution, resolution)
        # p_interp = griddata(pts[:,:2], p, grids, fill_value=0.0).reshape(resolution, resolution)
        Ux_interp = griddata(pts[:,:2], Ux, grids, method='nearest', fill_value=0.0).reshape(resolution, resolution)
        # Uy_interp = griddata(pts[:,:2], Uy, grids, fill_value=0.0).reshape(resolution, resolution)
        # Uz_interp = griddata(pts[:,:2], Uz, grids, fill_value=0.0).reshape(resolution, resolution)

        # plot_sol(nut_interp, p_interp, Ux_interp, Uy_interp, Uz_interp, file)

        Ux_interp = Ux_interp.astype(np.float32)

        fig = plt.figure(figsize = (5, 5), dpi=100)
        plt.rcParams.update({'font.size': 15})
        plt.subplots_adjust(wspace=1.2, hspace = 0.5, left = 0.15, right=0.85, top=0.9, bottom=0.1)

        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(Ux_interp, cmap='jet')
        ax.title.set_text(f'Ux')
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        cb = fig.colorbar(im, cax=cax)

        f_name = file[0].split("/")[-1][:-4]
        fig_test_name = f'./interpolated_solution/{f_name}.jpg'
        plt.savefig(fig_test_name)
        plt.close()

        # nut_l.append(nut_interp)
        # p_l.append(p_interp)
        Ux_l.append(Ux_interp)
        # Uy_l.append(Uy_interp)
        # Uz_l.append(Uz_interp)
        params.append([float(s) for s in subdir.split('_')[2:]])

        print(f'Finished processing case: {cnt}')
        cnt += 1
        # if cnt > 3:
        #     break
# nut_l = np.asarray(nut_l)
# p_l = np.asarray(p_l)
Ux_l = np.asarray(Ux_l)
# Uy_l = np.asarray(Uy_l)
# Uz_l = np.asarray(Uz_l)

# np.save('nut_l', nut_l)
# np.save('p_l', p_l)
np.save('Ux_l', Ux_l)
# np.save('Uy_l', Uy_l)
# np.save('Uz_l', Uz_l)


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





        # # fig = plt.figure(figsize = (18, 5), dpi=100)
        # # plt.rcParams.update({'font.size': 20})
        # # plt.subplots_adjust(wspace=1, hspace = 0.5, left = 0.05, right=0.85, top=0.85, bottom=0.05)

        # # ax = fig.add_subplot(1, 3, 1, projection='3d')
        # # im = ax.scatter(pts[:,0], pts[:,1], pts[:,2],\
        # #     c=nut.flatten(), cmap='jet')
        # # ax.title.set_text(f'nut')
        # # ax.set_box_aspect([np.ptp(i) for i in (pts[:,0], pts[:,1],\
        # #     pts[:,2])])  # equal aspect ratio
        # # cbaxes = fig.add_axes([0.28, 0.25, 0.02, 0.25])
        # # plt.colorbar(im, cax = cbaxes)

        # # ax = fig.add_subplot(1, 3, 2, projection='3d')
        # # im = ax.scatter(pts[:,0], pts[:,1], pts[:,2],\
        # #     c=p.flatten(), cmap='jet')
        # # ax.title.set_text(f'p')
        # # # ax.set_xlim(0, 0.25)
        # # # ax.set_ylim(0, 0.5)
        # # # ax.set_zlim(0, 0.2)
        # # ax.set_box_aspect([np.ptp(i) for i in (pts[:,0], pts[:,1],\
        # #     pts[:,2])])  # equal aspect ratio
        # # cbaxes = fig.add_axes([0.56, 0.25, 0.02, 0.25])
        # # plt.colorbar(im, cax = cbaxes)

        # # ax = fig.add_subplot(1, 3, 3, projection='3d')
        # # im = ax.scatter(pts[:,0], pts[:,1], pts[:,2],\
        # #     c=U[:,0].flatten(), cmap='jet')
        # # # im = ax.scatter(obs_w_pos_step[:,0], obs_w_pos_step[:,1], obs_w_pos_step[:,2],\
        # # #     c='red')
        # # ax.title.set_text(f'U (x)')
        # # # ax.legend(['plate', 'obstcle'], loc = 'lower right', bbox_to_anchor=(1.7, 0.1))
        # # # ax.set_xlim(0, 0.25)
        # # # ax.set_ylim(0, 0.5)
        # # # ax.set_zlim(0, 0.2)
        # # ax.set_box_aspect([np.ptp(i) for i in (pts[:,0], pts[:,1],\
        # #     pts[:,2])])  # equal aspect ratio

        # # # plt.savefig(f'./meshgraphnets/figs/case_{i}_time step_{time_step}_graph_translation_z')
        # # plt.show()
        # # f_name = file[0].split("\\")[-1][:-4]
        # # fig_test_name = f'./{f_name}.jpg'
        # # plt.savefig(fig_test_name)


