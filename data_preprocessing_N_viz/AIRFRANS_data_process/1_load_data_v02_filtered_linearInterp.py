import glob
import meshio
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from tqdm import tqdm

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
subdomain = 32
# subdomain = 64
# subdomain = 128
# subdomain = 256
# subdomain = 512
num_patches = resolution//subdomain
# num_ticks = num_patches+1
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
        print('x_min, x_max: ', x_min, x_max)
        print('y_min, y_max: ', y_min, y_max)
        print('numpy.rint')
        print('x_min, x_max: ', round(x_min, 2), round(x_max, 2))
        print('y_min, y_max: ', round(y_min, 2), round(y_max, 2))
        x_patch_size = (round(x_max, 2) - round(x_min, 2))/num_patches
        x_patch_div_mn = np.arange(round(x_min, 2), round(x_max, 2), x_patch_size)
        x_patch_div_mx = np.arange(round(x_min, 2)+x_patch_size,\
         round(x_max, 2)+x_patch_size, x_patch_size)
        print('x_patch_div_mn, x_patch_div_mx', x_patch_div_mn.shape, x_patch_div_mx.shape)
        x_patch_div = [[mn, mx] for mn, mx in zip(x_patch_div_mn, x_patch_div_mx)]

        y_patch_size = (round(y_max, 2) - round(y_min, 2))/num_patches
        y_patch_div_mn = np.arange(round(y_min, 2), round(y_max, 2), y_patch_size)
        y_patch_div_mx = np.arange(round(y_min, 2)+y_patch_size,\
         round(y_max, 2)+y_patch_size, y_patch_size)
        print('y_patch_div_mn, y_patch_div_mx', y_patch_div_mn.shape, y_patch_div_mx.shape)
        y_patch_div = [[mn, mx] for mn, mx in zip(y_patch_div_mn, y_patch_div_mx)]

        idx_2_pts = {}
        idx_2_ptsSol = {}
        idx_2_grids = {}
        from collections import defaultdict
        idx_2_neigh = defaultdict(list)
        xx, yy = np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        xxx, yyy = np.meshgrid(xx, yy)
        grids = np.concatenate((xxx.reshape(-1, 1), yyy.reshape(-1, 1)), axis=-1)
        moves = [0, 1, 0, -1, 0]
        for i in tqdm(range(resolution//subdomain)):
            for j in tqdm(range(resolution//subdomain)):
                x_begin, x_end = x_patch_div[i]
                y_begin, y_end = y_patch_div[j]

                idx_pts = (x_begin<=pts[:,0]) & (pts[:,0]<=x_end) &\
                 (y_begin<=pts[:,1]) & (pts[:,1]<=y_end)
                idx_2_pts[(i, j)] = pts[idx_pts, :2]
                idx_2_ptsSol[(i, j)] = Ux[idx_pts]

                idx_grids = (x_begin<=grids[:,0]) & (grids[:,0]<=x_end) &\
                 (y_begin<=grids[:,1]) & (grids[:,1]<=y_end)
                idx_2_grids[(i, j)] = grids[idx_grids, :2]
                idx_2_neigh[(i, j)].append([i, j])
                for move_idx in range(4):
                    i_new = i + moves[move_idx]
                    j_new = j + moves[move_idx+1]
                    if 0<=i_new<resolution//subdomain and 0<=j_new<resolution//subdomain:
                        idx_2_neigh[(i, j)].append([i_new, j_new])

                # for i_new, j_new in [(i-1, j-1), (i-1,j+1), (i+1,j-1), (i+1,j+1)]:
                #     if 0<=i_new<resolution//subdomain and 0<=j_new<resolution//subdomain:
                #         idx_2_neigh[(i, j)].append([i_new, j_new])


        idx_2_interpSol = {}
        for i in tqdm(range(resolution//subdomain)):
            for j in tqdm(range(resolution//subdomain)):
                key = (i, j)
                neighbors = idx_2_neigh[key]
                neigh_pts = []
                neigh_sol = []
                for neighbor in neighbors:
                    neigh_pts.extend(idx_2_pts[key])
                    neigh_sol.extend(idx_2_ptsSol[key])
                neigh_pts = np.asarray(neigh_pts)
                neigh_sol = np.asarray(neigh_sol)
                struct_grid = idx_2_grids[key]
                # print('neigh_pts, neigh_sol', neigh_pts.shape, neigh_sol.shape)
                # print('struct_grid shape: ', struct_grid.shape)
                if len(neigh_pts)==0:
                    interpS = np.zeros((subdomain, subdomain)).astype(np.float32)
                else:
                    # print('neigh_pts, neigh_sol', neigh_pts.shape, neigh_sol.shape)
                    # print('struct_grid shape: ', struct_grid.shape)
                    # interpS = griddata(neigh_pts, neigh_sol, struct_grid,method='linear', fill_value=0.0).reshape(subdomain, subdomain)
                    interpS = griddata(neigh_pts, neigh_sol, struct_grid,method='nearest', fill_value=0.0).reshape(subdomain, subdomain)

                idx_2_interpSol[key] = interpS
                # print('neigh_pts, neigh_sol', neigh_pts.shape, neigh_sol.shape)
                # print('struct_grid shape: ', struct_grid.shape)
                # print('interpS shape: ', interpS.shape)
                # raise
                
        # idx_2_pts
        # idx_2_ptsSol
        # idx_2_grids
        # idx_2_neigh
        Ux_interp_linear = np.zeros_like(xxx)
        for i in range(resolution//subdomain):
            for j in range(resolution//subdomain):
                patch_xS = (resolution//subdomain-i-1)*subdomain
                patch_xE = (resolution//subdomain-i)*subdomain
                # print('patch_xS, patch_xE: ', patch_xS, patch_xE)
                patch_yS = j*subdomain
                patch_yE = (j+1)*subdomain
                # Ux_interp_linear[patch_xS:patch_xE, patch_yS:patch_yE] = idx_2_interpSol[(i,j)]
                Ux_interp_linear[patch_xS:patch_xE, patch_yS:patch_yE] = \
                np.rot90(idx_2_interpSol[(i,j)], k=1, axes=(0, 1))
                # print('patch_yS, patch_yE: ', patch_yS, patch_yE)
                # print(Ux_interp[patch_xS:patch_xE, patch_yS:patch_yE].shape)

            #     raise
            # import pdb; pdb.set_trace()
        # raise
        Ux_interp_linear = np.rot90(Ux_interp_linear, 3)

        

        # xx, yy = np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        # xxx, yyy = np.meshgrid(xx, yy)
        # grids = np.concatenate((xxx.reshape(-1, 1), yyy.reshape(-1, 1)), axis=-1)


        # nut_interp = griddata(pts[:,:2], nut, grids, fill_value=0.0).reshape(resolution, resolution)
        # p_interp = griddata(pts[:,:2], p, grids, fill_value=0.0).reshape(resolution, resolution)
        Ux_interp = griddata(pts[:,:2], Ux, grids, method='nearest', fill_value=0.0).reshape(resolution, resolution)
        # Uy_interp = griddata(pts[:,:2], Uy, grids, fill_value=0.0).reshape(resolution, resolution)
        # Uz_interp = griddata(pts[:,:2], Uz, grids, fill_value=0.0).reshape(resolution, resolution)

        # plot_sol(nut_interp, p_interp, Ux_interp, Uy_interp, Uz_interp, file)

        Ux_interp = Ux_interp.astype(np.float32)

        Ux_interp_linear = Ux_interp_linear.astype(np.float32)

        fig = plt.figure(figsize = (12, 5), dpi=100)
        plt.rcParams.update({'font.size': 15})
        plt.subplots_adjust(wspace=1.2, hspace = 0.5, left = 0.15, right=0.85, top=0.9, bottom=0.1)

        ax = fig.add_subplot(1, 3, 1)
        im = ax.imshow(Ux_interp, cmap='jet', origin='lower')
        ax.title.set_text(f'Original interp(Ux)')
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        cb = fig.colorbar(im, cax=cax)

        ax = fig.add_subplot(1, 3, 2)
        im = ax.imshow(Ux_interp_linear, cmap='jet', origin='lower')
        ax.title.set_text(f'New interp(Ux)')
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        cb = fig.colorbar(im, cax=cax)

        ax = fig.add_subplot(1, 3, 3)
        im = ax.imshow(Ux_interp_linear-Ux_interp, cmap='jet', origin='lower')
        ax.title.set_text(f'Difference(Ux)')
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        cb = fig.colorbar(im, cax=cax)

        f_name = file[0].split("/")[-1][:-4]
        fig_test_name = f'./interpolated_solution_new/{f_name}.jpg'
        plt.savefig(fig_test_name)
        plt.close()
        # raise
        # nut_l.append(nut_interp)
        # p_l.append(p_interp)
        # Ux_l.append(Ux_interp)
        # Uy_l.append(Uy_interp)
        # Uz_l.append(Uz_interp)
        
        Ux_l.append(Ux_interp_linear)
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


