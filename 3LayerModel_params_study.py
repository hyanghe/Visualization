from zipfile import ZipFile
# from SALib.sample import sobol_sequence
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import shutil
import re
import math
import time
import pandas as pd
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

def get_all_file_paths(directory):
    file_paths = []
    file_directories = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
        file_directories.append(directories)
    return file_paths, file_directories

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False


class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs

# study_focus = 'HTC'
# trial = f"{study_focus}_Trend_study"

# params = OrderedDict(
#     HTC = [1e-7, 5e-7, 1e-6, 5e-6, 5e-5],
#     Die_xy = [4000],
#     Die_z = [20, 50, 100, 150, 200],
#     t_Diel = [1.0, 3.0, 6.0, 8.0, 10.0],
#     t_Insulator = [0.01, 0.02, 0.03, 0.04, 0.05],
#     # K_Diel = [0.00138, 0.0138, 0.138]
# )

# params = OrderedDict(
#     HTC = [1e-7, 5e-7, 1e-6, 5e-6, 5e-5],
#     Die_xy = [4000],
#     Die_z = [100],
#     t_Diel = [6.0],
#     t_Insulator = [0.05],
# )


params = OrderedDict(
    HTC = [1e-6],
    Die_xy = [4000],
    Die_z = [100],
    t_Diel = [6.0],
    t_Insulator = [0.05],
)

study_params = ['HTC', 'Die_z', 't_Diel', 't_Insulator']
# study_params = ['t_Diel', 't_Insulator']

def main():
    # path to folder which needs to be zipped
    # directory = './One_tile_3D_data'
    # directory = './One_tile_3D_data_grid_size_one_batch_run_51200'
    directory = f'./One_tile_3D_data_cluster_v01'
    for study_focus in study_params:
        params = OrderedDict(
            HTC=[1e-6],
            Die_xy=[4000],
            Die_z=[100],
            t_Diel=[6.0],
            t_Insulator=[0.05],
        )
        if study_focus == 'HTC':
            params[study_focus] = [1e-7, 5e-7, 1e-6, 5e-6, 5e-5]
        elif study_focus == 'Die_z':
            params[study_focus] = [20, 50, 100, 150, 200]
        elif study_focus == 't_Diel':
            params[study_focus] = [1.0, 3.0, 6.0, 8.0, 10.0]
        elif study_focus == 't_Insulator':
            params[study_focus] = [0.01, 0.02, 0.03, 0.04, 0.05]
        # elif  study_param == 'K_Diel':
        #     params[study_param] =[0.00138, 0.0138, 0.138]

        run_blder = RunBuilder()
        RUNS_5 = run_blder.get_runs(params)

        T_max_ls = []
        T_min_ls = []
        deltaT_ls = []

        save_data_path = f'./Param_study/{study_focus}/'
        os.makedirs(os.path.dirname(save_data_path), exist_ok=True, mode=0o777)

        file_paths, file_dir = get_all_file_paths(directory)
        # print("The length of file_paths is: ", len(file_paths))
        file_directories = file_dir[0]

        filtered_folders = []
        for run_item in RUNS_5:
            Die_xy = run_item.Die_xy
            HTC = run_item.HTC
            Die_z = run_item.Die_z
            t_Diel = run_item.t_Diel
            t_Insulator = run_item.t_Insulator
            search_dir = f"HTC_{HTC}_xy_{Die_xy}_z_{Die_z}_tDiel_{t_Diel}_tIns_{t_Insulator}"
            filtered_folders.append(search_dir)



        counter = 0
        # for folder in file_directories:
        plt.subplots_adjust(left=0.25, bottom=0.05, right=0.85, top=0.85, wspace=0.01, hspace=0.01)
        fig, ax = plt.subplots(2, len(filtered_folders), figsize=(5*len(filtered_folders)-10, 5))

        gs = ax[1, 0].get_gridspec()

        for idx_exp, folder in enumerate(filtered_folders):
            match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
            temp_str = re.findall(match_number, folder.split()[0])
            # float(re.findall(match_number, folder.split()[0])[0])
            temp_dict = {'HTC': float(temp_str[0]), 'Die_z': int(temp_str[2]), 't_Diel': float(temp_str[3]), 't_Insulator': float(temp_str[4])}

            source_dir = directory + '/' + folder + '/'
            # txt_name = "commands_static_grid_power.txt"
            # source_file_name = source_dir + txt_name
            # f = open(source_file_name, 'r')
            # lines = f.readlines()

            # shift = 12
            # Die_xy = int(re.findall("\d+", lines[10+shift].split()[0])[0])
            # Die_z = int(re.findall("\d+", lines[12+shift].split()[0])[0])
            # # C_scale = int(re.findall("\d+", lines[43].split()[0])[0])
            # match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
            # t_Insulator = float(re.findall(match_number, lines[13+shift].split()[0])[0])
            # t_Diel = float(re.findall(match_number, lines[14+shift].split()[0])[0])
            # K_Diel = float(re.findall(match_number, lines[52+shift].split()[1])[0])
            # top_HTC = float(re.findall(match_number, lines[65+shift].split()[0])[0])
            # bottom_HTC = float(re.findall(match_number, lines[66+shift].split()[0])[0])
            # HTC = top_HTC

            power_map_idx = int(re.findall("\d+", folder.split()[0])[0])
            # target_folder_name = save_data_path + f"Powermap_{power_map_idx}/"


            data_path = source_dir + 'Nodal_T_heat.txt'
            os.makedirs(save_data_path + folder, mode=0o777, exist_ok=True)
            shutil.copy(data_path, save_data_path + folder + '/Nodal_T_heat.txt')

            f = open(data_path, 'r')
            lines = f.readlines()
            lines[0] = 'node, x, y, z, T' + '\n'
            f.close()
            f = open(data_path, 'w')
            f.writelines(lines)
            f.close()

            df = pd.read_csv(data_path)
            df.columns = df.columns.str.replace(' ', '')
            x_coord = df['x']
            y_coord = df['y']
            x_coord = np.asarray(x_coord).reshape(-1, 1)
            y_coord = np.asarray(y_coord).reshape(-1, 1)
            pts = np.hstack((x_coord, y_coord))
            T = df['T']
            T = np.asarray(T)
            resolution = 80
            x, y = np.linspace(x_coord.min(), x_coord.max(), resolution), np.linspace(y_coord.min(), y_coord.max(), resolution)
            xx, yy = np.meshgrid(x, y)
            xx, yy = xx.reshape(-1, 1), yy.reshape(-1, 1)
            pts_interp = np.hstack((xx, yy))
            T_interp = griddata(pts, T, pts_interp).reshape(resolution, resolution)

            # # plt.axis('off')
            # pwr_block1 = np.load(source_dir + "power_map.npy")
            # # pwr_block1 = ndimage.rotate(pwr_block1, 90)
            # pwr_block1 = np.transpose(pwr_block1)
            # ax.set_title(f'Power map (total power: {pwr_block1.sum():.2f} mW)')
            # im = ax[].imshow(pwr_block1, vmin=pwr_block1.min(), vmax=pwr_block1.max(), origin='lower', cmap='jet')
            # # im = ax.scatter(x_coord, y_coord, c=T, cmap='jet')
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="7%", pad="2%")
            # cb = fig.colorbar(im, cax=cax)
            #

            im = ax[0][idx_exp].imshow(T_interp, cmap='jet', origin='lower')
            ax[0][idx_exp].set_title(f'{study_focus}: {temp_dict[study_focus]}, deltaT: {T_interp.max()-T_interp.min():.2f}, \n Tmax: {T_interp.max():.2f}, Tmin: {T_interp.min():.2f}')
            T_max_ls.append(T_interp.max())
            T_min_ls.append(T_interp.min())
            deltaT_ls.append(T_interp.max()-T_interp.min())
            divider = make_axes_locatable(ax[0][idx_exp])
            cax = divider.append_axes("right", size="7%", pad="2%")
            cb = fig.colorbar(im, cax=cax)
            # plt.close()
        for axis in ax[1,:]:
            axis.remove()
        axbig = fig.add_subplot(gs[1,1:])
        axbig.plot(params[study_focus], T_max_ls, 'r.-')
        axbig.plot(params[study_focus], T_min_ls, 'go-')
        axbig.plot(params[study_focus], deltaT_ls, 'b*-')
        axbig.legend(['Tmax', 'Tmin', 'deltaT'])
        axbig.set_xlabel(f'{study_focus}')
        axbig.set_ylabel(f'Temperature (C)')
        # axbig.annotate(f'{study_focus}', (-0.15, 0.5), xycoords='axes fraction', va='center', fontsize=15)
        fig.tight_layout()
        # plt.show()
        plt.savefig(save_data_path + f'{study_focus}_influence.jpg')
        plt.close()

        # data_path = source_dir + 'Nodal_T_all.txt'
        # f = open(data_path, 'r')
        # lines = f.readlines()
        # lines[0] = 'node, x, y, z, T' + '\n'
        # f.close()
        # f = open(data_path, 'w')
        # f.writelines(lines)
        # f.close()

        # df = pd.read_csv(data_path)
        # df.columns = df.columns.str.replace(' ', '')
        # x_coord = df['x']
        # y_coord = df['y']
        # z_coord = df['z']
        # x_coord = np.asarray(x_coord).reshape(-1, 1) - x_coord.min()
        # y_coord = np.asarray(y_coord).reshape(-1, 1) - y_coord.min()
        # z_coord = np.asarray(z_coord).reshape(-1, 1) - z_coord.min()
        # pts = np.hstack((x_coord, y_coord, z_coord))
        # T = df['T']
        # T = np.asarray(T)
        # resolution = 80
        # x, y, z = np.linspace(x_coord.min(), x_coord.max(), resolution),\
        #           np.linspace(y_coord.min(), y_coord.max(), resolution), \
        #           np.linspace(z_coord.min(), z_coord.max(), resolution//16)
        # xx, yy, zz = np.meshgrid(x, y, z)
        # xx, yy, zz = xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)
        #
        # pts_interp = np.hstack((xx, yy, zz))
        # T_interp = griddata(pts, T, pts_interp).reshape(resolution, resolution, resolution)
        #
        # fig = plt.figure(figsize=(10, 5))
        # plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.85, wspace=0.2, hspace=0.3)
        #
        # ax = fig.add_subplot(121)
        # # plt.axis('off')
        # pwr_block1 = np.load(temp_working_dir + "power_map.npy")
        # # pwr_block1 = ndimage.rotate(pwr_block1, 90)
        # pwr_block1 = np.transpose(pwr_block1)
        # ax.set_title(f'Power map (total power: {pwr_block1.sum():.2f} mW)')
        # im = ax.imshow(pwr_block1, vmin=pwr_block1.min(), vmax=pwr_block1.max(), origin='lower', cmap='jet')
        # # im = ax.scatter(x_coord, y_coord, c=T, cmap='jet')
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="7%", pad="2%")
        # cb = fig.colorbar(im, cax=cax)
        #
        # ax = fig.add_subplot(122, projection='3d')
        # # plt.axis('on')
        # T_interp = T_interp
        # # im = ax.imshow(T_interp, cmap='jet', origin='lower')
        # im = ax.scatter(xx.flatten(), yy.flatten(), zz.flatten(), c=T_interp.flatten(), alpha=0.5, cmap='jet', origin='lower')
        # ax.set_title(f'Interpolated T (Tmax: {T_interp.max():.2f}, Tmin: {T_interp.min():.2f}, deltaT: {T_interp.max()-T_interp.min():.2f})')
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="7%", pad="2%")
        # cb = fig.colorbar(im, cax=cax)
        # plt.savefig(temp_working_dir + 'Data_visual.jpg')
        # # plt.show()
        # plt.close()


if __name__ == "__main__":
    main()
    print("Done")
    print("Done")