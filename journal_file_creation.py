import os
import shutil

###### The following is for data interpolation
# # src_file = os.getcwd() + '\\' + 'run_fluent_new_v03_fluent.jou'
# # tgt_file = os.getcwd() + '\\' + 'run_fluent_new_v04_fluent.jou'
#
# src_file = os.getcwd() + '\\' + 'run_fluent_new_v03_mlsolver.jou'
# tgt_file = os.getcwd() + '\\' + 'run_fluent_new_v04_mlsolver.jou'
#
#
# shutil.copy(src_file, tgt_file)
# journal_file = tgt_file
#
# for i in range(2, 101):
#     with open(journal_file, 'r') as journal:
#         lines_j = journal.readlines()
#         lines_new = lines_j[:2]
#         search_key = '1'
#         idx_l2 = lines_new[0].index(search_key)
#         lines_new[0] = lines_new[0][:idx_l2] + str(i) + lines_new[0][idx_l2 + len(search_key):]
#         idx_l5 = lines_new[1].index(search_key)
#         lines_new[1] = lines_new[1][:idx_l5] + str(i) + lines_new[1][idx_l5 + len(search_key):]
#     with open(tgt_file, 'a') as jour:
#         jour.write('\n')
#         # for line_new in lines_new[1:28]:
#         for line_new in lines_new:
#             jour.write(line_new)


### The following is for image generation for making videos
src_file = os.getcwd() + '\\' + 'image_generation_fluent.jou'
tgt_file = os.getcwd() + '\\' + 'image_generation_fluent_v01.jou'

# src_file = os.getcwd() + '\\' + 'image_generation_mlsolver.jou'
# tgt_file = os.getcwd() + '\\' + 'image_generation_mlsolver_v01.jou'

shutil.copy(src_file, tgt_file)
journal_file = tgt_file

for i in range(2, 51):
    with open(journal_file, 'r') as journal:
        lines_j = journal.readlines()
        lines_new = lines_j[:19]
        lines_exclude = [2, 6, 7, 8]
        lines_new = [l for idx, l in enumerate(lines_new) if idx not in lines_exclude]
        search_key = '1'
        idx_l2 = lines_new[3].index(search_key)
        lines_new[3] = lines_new[3][:idx_l2] + str(i) + lines_new[3][idx_l2 + len(search_key):]
        idx_l5 = lines_new[13].index(search_key)
        lines_new[13] = lines_new[13][:idx_l5] + str(i) + lines_new[13][idx_l5 + len(search_key):]
    with open(tgt_file, 'a') as jour:
        jour.write('\n')
        # for line_new in lines_new[1:28]:
        for line_new in lines_new:
            jour.write(line_new)
