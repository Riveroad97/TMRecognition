import os
import natsort
import pickle
import numpy as np
import random
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

root_dir = '/workspace/ENT/Recognition'
img_dir = os.path.join(root_dir, 'Frame/video_frame_train')
phase_dir = os.path.join(root_dir, 'Label/frame_label_train')

print(root_dir)
print(img_dir)
print(phase_dir)

#cortical=====================
def get_dirs(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names = natsort.natsorted(file_names)
    file_paths = natsort.natsorted(file_paths)
    return file_names, file_paths


def get_files(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if not os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names = natsort.natsorted(file_names)
    file_paths = natsort.natsorted(file_paths)
    return file_names, file_paths
#cortical=====================


#cortical=====================
img_dir_names, img_dir_paths = get_dirs(img_dir)
phase_file_names, phase_file_paths = get_files(phase_dir)

phase_dict = {}
phase_dict_key = ['Cortical_drilling', 'Drilling_of_the_mastoid', 'Drilling_the_antrum', 'Drilling_near_the_facial_nerve', 'Posterior_tympanotomy', 'no_drill']

for i in range(len(phase_dict_key)):
    phase_dict[phase_dict_key[i]] = i
print(phase_dict)
#cortical=====================


#cortical=====================
all_info_all = []

for j in range(len(phase_file_names)):

    phase_file = open(phase_file_paths[j])

    video_name = phase_file_paths[j].split('/')[-1].split('.')[0]
    
    print("video_file_name:", video_name)

    info_all = []
    first_line = True
    for phase_line in phase_file:
        phase_split = phase_line.split()
        if first_line:
            first_line = False
            continue
        info_each = []
        if phase_split[1] == 'background':
            continue
        else:
            img_file_each_path = os.path.join(img_dir_paths[j], str(phase_split[0]).zfill(5) + '.png')
            info_each.append(img_file_each_path)
            info_each.append(phase_dict[phase_split[1]])
        info_all.append(info_each)
    all_info_all.append(info_all)
    #cortical=====================

total_dict = {}

cor_len = 0
mas_len = 0
ant_len = 0
fac_len = 0
tym_len = 0
ndr_len = 0

for i in tqdm(range(len(img_dir_names))):
    img_dir_name = img_dir_names[i]
    dir_dict = {img_dir_name : {'path':[], 'label' : []}}
    
    dir_info = all_info_all[i]
    cortical_idx = []
    mastoid_idx = []
    antrum_idx = []
    facial_idx = []
    tympanotomy_idx = []
    nodrill_idx = []

    for j in range(len(all_info_all[i])):
        if all_info_all[i][j][1] == 0:
            cortical_idx.append(j)
        elif all_info_all[i][j][1] == 1:
            mastoid_idx.append(j)
        elif all_info_all[i][j][1] == 2:
            antrum_idx.append(j)
        elif all_info_all[i][j][1] == 3:
            facial_idx.append(j)
        elif all_info_all[i][j][1] == 4:
            tympanotomy_idx.append(j)
        elif all_info_all[i][j][1] == 5:
            nodrill_idx.append(j)
        

    class_len = [len(cortical_idx), len(mastoid_idx), len(antrum_idx), len(facial_idx), len(tympanotomy_idx), len(nodrill_idx)]

    cor_len += len(cortical_idx)
    mas_len += len(mastoid_idx)
    ant_len += len(antrum_idx)
    fac_len += len(facial_idx)
    tym_len += len(tympanotomy_idx)
    ndr_len += len(nodrill_idx)

    cortical_path = [dir_info[i][0] for i in cortical_idx]
    mastoid_path = [dir_info[i][0] for i in mastoid_idx]
    antrum_path = [dir_info[i][0] for i in antrum_idx]
    facial_path = [dir_info[i][0] for i in facial_idx]
    tympanotomy_path = [dir_info[i][0] for i in tympanotomy_idx]
    nodrill_path = [dir_info[i][0] for i in nodrill_idx]
    
    cortical_label = [dir_info[i][1] for i in cortical_idx]
    mastoid_label = [dir_info[i][1] for i in mastoid_idx]
    antrum_label = [dir_info[i][1] for i in antrum_idx]
    facial_label = [dir_info[i][1] for i in facial_idx]
    tympanotomy_label = [dir_info[i][1] for i in tympanotomy_idx]
    nodrill_label = [dir_info[i][1] for i in nodrill_idx]

    total_path = cortical_path + mastoid_path + antrum_path + facial_path + tympanotomy_path + nodrill_path
    total_label = cortical_label + mastoid_label + antrum_label + facial_label + tympanotomy_label + nodrill_label
        
    dir_dict[img_dir_name]['path'] = total_path
    dir_dict[img_dir_name]['label']= total_label

    total_dict.update(dir_dict)

print(cor_len, mas_len, ant_len, fac_len, tym_len, ndr_len)
with open('ent_6phase_no_drill_add.pkl', 'wb') as f:
    pickle.dump(total_dict, f)