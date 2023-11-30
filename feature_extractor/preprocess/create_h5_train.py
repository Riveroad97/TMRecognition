import h5py
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image

data = 'ent_6phase_no_drill_add_back.pkl'
with open(data, 'rb') as f:
    all_data = pickle.load(f)

video_list = list(all_data.keys())
print(video_list[0])

hdf5_path = 'ent_6phase_no_drill_add_back.hdf5'
hf = h5py.File(hdf5_path, 'w', libver='latest')

groups = {}
for vid in tqdm(video_list):
    images = all_data[vid]['path']
    labels = all_data[vid]['label']

    for idx, v in tqdm(enumerate(images)):
        video_name = v.split('/')[-2]
        frame_name = v.split('/')[-1]
        image_name = f'{video_name}_{frame_name}'

        group_name = vid
        lbl = labels[idx]

        if group_name not in groups:
            grp = hf.create_group(group_name)
            groups[group_name] = grp
        else:
            grp = groups[group_name]
        
        img = Image.open(v)
        img_data = img.convert("RGB")
        img_data = img_data.resize((256, 256), Image.BICUBIC)
        img_data = np.array(img_data)

        ds = grp.create_dataset(image_name, data=img_data, compression="gzip", compression_opts=9)
        ds.attrs['class'] = lbl

hf.close()