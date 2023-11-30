import glob
import h5py
import natsort
import numpy as np
from tqdm import tqdm
from PIL import Image

root_dir = '/workspace/ENT/Recognition/Frame'

test_image = natsort.natsorted(glob.glob(root_dir + '/video_frame_test/*/*.png'))
print(len(test_image))

hdf5_path = 'ent_5phase_test.hdf5'
hf = h5py.File(hdf5_path, 'w', libver='latest')

groups = {}

for idx, v in tqdm(enumerate(test_image)):
    video_name = v.split('/')[-2]
    frame_name = v.split('/')[-1]
    image_name = f'{video_name}_{frame_name}'
    dataset_name = 'test'

    if dataset_name not in groups:
        grp = hf.create_group(dataset_name)
        groups[dataset_name] = grp
    else:
        grp = groups[dataset_name]
    
    img = Image.open(v)
    img_data = img.convert("RGB")
    img_data = img_data.resize((256, 256), Image.BICUBIC)
    img_data = np.array(img_data)

    ds = grp.create_dataset(image_name, data=img_data, compression="gzip", compression_opts=9)
hf.close()