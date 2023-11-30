import os
import json
import torch
import pathlib
import argparse
import numpy as np
from tqdm import tqdm

from core.utils import *
from core.create_model import *
from core.datasets.H5Dataset import HDF5Dataset, HDF5Dataset_Test

# Fix random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

parser = argparse.ArgumentParser('Feature_extractor', add_help=False)
parser.add_argument('-c',
                    '--config',
                    default='configs/test/ent_6class_labeled_train.json',
                    type=str)
args = parser.parse_args()

def main(config):
    # GPU Setting
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # Select Model
    print(f"Creating model  : {config['name']}")
    checkpoint = torch.load(config['checkpoint'])
    model = create_model(name=config['name'], num_class = config['num_class']).to(device)
    model.load_state_dict(checkpoint['model'])

    if config['action'] == "train":
        dataset = HDF5Dataset(mode='test', file_path=config['data'])
    else:
        dataset = HDF5Dataset_Test(mode='test', file_path=config['data'])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False,  pin_memory=True, drop_last=False)

    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataloader):
            image = batch[0].to(device).float()
            
            img_name = batch[2][0] # 10_1_85555.png
            vid = img_name[:-10] # 10_1
            frame_num = img_name[-9:].split('.')[0] # 85555.png

            save_path = os.path.join(config['save_path'], vid)
            pathlib.Path(save_path).mkdir(exist_ok=True, parents=True)

            feature = model.extract_feat(image).detach().cpu().numpy()

            save_feature = os.path.join(save_path, frame_num + '.npy')
            np.save(save_feature, feature)

if __name__ == '__main__':
    config = json.load(open(args.config))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
   
    main(config)