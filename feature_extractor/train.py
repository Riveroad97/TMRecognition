import os
import json
import torch
import pathlib
import argparse
import numpy as np

from core.utils import *
from core.engine import *
from core.create_model import *
from core.lr_scheduler import *
from core.datasets.H5Dataset import HDF5Dataset


parser = argparse.ArgumentParser('ENT Feature_extractor', add_help=False)
parser.add_argument('-c',
                    '--config',
                    default='configs/train/ent_6class_labeled.json',
                    type=str)
args = parser.parse_args()

# fix random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


def fix_optimizer(optimizer):
    # Optimizer Error fix...!
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()


def main(config):
    # if config['action'] == 'train':
    # GPU Setting
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    

    print("Loading dataset ....")
    
    total_video_list = np.load('data/total_video_list.npy')
    valid_list = ['2_1','3_1', '3_2', '11_1', '11_2','23_1', '23_2']
    train_list = [i for i in total_video_list if i not in valid_list]
    
    train_dataset = HDF5Dataset(mode='train', 
                                file_path=config['data'],
                                blacklist=valid_list)
    valid_dataset = HDF5Dataset(mode='valid',
                                file_path=config['data'],
                                blacklist=train_list)
                                    

    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                              batch_size=config['batch_size'], 
                                              num_workers=config['num_workers'], 
                                              shuffle=True,  pin_memory=True, drop_last=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, 
                                              batch_size=256, 
                                              num_workers=config['num_workers'], 
                                              shuffle=False, pin_memory=True, drop_last=False)

    if config['action'] == "teacher_student":
        # Select Model
        print(f"Loading teacher model  : {config['teacher_name']}")
        print(f"Creating model  : {config['name']}")
        model = create_model(name=config['name'], num_class=config['num_class'])
        
        teacher_checkpoint = torch.load(config['teacher_ckpt'], map_location='cpu')
        teacher_model = create_model(name=config['teacher_name'], num_class=config['num_class'])
        teacher_model.load_state_dict(teacher_checkpoint['model'], strict=False)
        print(model)

        # Multi GPU
        if config['gpu_mode'] == 'DataParallel':
            teacher_model = torch.nn.DataParallel(teacher_model)
            model = torch.nn.DataParallel(model)

            teacher_model.to(device)
            model.to(device)
        elif config['gpu_mode'] == 'Single':
            teacher_model.to(device)
            model.to(device)

        else :
            raise Exception('Error...! gpu_mode')
        
    else:
        # Select Model
        print(f"Creating model  : {config['name']}")
        model = create_model(name=config['name'], num_class=config['num_class'])
        print(model)

        # Multi GPU
        if config['gpu_mode'] == 'DataParallel':
            model = torch.nn.DataParallel(model)
            model.to(device)
        elif config['gpu_mode'] == 'Single':
            model.to(device)

        else :
            raise Exception('Error...! gpu_mode')
        

    # Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW([dict(params=model.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)])
    scheduler = create_scheduler(optimizer=optimizer, config=config)
    

    # Save Weight & Log
    save_path = os.path.join(config['save_path'], 'weight')
    pathlib.Path(save_path).mkdir(exist_ok=True, parents=True)

    log_save_path = os.path.join(config['save_path'], 'logs')
    pathlib.Path(log_save_path).mkdir(exist_ok=True, parents=True)


    # Etc training setting
    print(f"Start training for {config['epochs']} epochs")

    start_epoch = 0

    # Resume
    if config['resume'] == 'on':
        model, optimizer, scheduler, start_epoch = load(ckpt_dir=save_path, model=model, optimizer=optimizer, lr_scheduler=scheduler)
        fix_optimizer(optimizer)


    # Whole LOOP Train & Valid 
    for epoch in range(start_epoch, config['epochs']):
        # Train & Valid
        if config['name'] == 'ENTNet_Labeled':
            train_logs = train_ENTNet_Labeled(model, trainloader, optimizer, device, epoch, config)
            print("Averaged train_stats: ", train_logs)
            valid_logs = valid_ENTNet_Labeled(model, validloader, device, epoch, config)
            print("Averaged valid_stats: ", valid_logs)
        elif config['name'] == 'ENTNet_Soft_Pseudo':
                train_logs = train_ENTNet_Soft_Pseudo(model, teacher_model, trainloader, optimizer, device, epoch, config)
                print("Averaged train_stats: ", train_logs)
                valid_logs = valid_ENTNet_Labeled(model, validloader, device, epoch, config)
                print("Averaged valid_stats: ", valid_logs)
        elif config['name'] == 'ENTNet_Hard_Pseudo':
            train_logs = train_ENTNet_Hard_Pseudo(model, teacher_model, trainloader, optimizer, device, epoch, config)
            print("Averaged train_stats: ", train_logs)
            valid_logs = valid_ENTNet_Labeled(model, validloader, device, epoch, config)
            print("Averaged valid_stats: ", valid_logs)
        
        if epoch % 1 == 0:
            save(ckpt_dir=save_path, model=model, optimizer=optimizer, lr_scheduler=scheduler, epoch=epoch+1, config=config)
        
        log_stats = {**{f'train_{k}': v for k, v in train_logs.items()},
                    **{f'valid_{k}': v for k, v in valid_logs.items()},
                    'epoch': epoch}

        with open(log_save_path +'/log.txt', 'a') as f:
            f.write(json.dumps(log_stats) + '\n')
        
        scheduler.step()


if __name__ == '__main__':
    config = json.load(open(args.config))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
   
    main(config)
    

    


