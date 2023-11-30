import cv2
import h5py
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
from torch.utils.data import Dataset

train_transform = A.Compose([
                    A.ShiftScaleRotate(shift_limit=(-0.1, 0.1), scale_limit=(-0.5, 0.5), rotate_limit=(-180, 180), p=0.4, border_mode=cv2.BORDER_CONSTANT),
                    A.OneOf(
                        [
                            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=1),
                            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.1), contrast_limit=(-0.2, 0.1), p=1),
                        ],
                        p=0.3
                    ),
                    A.OneOf(
                        [
                            A.GaussNoise(p=1, var_limit=(0.0, 26.849998474121094)),
                            A.ISONoise(p=1, intensity=(0.05000000074505806, 0.12999999523162842), color_shift=(0.009999999776482582, 0.26999998092651367)),
                        ],
                        p=0.3
                    ),
                    A.OneOf(
                        [
                            A.MedianBlur(blur_limit=5, p=1),
                            A.MotionBlur(blur_limit=5, p=1)
                            ],
                            p=0.3,
                            ),
                    A.HorizontalFlip(p=0.4),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                            ])


test_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()])


class HDF5Dataset(Dataset):
    def __init__(self, file_path, mode, blacklist=[]):

        self.file_path = file_path
        self.mode = mode
        self.length = None
        self._idx_to_name = {}
        self.blacklist = blacklist

        self.transform = train_transform if mode == 'train' else test_transform

        with h5py.File(self.file_path, 'r') as hf:
            for gname, group in hf.items():
                if gname not in self.blacklist:
                    start_idx = self.length if self.length else 0
                    self.length = start_idx + len(group)
                    for i, meta in enumerate(group.items()):
                        self._idx_to_name[start_idx + i] = meta[0]

        print(f'{mode}dataset: Load dataset with {self.__len__()} images.')

    def __len__(self):
        assert self.length is not None
        return self.length

    def _open_hdf5(self):
        self._hf = h5py.File(self.file_path, 'r')

    def __getitem__(self, index):
        if not hasattr(self, '_hf'):
            self._open_hdf5()

        assert self._idx_to_name is not None
        img_name = self._idx_to_name[index]

        video_name_list = img_name.split('_')
        if len(video_name_list) == 3:
            video_name = video_name_list[0] + '_' + video_name_list[1]
        else:
            video_name = video_name_list[0]

        ds = self._hf[video_name][img_name]
        image = ds[()]
        label = torch.tensor(ds.attrs['class'])

        image = self.transform(image=image)['image']

        return image, label, img_name
    
    
class HDF5Dataset_Test(Dataset):
    def __init__(self, file_path, mode):

        self.file_path = file_path
        self.mode = mode
        self.length = None
        self._idx_to_name = {}

        self.transform = train_transform if mode == 'train' else test_transform

        with h5py.File(self.file_path, 'r') as hf:
            for gname, group in hf.items():
                if gname == mode:
                    self.length = len(group)
                    for i, meta in enumerate(group.items()):
                        self._idx_to_name[i] = meta[0]

        print(f'{mode}dataset: Load dataset with {self.__len__()} images.')

    def __len__(self):
        assert self.length is not None
        return self.length

    def _open_hdf5(self):
        self._hf = h5py.File(self.file_path, 'r')

    def __getitem__(self, index):
        if not hasattr(self, '_hf'):
            self._open_hdf5()

        assert self._idx_to_name is not None
        img_name = self._idx_to_name[index]

        ds = self._hf[self.mode][img_name]
        image = ds[()]
    
        image = self.transform(image=image)['image']

        return image, img_name