import os
import numpy as np

from torch.utils.data import Dataset, DataLoader

phase2label_dicts = {
    'ent6':{
    'Cortical_drilling':0,
    'Drilling_of_the_mastoid':1,
    'Drilling_the_antrum':2,
    'Drilling_near_the_facial_nerve':3,
    'Posterior_tympanotomy':4,
    'no_drill':5,
    'background':6
    },
    'ent5':{
    'Cortical_drilling':0,
    'Drilling_of_the_mastoid':1,
    'Drilling_the_antrum':2,
    'Drilling_near_the_facial_nerve':3,
    'Posterior_tympanotomy':4
    },
    
}

def phase2label(phases, phase2label_dict):
    labels = [phase2label_dict[phase] if phase in phase2label_dict.keys() else 1 for phase in phases] # 나머지 것들은 1로 처리
    return labels

def label2phase(labels, phase2label_dict):
    label2phase_dict = {phase2label_dict[k]:k for k in phase2label_dict.keys()}
    phases = [label2phase_dict[label] if label in label2phase_dict.keys() else 'HardFrame' for label in labels]
    return phases


class TestVideoDataset(Dataset):
    def __init__(self, dataset, root, sample_rate, name, mode):
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.videos = []
        self.labels = []

        self.video_names = []
        if name == '':
            name = ''
        else:
            name = f'_{name}'
        video_feature_folder = os.path.join(root, 'Feature/video_feature' + f'{name}') + f'/{mode}'
        label_folder = os.path.join(root, 'Label/frame_label')


        for v_f in os.listdir(video_feature_folder):
            v_f_abs_path = os.path.join(video_feature_folder, v_f)
            v_label_file_abs_path = os.path.join(label_folder, v_f.split('.')[0] + '.txt')
            
            labels = self.read_labels(v_label_file_abs_path) 
            labels = labels[::sample_rate]
            videos = np.load(v_f_abs_path)[::sample_rate,]

            self.videos.append(videos)
            self.labels.append(labels)

            self.video_names.append(v_f)
       
        print('VideoDataset: Load dataset {} with {} videos.'.format(self.dataset, self.__len__()))
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, item):
        video, label, video_name = self.videos[item], self.labels[item],  self.video_names[item]
        return video, label, video_name
    
    def read_labels(self, label_file):
        with open(label_file,'r') as f:
            phases = [line.strip().split('\t')[1] for line in f.readlines()][1:] # 첫번째 줄 날림
            labels = phase2label(phases, phase2label_dicts[self.dataset])
        return labels
    
    
class TestVideoDataset_5class(Dataset):
    def __init__(self, dataset, root, sample_rate, name, mode):
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.videos = []
        self.labels = []
        ###      
        self.video_names = []
        if name == '':
            name = ''
        else:
            name = f'_{name}'
        video_feature_folder = os.path.join(root, 'Feature/video_feature' + f'{name}') + f'/{mode}'
        label_folder = os.path.join(root, 'Label/frame_label_5class')
       
        for v_f in os.listdir(video_feature_folder):
            v_f_abs_path = os.path.join(video_feature_folder, v_f)
           
            v_label_file_abs_path = os.path.join(label_folder, v_f.split('.')[0] + '.npy')
         
            labels = np.load(v_label_file_abs_path, allow_pickle=True).tolist()
            labels = labels[::sample_rate]

            videos = np.load(v_f_abs_path)[::sample_rate,]
           
            if len(labels) != videos.shape[0]:
                print(v_label_file_abs_path)

            self.videos.append(videos)
           
            self.labels.append(labels)
            phase = 1
            for i in range(len(labels)-1):
                if labels[i] == labels[i+1]:
                    continue
                else:
                    phase += 1
          
            self.video_names.append(v_f)
       
        print('VideoDataset: Load dataset {} with {} videos.'.format(self.dataset, self.__len__()))
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, item):
        video, label, video_name = self.videos[item], self.labels[item], self.video_names[item]
        return video, label, video_name
    

class TestVideoDataset_6class(Dataset):
    def __init__(self, dataset, root, sample_rate, name, mode):
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.videos = []
        self.labels = []
        ###      
        self.video_names = []
        if name == '':
            name = ''
        else:
            name = f'_{name}'
        video_feature_folder = os.path.join(root, 'Feature/video_feature' + f'{name}') + f'/{mode}'
        label_folder = os.path.join(root, 'Label/frame_label_6class')

        for v_f in os.listdir(video_feature_folder):
            v_f_abs_path = os.path.join(video_feature_folder, v_f)
           
            v_label_file_abs_path = os.path.join(label_folder, v_f.split('.')[0] + '.npy')
         
            labels = np.load(v_label_file_abs_path, allow_pickle=True).tolist()
            labels = labels[::sample_rate]
           
            videos = np.load(v_f_abs_path)[::sample_rate,]
           
            if len(labels) != videos.shape[0]:
                print(v_label_file_abs_path)
           

            self.videos.append(videos)
           
            self.labels.append(labels)
            phase = 1
            for i in range(len(labels)-1):
                if labels[i] == labels[i+1]:
                    continue
                else:
                    phase += 1
          
            self.video_names.append(v_f)
       
        print('VideoDataset: Load dataset {} with {} videos.'.format(self.dataset, self.__len__()))
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, item):
        video, label, video_name = self.videos[item], self.labels[item], self.video_names[item]
        return video, label, video_name