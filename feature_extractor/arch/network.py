import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from resnest.torch import resnest50


class Resnest50_Encoder(nn.Module):
    def __init__(self):
        super(Resnest50_Encoder, self).__init__()
        resnet = resnest50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
    
    def forward(self, x):
        x = self.share.forward(x)
        return x

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output


## Only Labeled Data 
class ENTNet_Labeled(nn.Module):
    def __init__(self, num_classes=6):
        super(ENTNet_Labeled, self).__init__()

        # Backbone
        self.backbone = Resnest50_Encoder()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes)
        )

        self.initialize_classifier()

        # Loss
        self.criterion = nn.CrossEntropyLoss()
    
    def initialize_classifier(self):
        for layer in self.classifier.children():
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def extract_feat(self, x):
        x = self.backbone(x)
        return x


## Soft Pseudo Label
class Soft_Pseudo_Label_Loss(nn.Module):
    def __init__(self, alpha=0.1, T=10):
        super(Soft_Pseudo_Label_Loss, self).__init__()
        self.alpha = alpha
        self.criterion = nn.KLDivLoss(reduction='none')
        self.T = T
    
    def forward(self, outputs, labels, teacher_outputs):
        KD_losses = self.criterion(F.log_softmax(outputs/self.T, dim=1),
                                   F.softmax(teacher_outputs/self.T, dim=1)) * (self.alpha * self.T * self.T)
        
        valid_CE = (labels != 6)
        CE_losses = F.cross_entropy(outputs[valid_CE], labels[valid_CE], reduction='none') * (1. - self.alpha)
        
        total_losses = KD_losses.sum(dim=1)
        total_losses[valid_CE] += CE_losses
        
        return total_losses.mean()
    

class ENTNet_Soft_Pseudo(nn.Module):
    def __init__(self, num_classes=6):
        super(ENTNet_Soft_Pseudo, self).__init__()

        # Backbone
        self.backbone = Resnest50_Encoder()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes)
        )

        self.initialize_classifier()

        # Loss
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_train = Soft_Pseudo_Label_Loss()

    def initialize_classifier(self):
        for layer in self.classifier.children():
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def extract_feat(self, x):
        x = self.backbone(x)
        return x


## Hard Pseudo Label
class Hard_Pseudo_Label_Loss(nn.Module):
    def __init__(self, alpha=0.1, T=10):
        super(Hard_Pseudo_Label_Loss, self).__init__()
        self.alpha = alpha
        
        self.T = T
    
    def forward(self, outputs, labels, teacher_outputs):
        valid_CE = (labels != 6)
        CE_losses = F.cross_entropy(outputs[valid_CE], labels[valid_CE], reduction='mean')
        valid_UN = (labels == 6)
        Unlabeled = F.cross_entropy(outputs[valid_UN], teacher_outputs[valid_UN], reduction='mean')
        
        total_losses = (CE_losses + Unlabeled) / 2
        return total_losses


class ENTNet_Hard_Pseudo(nn.Module):
    def __init__(self, num_classes=6):
        super(ENTNet_Hard_Pseudo, self).__init__()

        # Backbone
        self.backbone = Resnest50_Encoder()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes)
        )

        self.initialize_classifier()

        # Loss
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_train = Hard_Pseudo_Label_Loss()

    def initialize_classifier(self):
        for layer in self.classifier.children():
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def extract_feat(self, x):
        x = self.backbone(x)
        return x