from arch.network import *

def create_model(name, num_class):
    
    if name == "NTNet_Labeled":
        model = ENTNet_Labeled(num_classes=num_class)

    elif name == "ENTNet_Soft_Pseudo":
        model = ENTNet_Soft_Pseudo(num_classes=num_class)
    
    elif name == "ENTNet_Hard_Pseudo":
        model = ENTNet_Hard_Pseudo(num_classes=num_class)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)

    return model