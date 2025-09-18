import torch
import torch.nn as nn

import numpy as np

def labels_to_class_weights(labels, n_classes):
    '''
    Compute class weight which is the inverse frequency of training labels
    Args:
        labels (tuple[ndarray[float]]): labels per image, each Nx5 where N is the number of objects and 5 for class object, x,y,w,h
            with (x,y) is a normalized box center and (w,h) is a normalized width,height
        n_classes (int): number of classes
    '''
    # check whether labels have been loaded
    if len(labels)==0 or labels[0] is None: return torch.Tensor()

    labels=np.concatenate(labels, axis=0) # Nx5 where N is boxes from all data in dataset
    classes=labels[:,0].astype(np.int32)
    frequency=np.bincount(classes, minlength=n_classes)
    print('frequency ', frequency)
    # replace empty frequency with 1
    frequency[frequency==0]=1
    weights=1/frequency # weight is the inverse class frequency
    # normalize so weight sum to 1
    weights/=weights.sum()
    return torch.from_numpy(weights)
    
def setup_optimizer(model, learning_rate, momentum, weight_decay):
    '''
    Setup optimizer by avoiding imposing weight decay on batchnorm as well as bias. We avoid imposing weight decay on bias since bias represent 
    the mean shift and for batch norm, we want mean statistics. Thus, we only apply weight decay to convolution weights (i.e., the network is fully CNN)
    We note that the original work https://github.com/WongKinYiu/yolov7/blob/main/train_aux.py#L41 use SGD but we use AdamW
    Args:
        learning_rate (float): initial learning rate
        momentum (float): optimizer momentume
        weight_decay (float): weight decay
    '''
    # optimizer parameter groups: no-decay pg0, decay pg1
    pg0,pg1=[],[]
    for i, (k, v) in enumerate(model.named_modules()):
        # we have nn.BatchNorm2d bias which get detected here
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter): pg0.append(v.bias) 
        if isinstance(v, nn.BatchNorm2d): pg0.append(v.weight); continue
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter): pg1.append(v.weight); continue
        if hasattr(v, 'im'): # module list
            for iv in v.im: pg0.append(iv.implicit)
            continue
        if hasattr(v, 'ia'): # module list
            for iv in v.ia: pg0.append(iv.implicit)
            continue
    optimizer=torch.optim.AdamW(pg0, lr=learning_rate, betas=(momentum, 0.999), weight_decay=0.) # adjust beta1 to momentum
    optimizer.add_param_group({'params':pg1, 'weight_decay':weight_decay}) # pg1 with weight_decay
    del pg0, pg1
    return optimizer