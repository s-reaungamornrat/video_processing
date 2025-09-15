import torch
import torch.nn as nn


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