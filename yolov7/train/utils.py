import torch
import torch.nn as nn

import numpy as np

from video_processing.yolov7.utils.general import one_cycle

def train_an_epoch(args, model, model_ema, optimizer, train_loss_module, train_loader, epoch):
    '''
    Train each epoch
    Args:
        hyp (dict): hyperparameters
    Returns:
         mean_weighted_loss (Tensor[float]): stack of mean loss, and mean weighted box_loss, objectness_loss, class_loss
         mean_unweigthed_loss  (Tensor[float]): stack of mean box_loss, objectness_loss, class_loss
    '''
    # number of warmup iterations, e.g., max(3 epochs, 1K iterations)
    n_warmup=max(model.hyp['warmup_epochs']*len(train_loader), 1000)
    print(f'In train.utils n_warmup {n_warmup}')

    device=next(model.parameters()).device
    
    # train epoch
    mean_weighted_loss=torch.zeros(4, device=device) # loss, and weighted box_loss, objectness_loss, class_loss
    mean_unweigthed_loss=torch.zeros(3, device=device) # box_loss, objectness_loss, class_loss
    
    accumulate=None
    optimizer.zero_grad() # we will simulate large batch size
    for it, (imgs, targets, paths, _) in enumerate(train_loader, 1):
        # number of accumulated batches used to train model so far since the start
        n_batches=(it-1)+len(train_loader)*(epoch-1) # epoch count from 1
        imgs=imgs.to(device, dtype=torch.float32, non_blocking=True) / 255.0 # uint8 to float32 from 0-255 to 0-1
        targets=targets.to(device, dtype=torch.float32, non_blocking=True)
        # warm-up
        if n_batches <= n_warmup:
            # simulate large batch size by determine when to accumulate gradients. During warmup, accumulate after every batch
            # while later on accumulate when the number of batches matches the nominal batch size
            accumulate = max(1, np.interp(n_batches, [0, n_warmup], [1, args.nominal_batch_size / args.batch_size]).round())
            for g_idx, params in enumerate(optimizer.param_groups):
                # lr of bias falls from warmup_bias_lr to lr0 while other lrs rise from 0. to lr0
                params['lr']=np.interp(n_batches, [0, n_warmup], [model.hyp['warmup_bias_lr'] if g_idx==2 else 0.0, 
                                                        params['initial_lr']*one_cycle(1., model.hyp['lrf'], args.epochs)(epoch-1)])
                if 'momentum' in params: params['momentum']=np.interp(n_batches, [0, n_warmup], [model.hyp['warmup_momentum'], model.hyp['momentum']])
    
        # forward
        predictions=model(imgs)
        loss, weighted_loss, unweighted_loss=train_loss_module(predictions, targets, images=imgs, matching_threshold=model.hyp['anchor_t'],
                                                               box_weight=model.hyp['box'], obj_weight=model.hyp['obj'], 
                                                               cls_weight=model.hyp['cls'])
        # backward
        loss.backward()
        if accumulate is None or n_batches%accumulate==0: # update parameters
            optimizer.step()
            optimizer.zero_grad()
            if model_ema is not None: model_ema.update(model)
            
        # update mean loss
        mean_weighted_loss=(mean_weighted_loss*it + weighted_loss)/(it+1) 
        mean_unweigthed_loss=(mean_unweigthed_loss*it + unweighted_loss)/(it+1)
        # print
        if args.print_freq>0 and it%args.print_freq==0:
            mem='%.3gG' % (torch.cuda.memory_reserved()/1E9 if torch.cuda.is_available() else 0) # GB
            print('{} [{:.2f}%]: {} | {} | acc {}'.format(it, 100*it/len(train_loader),
                                          ', '.join(f'{n}:{l:.3f}' for n, l in zip('loss, w-bb,w-obj,w-cls'.split(','),mean_weighted_loss.cpu().tolist())),
                                          ', '.join(f'{n}:{l:.3f}' for n, l in zip('bb,obj,cls'.split(','),mean_unweigthed_loss.cpu().tolist())),
                                                          accumulate)
                  )
        if args.dev_mode: break
            
    return mean_weighted_loss, mean_unweigthed_loss
    
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
    pg0,pg1,pg2=[],[],[]
    for i, (k, v) in enumerate(model.named_modules()):
        # we have nn.BatchNorm2d bias which get detected here
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter): pg2.append(v.bias) 
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
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2
    return optimizer