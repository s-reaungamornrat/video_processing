import os
import re
import sys
import yaml
import random

import torch
import torch.nn as nn
import numpy as np

seed=0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

from video_processing.yolov7.parameter_parser import parser
from video_processing.yolov7.models.model import Model
from video_processing.yolov7.models.ema import ModelEMA
from video_processing.yolov7.loss.module import ComputeLoss
from video_processing.yolov7.train.utils import setup_optimizer, labels_to_class_weights, train_an_epoch
from video_processing.yolov7.dataset.coco_dataset import LoadImagesAndLabels
from video_processing.yolov7.utils.general import one_cycle, check_image_size
from video_processing.yolov7.dataset.anchors import check_anchor_matching
from video_processing.yolov7.test.utils import validation

def main(args, hyp, data_dict):
    '''
    Args:
        args: input arguments
        hyp (dict): hyper parameters
        data_dict (dict): data settings
    '''

    device=torch.device('cpu' if not torch.cuda.is_available() or args.device=='cpu' else 'cuda')
    print(f'Computing device {device} with batch size {args.batch_size}')
    
    # number of classes
    nc=1 if args.single_cls else int(data_dict['nc']) 
    names=['item'] if args.single_cls and len(data_dict['names'])!=1 else data_dict['names'] # class names
    assert len(names)==nc, f'There are {len(names)} class names but {nc} classes' 
    
    # save running
    with open(os.path.join(args.output_dirpath, 'hyp.yaml'), 'w') as f: yaml.dump(hyp, f, sort_keys=False)
    with open(os.path.join(args.output_dirpath, 'args.yaml'), 'w') as f: yaml.dump(vars(args), f, sort_keys=False)
    
    # define model and optimizers
    model=Model(args.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # it is safer to move model to device first and then create optimizer
    optimizer=setup_optimizer(model, learning_rate=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=one_cycle(1., hyp['lrf'], args.epochs))
    model_ema=ModelEMA(model)
    start_epoch, best_fitness=1, -np.inf
    if args.resume is not None and os.path.isfile(os.path.join(args.checkpoint_dirpath, args.resume)):
        resume_fpath=os.path.join(args.checkpoint_dirpath, args.resume)
        if os.path.isfile(os.path.join(args.checkpoint_dirpath, args.best_checkpoint_fname)):
            resume_fpath=os.path.join(args.checkpoint_dirpath, args.best_checkpoint_fname)
        checkpoint = torch.load(resume_fpath, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']+1
        best_fitness=checkpoint['best_fitness']
        model_ema.ema.load_state_dict(checkpoint['ema'])
        model_ema.updates=checkpoint['updates']

    # wrap distributed training
    
    # check that the image size divisible by stride
    max_stride=max(int(model.stride.max()), 32)
    assert check_image_size(image_size=args.img_size, stride=max_stride), f'{args.img_size} must be divisible by {max_stride}'

    # train/val data loader
    train_dataset=LoadImagesAndLabels(data_dirpath=args.data_dirpath, image_paths=data_dict['train'], img_size=args.img_size[0],
                                augment=True, hyp=hyp, n_data=args.n_training_data, correct_exif=args.correct_exif)
    val_dataset=LoadImagesAndLabels(data_dirpath=args.data_dirpath, image_paths=data_dict['val'], img_size=args.img_size[0],
                                augment=False, hyp=hyp, n_data=args.n_val_data, correct_exif=args.correct_exif)

    train_loader=torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True, 
                                            collate_fn=LoadImagesAndLabels.collate_fn)
    val_loader=torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True, 
                                            collate_fn=LoadImagesAndLabels.collate_fn)

    check_anchor_matching(dataset=train_dataset, model=model, thr=4., imgsz=args.img_size[0])
    print(f'There are total {len(train_dataset)} training data and {len(val_dataset)} validation data')
    
        
    # model parameters
    nl=model.model[-1].nl
    print("hyp['box'] ", hyp['box'],  " hyp['cls'] ", hyp['cls'], " hyp['obj] ", hyp['obj'] )
    hyp['box']*=3./nl # box-regression loss weight scaled to layer
    hyp['cls']*= nc/80. * 3./nl # classification loss weight scaled to classes and layers
    hyp['obj']*=(args.img_size[0]/640)**2 *3./nl # objectness loss weight scaled to image size and layers
    hyp['label_smoothing']=args.label_smoothing
    model.nc=nc # attach number of classes to model
    model.hyp=hyp
    # blending factor between fixed objectness of 1 and IoU between prediction and ground truth
    # used to set target objectness, i.e., target_objectness = (1-gr)+gr*iou
    model.gr=1. 
    # model.class_weights=nc*labels_to_class_weights(labels=train_dataset.labels, n_classes=nc).to(device)
    # model.names=data_dict['names']
    print("hyp['box'] ", hyp['box'],  " hyp['cls'] ", hyp['cls'], " hyp['obj] ", hyp['obj'], ' args.label_smoothing ', args.label_smoothing )
    # print('model.class_weights ', model.class_weights)
        
    scheduler.last_epoch=start_epoch-1 # do not move?
    train_loss_module=ComputeLoss(model, cls_pw=hyp['cls_pw'], obj_pw=hyp['obj_pw'], label_smoothing=args.label_smoothing, use_aux=True)
    val_loss_module=ComputeLoss(model, cls_pw=hyp['cls_pw'], obj_pw=hyp['obj_pw'], label_smoothing=args.label_smoothing, use_aux=False)
    
    for epoch in range(start_epoch, args.epochs):

        # train a model
        model.train()
        mean_weighted_loss, mean_unweigthed_loss=train_an_epoch(args, model, model_ema, optimizer, train_loss_module, train_loader, epoch=epoch)

        # scheduler
        scheduler.step()

        # mAP validation model
        model_ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride']) #, 'class_weights'])

        # validation
        results, maps, txt=validation(model=model_ema.ema, dataloader=val_loader, val_loss_module=val_loss_module, hyp=hyp, 
                                      conf_thres=0.001, iou_thres=0.6, verbose=False, n_its=1 if args.dev_mode else None)
        
        # compute model fitness as a weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        fitness=(np.array(results[:4]) * w).sum()

        lr_txt=', '.join([f"{params['lr']:.3f}"  for params in optimizer.param_groups])
        print(f'Epoch {epoch}: fitness {fitness:.3f}, train-loss={mean_weighted_loss[0].item():.3f}, {txt}, lr=({lr_txt})', flush=True)

        # save parameters
        training_state_dict={'epoch': epoch, 'model': model.state_dict(), 'ema':model_ema.ema.state_dict(),  'optimizer': optimizer.state_dict(),
                             'scheduler': scheduler.state_dict(), 'updates':model_ema.updates, 'best_fitness': best_fitness}
        torch.save(training_state_dict, os.path.join(args.checkpoint_dirpath, args.resume))
        if fitness > best_fitness:
            best_fitness=fitness
            training_state_dict['best_fitness']=best_fitness
            torch.save(training_state_dict, os.path.join(args.checkpoint_dirpath, args.best_checkpoint_fname))
        

if __name__ == '__main__':

    args=parser.parse_args()

    if not os.path.isdir(args.output_dirpath):os.makedirs(args.output_dirpath)
    args.checkpoint_dirpath=os.path.join(args.output_dirpath, args.checkpoint_dirname)
    if not os.path.isdir(args.checkpoint_dirpath): os.makedirs(args.checkpoint_dirpath)
    
    # hyperparameters
    with open(args.hyp) as f: hyp=yaml.load(f, Loader=yaml.SafeLoader)
    # data
    with open(args.data) as f: data_dict=yaml.load(f, Loader=yaml.SafeLoader)

    
    main(args, hyp, data_dict)