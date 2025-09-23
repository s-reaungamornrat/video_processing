import torch
import torchvision

import numpy as np

from video_processing.yolov7.metrics.utils import ap_per_class
from video_processing.yolov7.utils.general import non_max_suppression
from video_processing.yolov7.dataset.coords import adjust_coords, xywh2xyxy

@torch.no_grad()
def statistics_per_image(outputs, targets, iouv, input_image_size, shapes, stats):
    '''
    Args:
        outputs (list[Tensor[float]]): Nx6 output per image, where N is the number of boxes, and 6 is for (x1,y1,x2,y2, class-prob, class-index)
            where (x1,y1,x2,y2) in pixel unit
        targets (Tensor[float]): Mx6 where M is the number of boxes, and 6 is for (image-index, class-index, x,y,w,h) where (x,y,w,h) in pixel unit
            and (x,y) is the box center
        iouv (Tensor[float]): U multiple IoU threshold 
        input_image_size (tuple[int]): size of input image to the model, H, W
        shapes (tuple): ((H0,W0), (factor_w, factor_h), (shift_x, shift_y)) where H0,W0 are original image size,
                        (factor_w, factor_h) is the ratio of new/original for width and height and 
                        (shift_x, shift_y) is the shift of pixel indices
        stats (list): in-place returns of the stack of (correct, predicted-class-prob, predicted-class-indices, target-class-index)
    '''
    # statistic per image
    for img_idx, output in enumerate(outputs): # output is xyxy, class-prob, class-index where xyxy in pixel unit
        
        labels=targets[targets[:,0]==img_idx, 1:] # class-index, x,y,w,h in pixel unit
        target_classes=labels[:,0].tolist() if len(labels)>0 else []
        if len(output)==0:
            if len(labels)>0:stats.append((torch.zeros(0, iouv.numel(), dtype=torch.bool), torch.tensor(), torch.tensor(), target_classes))
            continue
    
        # adjust boxed to coordinates on original image size
        boxes=adjust_coords(input_image_size=input_image_size, boxes=output[:,:4], original_image_size=shapes[img_idx][0], size_factor=shapes[img_idx][1][0],
                            shift=shapes[img_idx][1][0])
        output[:,:4]=boxes
    
        # check whether each prediction is correct across multiple IoU thresholds
        correct=torch.zeros(output.shape[0], iouv.numel(), dtype=torch.bool, device=output.device)
        if len(labels)>0:
            target_class_indices=labels[:,0] # 1D tensor
            target_boxes=xywh2xyxy(labels[:,1:]) # Qx4 for x1,y1,x2,y2 in pixel units
            # adjust boxed to coordinates on original image size
            target_boxes=adjust_coords(input_image_size=input_image_size, boxes=target_boxes, original_image_size=shapes[img_idx][0],
                                       size_factor=shapes[img_idx][1][0], shift=shapes[img_idx][1][0])
            check_prediction_correctness(output=output, target_boxes=target_boxes, target_class_indices=labels[:,0], iouv=iouv, correct=correct)
        
        # Append statistics (correct, predicted-class-prob, predicted-class-indices, target-class-index)
        stats.append((correct.cpu(), output[:, 4].cpu(), output[:, 5].cpu(), target_classes))

@torch.no_grad()
def check_prediction_correctness(output, target_boxes, target_class_indices, iouv, correct):
    '''
    Check whether prediction yield IoU greater than any threshold IoU. This is class aware calculation
    Args:
        output (Tensor[float]): Nx6 where N is the number of prediction and 6 is for x1,y1,x2,y2,class-prob,class-index where x1,y1,x2,y2 are
            defined in pixel units in the original image size 
        target_boxes (Tensor[float]): Mx4 where M is the number of target boxes and 4 is for x1,y1,x2,y2 in pixel units in the original image size 
        target_class_indices (Tensor[float/int64]): M class indices associated with Mx4 target_boxes
        iouv (Tensor[float]): U multiple IoU threshold 
        correct (Tensor[bool]): NxU whether prediction greater any threshold IoU, in place 
    '''
    # check whether each prediction is correct across multiple IoU thresholds
    #correct=torch.zeros(output.shape[0], iouv.numel(), dtype=torch.bool, device=output.device)
    detected_target_indices=[] # to see how many target has been detected
    
    # process prediction and target per target class to avoid mixing across classes
    unique_class_indices=target_class_indices.unique()
    for cls_idx in unique_class_indices: # loop over each target classes
        # get indices to prediction and targets of the same class
        target_idx=(cls_idx==target_class_indices).nonzero(as_tuple=False).view(-1) # nonzero on 1D tensor should yield 1D indices, say size P
        pred_idx=(cls_idx==output[:,-1]).nonzero(as_tuple=False).view(-1) # nonzero on 1D tensor should yield 1D indices, say size Q
        if len(pred_idx)==0: continue
  
        # for the boxes of the same classes, compute QxP IoU and get the Q best IoU, i.e., for each output, which is the best matched target
        pairwise_ious, iuo_idx=torchvision.ops.box_iou(output[pred_idx,:4], target_boxes[target_idx]).max(1)

        # append detections
        detection_set=set()
        for j in ((pairwise_ious>iouv[0]).nonzero(as_tuple=False)):
            detected_target_idx=target_idx[iuo_idx[j]]
            # use detection_set to make sure that each target box matched only once for this class
            if detected_target_idx.item() in detection_set: continue
            
            detection_set.add(detected_target_idx.item())
            detected_target_indices.append(detected_target_idx)
            # check correctness by seeing whether IoU > anu IoU thresholds
            correct[pred_idx[j]]=pairwise_ious[j]>iouv
            #print('correct[pred_idx[j]] ', correct[pred_idx[j]])
            if len(detected_target_indices)==len(target_boxes): break # all target get already detected
    return correct

@torch.no_grad()
def validation(model, dataloader, val_loss_module, hyp, conf_thres=0.001, iou_thres=0.6,verbose=False, n_its=None):

    model.eval()
    
    device=next(model.parameters()).device

    nc=model.nc # number of classes

    # iou vector for mAP@0.5:0.95
    iouv=torch.linspace(0.5, 0.95, 10).to(device) # 10 elements
    
    
    # class_names={k:v for k, v in enumerate(model.names)}
    # coco91class=coco80_to_coco91_class()
    #str_header = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p=r=f1=mp=mr=map50=map=t0=t1=0.
    mean_weighted_loss=torch.zeros(4, device=device)
    mean_unweighted_loss=torch.zeros(3, device=device)
    stats,ap,ap_class=[],[],[]
    
    for b_idx, (imgs, targets, paths, shapes) in enumerate(dataloader, 1):
        
        if n_its is not None and b_idx>n_its: break # developing mode, stop early
            
        imgs=imgs.to(device=device, dtype=torch.float32, non_blocking=True)/255.
        targets=targets.to(device=device)

        formatted_outputs, raw_outputs=model(imgs)
        loss, weighted_losses, unweighted_losses=val_loss_module(raw_outputs, targets, images=imgs,matching_threshold=hyp['anchor_t'],
                                                                 box_weight=hyp['box'], obj_weight=hyp['obj'],  cls_weight=hyp['cls'])
        mean_weighted_loss=(mean_weighted_loss*b_idx + weighted_losses)/(b_idx+1) 
        mean_unweighted_loss=(mean_unweighted_loss*b_idx + unweighted_losses)/(b_idx+1)
        
        # Run NMS: Nx6 output per images in a batch, where 6 is for (x1,y1,x2,y2, class-prob, class-index),
        # where (x1,y1,x2,y2) are in pixel units and class-prob already incorporated objectness probability
        outputs=non_max_suppression(prediction=formatted_outputs, conf_thres=conf_thres, iou_thres=iou_thres,  multi_label=True)
    
    
        # convert target boxes to pixel units
        height,width=imgs.shape[-2:]
        targets[:,2:]*=torch.tensor([width,height,width,height], device=device,dtype=targets.dtype)[None]
    
        statistics_per_image(outputs, targets, iouv, input_image_size=imgs.shape[2:], shapes=shapes, stats=stats)
    
    
    # (correct, predicted-class-prob, predicted-class-indices, target-class-index)
    # concatenate items from each category together
    stats=[np.concatenate(stat, 0) for stat in zip(*stats)] 
    
    n_targets_per_class=torch.zeros(1)
    if len(stats)>0 and stats[0].any():
        p,r,ap,f1,ap_class=ap_per_class(*stats, v5_metric=False)
        ap50, ap=ap[:,0], ap.mean(1) # Ap@0.5, AP@0.5:0.95
        mp, mr, map50, map=p.mean(), r.mean(), ap50.mean(), ap.mean()
        n_targets_per_class=np.bincount(stats[3].astype(np.int64), minlength=nc)
        
    
    # Print results
    # pf = '%20s' + '%12i' + '%12.3g' * 4  # print format
    # print(pf % ('all', n_targets_per_class.sum(), mp, mr, map50, map))
    txt=f'val-loss={mean_weighted_loss[0].item():.3f}, targets={int(n_targets_per_class.sum())}, mp={mp:.3f}, mr={mr:.3f}, map50={map50:.3f}, map={map:.3f}'
    # print(f'Validation: loss={mean_weighted_loss[0].item():.3f}, targets={n_targets_per_class.sum()}, mp={mp:.3f}, mr={mr:.3f}, map50={map50:.3f}, map={map:.3f}', flush=True)
    
    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):  print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    
    # return results
    maps=np.zeros(nc)+map
    for i, c in enumerate(ap_class): maps[c]=ap[i]
        
    return (mp, mr, map50, map, mean_weighted_loss.cpu(), mean_unweighted_loss.cpu()), maps, txt

@torch.no_grad()
def validate(model, dataloader, val_loss_module, hyp, n_its=None):

    model.eval()
    
    device=next(model.parameters()).device
    
    mean_weighted_loss=torch.zeros(4, device=device)
    mean_unweighted_loss=torch.zeros(3, device=device)
    
    for b_idx, (imgs, targets, ratios, shifts) in enumerate(dataloader, 1):
        
        if n_its is not None and b_idx>n_its: break # developing mode, stop early
            
        imgs=imgs.to(device=device, dtype=torch.float32, non_blocking=True)/255.
        targets=targets.to(device=device)

        formatted_outputs, raw_outputs=model(imgs)
        loss, weighted_losses, unweighted_losses=val_loss_module(raw_outputs, targets, images=imgs,matching_threshold=hyp['anchor_t'],
                                                                 box_weight=hyp['box'], obj_weight=hyp['obj'],  cls_weight=hyp['cls'])
        mean_weighted_loss=(mean_weighted_loss*b_idx + weighted_losses)/(b_idx+1) 
        mean_unweighted_loss=(mean_unweighted_loss*b_idx + unweighted_losses)/(b_idx+1)
    
    return mean_weighted_loss, mean_unweighted_loss