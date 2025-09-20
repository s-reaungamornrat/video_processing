import math
import torch
import torchvision

from video_processing.yolov7.dataset.coords import xywh2xyxy

def one_cycle(start_val=0.0, end_value=1.0, steps=100):
    '''
    Create a sinusoidal starting with y0=start_val at x0=0 and end with yn=end_val at xn=steps 
    Args:
        start_val (float): start y value
        end_value (float): end y value
        steps (int): maximum x value
    Example:
        steps=300
        out=[one_cycle(start_val=1, end_value=0.2, steps=steps)(i) for i in range(steps)]
        _, ax=plt.subplots()
        ax.plot(out)
    '''
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (end_value - start_val) + start_val
    
def check_image_size(image_size, stride):
    '''
    Check whether image size is divisible by stride
    Args:
        image_size (list[int]): image size in the order of height and width
        stride (int): maximum model stride
    Returns:
        pass (bool): whether the image size divisible by stride
    '''
    for size in image_size:
        new_size=make_divisible(size, int(stride))
        if new_size!=size: return False
    return True

def make_grid(nx,ny):
    '''
    Create 2D image grid
    Args:
        nx (int): number of pixels along x
        ny (int): number of pixels along y
    Returns:
        grid (Tensor): 1x1xHxWx2 float coordinates
    '''
    yv, xv=torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
    # HxWx2 -> 1x1xHxWx2
    return torch.stack((xv, yv), dim=2).view(1,1, ny, nx, 2).float()
    
def make_divisible(x, divisor):
    '''
    Returned x that is divisible by divisor
    Args:
        x (number.Number): number
        divisor (num.Number): number
    Returns:
        y (number.Number): number divisible by divisor
    Example
        >> make_divisible(7, 2)
        >> 8
    '''
    return math.ceil(x/divisor) * divisor

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, multi_label=True, min_wh=2, max_wh=4096,
                       max_det=300, max_nms=3000, time_limit=10., redundant=True):
    '''
    Args:
        conf_thres (float): probability threshold for both objectness and class probability
        prediction (Tensor): Bx(AHW)xO where 
        min_wh (int): minumum box width/height in pixels
        max_wh (int): maximum box width/height in pixels
        max_det (int): maximum number of detections per image
        max_nms (int): maximum number of boxes for torchvision.ops.nms
        time_limit (float) maximum seconds before quitting
        redundant (bool): allow redundant detections
    Returns:
        output (list[Tensor[float]]): Nx6 output per images in a batch, where 6 is for (x1,y1,x2,y2, class-prob, class-index),
            where (x1,y1,x2,y2) are in pixel units and class-prob already incorporated objectness probability
    '''
    n_classes=prediction.shape[-1]-5
    are_candidates=prediction[...,4]>conf_thres # Bx? objectness

    output=[torch.zeros((0,6), device=prediction.device) for _ in range(prediction.shape[0])]
    for pidx, (pred_i, is_candidate) in enumerate(zip(prediction, are_candidates)):
        # pred_i is of size (AHW)xO and is_candidate is of size (AHW)
        candidates=pred_i[is_candidate] # (AHW)xO
        if candidates.shape[0]==0: continue
        if n_classes==1: candidates[:,5:]=candidates[:, 4:5] # p(class)=p(obj)*p(class|obj) but p(class|obj)=1
        else: candidates[:,5:]*=candidates[:, 4:5] # p(class)=p(obj)*p(class|obj)

        # boxes (center-x, center-y, width, height) in pixel units to (x1,y1,x2,y2) in pixel unit
        boxes=xywh2xyxy(candidates[:,:4]) # (AHW)x4

        # detection matrix Nx6 of (xyxy, cls-prob, cls-index start from 0)
        if multi_label:
            # Nx2 2D indices to (AHW) and C where C is the class dimension, where N is the 
            # number of candidates passing the condition 
            # i is index to candidates, while j is index to class but start at 0
            i, j=(candidates[:, 5:]>conf_thres).nonzero(as_tuple=False).T # from Nx2 to 2xN
            # boxes[i] Mx4 and j[:,None] Mx1
            # candidates[i, j+5, None] to make j index to class correctly we add 5 and add None to turn it to Mx1
            candidates=torch.cat((boxes[i], candidates[i, j+5, None], j[:,None].float()), dim=1)
        else: # best class only
            # (AHW)x1  (AHW)x1  (AHW)xC where C is the number of classes
            cls_prob, cls_idx=candidates[:,5:].max(dim=1, keepdim=True)
            candidates=torch.cat([boxes, cls_prob, cls_idx.float()], dim=1)[cls_prob.view(-1)>conf_thres]

        # Mx6
        if candidates.shape[0]==0: continue # no box
        if candidates.shape[0]>max_nms: # filter out some boxes
            candidates=candidates[candidates[:,4].argsort(descending=True)[:max_nms]] # sort by class prob

        # batch class-aware NMS
        # We are going to prevent boxes of different classes from suppressing each other by adding large offset
        # this offset is based on class index scaled by large value of max_wh
        class_index=candidates[:,5:]*max_wh # Qx1 where Q is the number of boxes
        # add offset based on class indices to boxes
        boxes, scores=candidates[:,:4]+class_index, candidates[:, 4]
        # score is p(obj)*p(class|obj) for multilable and p(obj) for single label
        box_indices=torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=iou_thres)
        if box_indices.shape[0]>max_det: box_indices=box_indices[:max_det] # limit number of detections
        output[pidx]=candidates[box_indices]

    return output