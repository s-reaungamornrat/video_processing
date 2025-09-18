import math
import torch
import torch.nn.functional as F
from video_processing.yolov7.dataset.coords import xywh2xyxy

def bbox_iou(box1, box2, form='xyxy', CIoU=True, DIoU=False, eps=1.e-7):
    '''
    Compute IOU of bounding boxes
    Args:
        box1 (Tensor): Nx4 where 4 is x1y1x2y2 or xywh (where xy is the center of boxes)
        box2 (Tensor): Nx4 where 4 is x1y1x2y2 or xywh (where xy is the center of boxes)
        form (str): format of boxes xyxyx for x1y1x2y2 or xywh
        CIoU (bool): whether to compute complete IOU see reference
        DIoU (bool):  whether to compute distance-based IOU see reference
    References:
        https://arxiv.org/abs/1911.08287v1
        https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
    '''
    assert form in ['xyxy', 'xywh']
    if form !='xyxy':
        box1=xywh2xyxy(box1) # Nx4
        box2=xywh2xyxy(box2) # Nx4

    b1_x1,b1_y1,b1_x2,b1_y2=box1[:,0],box1[:,1],box1[:,2],box1[:,3] # N where N is the number of boxes
    b2_x1,b2_y1,b2_x2,b2_y2=box2[:,0],box2[:,1],box2[:,2],box2[:,3]  # N where N is the number of boxes
    
    # intersection area
    intersect=(torch.min(b1_x2,b2_x2)-torch.max(b1_x1,b2_x1)).clamp(min=0.) * \
              (torch.min(b1_y2,b2_y2)-torch.max(b1_y1,b2_y1)).clamp(min=0.)
    # union area
    w1,h1=b1_x2-b1_x1, b1_y2-b1_y1+eps
    w2,h2=b2_x2-b2_x1, b2_y2-b2_y1+eps
    union=w1*h1 + w2*h2 -intersect +eps
    iou=intersect/union
    if not (CIoU or DIoU): return iou

    # distance or complete IoU
    cw=torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1) # convex width
    ch=torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1) # convex height
    c2=cw**2 + ch**2 +eps # convex diagonal squared
    rho2=((b2_x1+b2_x2-b1_x1-b1_x2)**2 +
          (b2_y1+b2_y2-b1_y1-b1_y2)**2) /4. # center distance squared
    if DIoU: return iou - rho2/c2
    # CIoU
    v=(4./math.pi**2)*torch.pow(torch.atan(w2/(h2+eps))-torch.atan(w1/(h1+eps)), 2.)
    with torch.no_grad(): alpha=v/(v-iou+(1.+eps))
    return iou-(rho2/c2 + v*alpha)

def box_regression(predictions, target_boxes, grid_cell, anchors):
    '''
    Args:
        predictions (Tensor[float]): Nx4 where N is the number of boxes and 4 is x,y,w,h predicted output
        target_boxes (Tensor[float]): Nx4 where N is the number of boxes and 4 is x,y,w,h in feature grid cell unit, with 
            (x,y) represents the center of boxes
        grid_cell (Tensor[int64]): top-left indices of grid cells containing the center of predicted/ground-truth boxes
        anchors (Tensor[float]): Nx2 where N is the number of boxes and 2 for width and height of anchors in feature-grid cell unit
    Returns:
        iou (Tensor[float]): N iou per pair
        avg_iou (Tensor[float]): scalar iou loss
    '''
    pred_xy=2.*predictions[:,:2].sigmoid()- 0.5 # n_targets x 2
    pred_wh=(2.*predictions[:,2:].sigmoid())**2 * anchors # n_targets x 2
    pred_boxes=torch.cat([pred_xy,pred_wh],dim=-1) # n_targets x 4 predicted boxes in grid cell coordinates

    target_boxes[:,:2]-=grid_cell # (x,y) is offset from the top-left grid indices
    iou=bbox_iou(box1=pred_boxes, box2=target_boxes, form='xywh', CIoU=True, DIoU=False, eps=1.e-7)
    return iou, (1.-iou).sum()/float(max(iou.numel(), 1))

def multilabel_classification_loss(predictions, target_class_indices, pos_weight, pos_value=1., neg_value=0.):
    '''
    Compute multilabel class losses using binary cross entropy, i.e., each object can be assigned many classes.
    Assuming predictions from valid samples (i.e., objects exist) so this function only focuses determining classes
    Args:
        predictions (Tensor[float]): NxC logits where N is the number of objects and C is the number of classes
        target_class_indices (Tensor[int64]): N target class indices
        pos_weight (Tensor[float]): weight of positive sample
        pos_value (float): probability value of positive class
        neg_value (float): probability value of negative class
    Returns:
        bce_loss (Tensor[float]): binary classification loss scalar value
    '''
    # here we do not need to multiply p_obj and p_cls since we only compute the classification loss
    # for positive samples (objects exist)
    target_classes=torch.full_like(predictions, neg_value, device=predictions.device) # n_targets x n_classes
    target_classes[range(predictions.shape[0]), target_class_indices]=pos_value # one-hot
    return F.binary_cross_entropy_with_logits(input=predictions, target=target_classes, pos_weight=pos_weight)