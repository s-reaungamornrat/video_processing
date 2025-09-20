import torch
import torch.nn.functional as F
from video_processing.yolov7.loss.utils import find_5_positive, determine_matching_targets
from video_processing.yolov7.loss.loss import box_regression, multilabel_classification_loss

class ComputeLoss:
    def __init__(self, model, cls_pw, obj_pw, label_smoothing, use_aux=True):
        '''
        Args:
            cls_pw (float/sequence): positive weight for class classification in BCEWithLogitsLoss
            obj_pw (float/sequence): positive weight for objectness in BCEWithLogitsLoss
            label_smoothing (float): label smoothing eps
            use_aux (bool): whether to compute loss for auxillary heads
        '''
        super(ComputeLoss, self).__init__()
        device=next(model.parameters()).device
        self.use_aux=use_aux
        self.class_positive_weight=torch.tensor([cls_pw], device=device)
        self.object_positive_weight=torch.tensor([obj_pw], device=device)

        # positive and negative class
        # https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
        self.positive_class, self.negative_class=1.0 - 0.5*label_smoothing, 0.5*label_smoothing

        # weight of the objectness loss contributed by different scales (detection heads at different scales)
        # preventing any scales to dominate the loss. Emphasize more on high resolution (e.g., stride 8) since 
        # capturing small structures is more difficult, while scaling down coarse resolution since large objects 
        # are easy to detect (objectness of large object is high and could dominate). These scales are determined 
        # based on the number of cells of each features returned from each level
        self.balance=[4.,1.,0.4] if model.model[-1].nl<4 else [4., 1., .25, .06, .02]
        # blending factor between fixed objectness of 1 and IoU between prediction and ground truth
        # used to set target objectness, i.e., target_objectness = (1-gr)+gr*iou
        self.gr=model.gr 
        for k in 'na,nc,nl,anchors,stride'.split(','): setattr(self, k, getattr(model.model[-1], k))

    def __call__(self, predictions, targets, images, matching_threshold, box_weight, obj_weight, cls_weight): 
        '''
        Args:
            predictions (list[Tensor]): list of 2NL BxAxHxWxO output where NL is the number of levels, (2 for output from main and auxillary heads)
            targets (list[Tensor]): list of Ntx6 targets per level, where Nt is the number of targets which may vary per level and 6 for image-index,
                class-index, x,y,w,h in normalized space relative to image width and height. (x,y) is the box center
            images (Tensor[float]): BxCxHxW we only need it for shape information
            matching_threshold (float): factor to which anchors allow to be bigger/smaller than target boxes
            box_weight (float): weight to box regression loss
            obj_weight (float): weight to objectness loss
            cls_weight (float): weight to classification loss
        Returns:
            loss (Tensor): scalar differentiable loss
            weighted_losses (Tensor): stack of weighted box_loss, objectness_loss, class_loss, loss
            unweighted_losses (Tensor): stack of original box_loss, objectness_loss, class_loss, loss
        '''
        if not self.use_aux: return self._aux_disabled_loss(predictions, targets, images, matching_threshold, box_weight, obj_weight, cls_weight)
        return self._aux_enabled_loss(predictions, targets, images, matching_threshold, box_weight, obj_weight, cls_weight)
        
    def _aux_disabled_loss(self, predictions, targets, images, matching_threshold, box_weight, obj_weight, cls_weight):
        '''
        Args:
            predictions (list[Tensor]): list of 2NL BxAxHxWxO output where NL is the number of levels, (2 for output from main and auxillary heads)
            targets (list[Tensor]): list of Ntx6 targets per level, where Nt is the number of targets which may vary per level and 6 for image-index,
                class-index, x,y,w,h in normalized space relative to image width and height. (x,y) is the box center
            images (Tensor[float]): BxCxHxW we only need it for shape information
            matching_threshold (float): factor to which anchors allow to be bigger/smaller than target boxes
            box_weight (float): weight to box regression loss
            obj_weight (float): weight to objectness loss
            cls_weight (float): weight to classification loss
        Returns:
            loss (Tensor): scalar differentiable loss
            weighted_losses (Tensor): stack of weighted box_loss, objectness_loss, class_loss, loss
            unweighted_losses (Tensor): stack of original box_loss, objectness_loss, class_loss, loss
        '''
        device=predictions[0].device
        indices, anchors,target_class_indices, target_boxes=find_5_positive(prediction=predictions[:self.nl], targets=targets, 
                            anchors=self.anchors, matching_threshold=matching_threshold, inside_grid_cell=.5)
        # list of 1D of WHWH (XYXY) grid size per level
        feature_grid_resolution=[torch.tensor(pred.shape, device=device)[[3,2,3,2]] for pred in predictions[:self.nl]]
        
        box_loss=torch.zeros(1, device=device)
        class_loss=torch.zeros(1, device=device)
        objectness_loss=torch.zeros(1, device=device) # objectness
        for level in range(self.nl):
            pred_l=predictions[level] # BxAxHxWxO prediction for level l 
            # image-index, anchor-index, grid-j, grid-i
            b, a, gj, gi=indices[level] # all 1D long indices
            target_objectness=torch.zeros_like(pred_l[...,0], device=device) # BxAxHxW
            
            n_targets=b.shape[0]
            if n_targets>0:
                # predictions corresponding to targets
                positive_pred_l=pred_l[b,a,gj,gi] # n_targets x O
                iou, iou_loss=box_regression(positive_pred_l[:,:4], target_boxes=target_boxes[level], grid_cell=torch.stack([gi, gj], dim=1), 
                                             anchors=anchors[level])
                box_loss+=iou_loss
        
                # BxAxHxW target objectness blending iou and fixed objectness of 1
                target_objectness[b,a,gj,gi]=(1.-self.gr)+self.gr*iou.detach().clamp(min=0.).type(pred_l.dtype)
        
                # classification
                if self.nc>1: # only for multiple classes
                    class_loss+=multilabel_classification_loss(predictions=positive_pred_l[:,5:], target_class_indices=target_class_indices[level], 
                                                   pos_weight=self.class_positive_weight, pos_value=self.positive_class, 
                                                   neg_value=self.negative_class)
            # objectness losses
            obj_loss=F.binary_cross_entropy_with_logits(input=pred_l[...,4], target=target_objectness, pos_weight=self.object_positive_weight)
            objectness_loss+=self.balance[level]*obj_loss
            
        
        loss=box_weight*box_loss+cls_weight*class_loss+obj_weight*objectness_loss
        weighted_losses=torch.cat((loss, box_weight*box_loss, obj_weight*objectness_loss, cls_weight*class_loss)).detach()
        unweighted_losses=torch.cat((box_loss, objectness_loss, class_loss)).detach()
        return float(images.shape[0])*loss, weighted_losses, unweighted_losses

    def _aux_enabled_loss(self, predictions, targets, images, matching_threshold, box_weight, obj_weight, cls_weight): 
        '''
        Args:
            predictions (list[Tensor]): list of 2NL BxAxHxWxO output where NL is the number of levels, (2 for output from main and auxillary heads)
            targets (list[Tensor]): list of Ntx6 targets per level, where Nt is the number of targets which may vary per level and 6 for image-index,
                class-index, x,y,w,h in normalized space relative to image width and height. (x,y) is the box center
            images (Tensor[float]): BxCxHxW we only need it for shape information
            matching_threshold (float): factor to which anchors allow to be bigger/smaller than target boxes
            box_weight (float): weight to box regression loss
            obj_weight (float): weight to objectness loss
            cls_weight (float): weight to classification loss
        Returns:
            loss (Tensor): scalar differentiable loss
            weighted_losses (Tensor): stack of loss, and weighted box_loss, objectness_loss, class_loss
            unweighted_losses (Tensor): stack of original box_loss, objectness_loss, class_loss
        '''
        device=predictions[0].device

        # find the targets in cell grid unit that match anchors for training auxillary head
        indices4aux, anch4aux,_,_ =find_5_positive(prediction=predictions[:self.nl], targets=targets, anchors=self.anchors,
                                                     matching_threshold=matching_threshold, inside_grid_cell=1.)
        bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux \
        =determine_matching_targets(prediction=predictions[:self.nl], targets=targets, indices=indices4aux, anch=anch4aux,
                                                       stride=self.stride, image_size=images.shape[2], n_classes=self.nc)
        
        indices4main, anch4main,_,_=find_5_positive(prediction=predictions[:self.nl], targets=targets, anchors=self.anchors,
                                                     matching_threshold=matching_threshold, inside_grid_cell=.5)
        bs, as_,gjs, gis, targets, anchors \
        =determine_matching_targets(prediction=predictions[:self.nl], targets=targets, indices=indices4main, anch=anch4main,
                                                       stride=self.stride, image_size=images.shape[2], n_classes=self.nc)
        # list of 1D of WHWH (XYXY) grid size per level
        feature_grid_resolution=[torch.tensor(pred.shape, device=device)[[3,2,3,2]] for pred in predictions[:self.nl]]
        
        box_loss=torch.zeros(1, device=device)
        class_loss=torch.zeros(1, device=device)
        objectness_loss=torch.zeros(1, device=device) # objectness
        for level in range(self.nl):
            pred_l=predictions[level] # BxAxHxWxO prediction for level l 
            pred_aux_l=predictions[self.nl+level] # BxAxHxWxO prediction from auxillary head for level l 
            # image-index, anchor-index, grid-j, grid-i
            b, a, gj, gi=bs[level], as_[level], gjs[level], gis[level] # all 1D long indices
            b_aux,a_aux, gj_aux, gi_aux=bs_aux[level], as_aux_[level], gjs_aux[level], gis_aux[level] # all 1D long indices
            target_objectness=torch.zeros_like(pred_l[...,0], device=device) # BxAxHxW
            target_objectness_aux=torch.zeros_like(pred_aux_l[...,0], device=device) # BxAxHxW
            
            n_targets=b.shape[0]
            if n_targets>0:
                # predictions corresponding to targets
                positive_pred_l=pred_l[b,a,gj,gi] # n_targets x O
                iou, iou_loss=box_regression(positive_pred_l[:,:4], target_boxes=targets[level][:,2:]*feature_grid_resolution[level][None], 
                       grid_cell=torch.stack([gi, gj], dim=1), anchors=anchors[level])
                box_loss+=iou_loss
        
                # BxAxHxW target objectness blending iou and fixed objectness of 1
                target_objectness[b,a,gj,gi]=(1.-self.gr)+self.gr*iou.detach().clamp(min=0.).type(pred_l.dtype)
        
                # classification
                if self.nc>1: # only for multiple classes
                    class_loss+=multilabel_classification_loss(predictions=positive_pred_l[:,5:], target_class_indices=targets[level][:,1].long(), 
                                                   pos_weight=self.class_positive_weight, pos_value=self.positive_class, 
                                                   neg_value=self.negative_class)
            n_aux=b_aux.shape[0] # number of target for auxillary head
            if n_aux>0:
                positive_pred_aux_l=pred_aux_l[b_aux, a_aux, gj_aux, gi_aux]
                iou_aux, iou_aux_loss=box_regression(positive_pred_aux_l[:,:4], target_boxes=targets_aux[level][:,2:]*feature_grid_resolution[level][None], 
                               grid_cell=torch.stack([gi_aux, gj_aux], dim=1), anchors=anchors_aux[level])
                box_loss+=0.25*iou_aux_loss
        
                # objectness target
                target_objectness_aux[b_aux, a_aux, gj_aux, gi_aux]=(1.-self.gr) + self.gr*iou_aux.detach().clamp(0).type(pred_l.dtype)
        
                if self.nc>1:
                    class_loss+=0.25*multilabel_classification_loss(predictions=positive_pred_aux_l[:,5:], target_class_indices=targets_aux[level][:,1].long(), 
                                       pos_weight=self.class_positive_weight, pos_value=self.positive_class, 
                                       neg_value=self.negative_class)
            # objectness losses
            obj_main_loss=F.binary_cross_entropy_with_logits(input=pred_l[...,4], target=target_objectness, pos_weight=self.object_positive_weight)
            obj_aux_loss=F.binary_cross_entropy_with_logits(input=pred_aux_l[...,4], target=target_objectness_aux, pos_weight=self.object_positive_weight)
            objectness_loss+=self.balance[level]*(obj_main_loss+0.25*obj_aux_loss)
            
        
        loss=box_weight*box_loss+cls_weight*class_loss+obj_weight*objectness_loss
        weighted_losses=torch.cat((loss, box_weight*box_loss, obj_weight*objectness_loss, cls_weight*class_loss)).detach()
        unweighted_losses=torch.cat((box_loss, objectness_loss, class_loss)).detach()
        return float(images.shape[0])*loss, weighted_losses, unweighted_losses