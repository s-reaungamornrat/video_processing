import copy

import torch
import torchvision
import torch.nn.functional as F

from video_processing.yolov7.dataset.coords import xywh2xyxy

def find_5_positive(prediction, targets, anchors, matching_threshold, inside_grid_cell=1.):
    '''
    Find the 5 most relevant grid-cell locations for each target object. In other words, assign, to each target box, 5 grid cells that locate 
    around the box center
    Args:
        anchors (Tensor): Nl x Na x 2 where Nl is the number of levels, and Na is the number of anchors and 2 for width and height in
            feature cell grids
        targets (Tensor): Nx6 where N is the number of targets and 6 for (image_idx, class, x, y, w, h) where image_idx in [0, batch_size)
            (x,y) is the normalized box center and (w,h) is the normalized width and height 
        prediction (list[Tensor]): sequence of 2Nl tensors, each BxNaxHxWxO. We input prediction since we need to know feature shape
        matching_threshold (float): how big/small anchors allowed to be relative to boxes
        inside_grid_cell (float): threshold to check whether targets fall inside a grid cell must be <1. since 
            each grid cell is of size 1x1
    Returns:
        indices (list[tuple[Tensor]]): (image-index, anchor-indices, grid-j, grid-i) per level, each item is 1D tensor of size Q where 
            Q may vary per level. grid-i and grid-j are the top-left corner of grid cell containing objects' center
        anch (list[Tensor]): matched anchors per level, each of size Qx2 where Q may vary but match that of indices for the same level
        target_class_indices (list[Tensor]): target class indices per level, each of size Q where Q may vary but match that of indices 
            for the same level
        target_boxes  (list[Tensor]): Qx4 target boxes per level, each of size Q defined in feature-grid unit where Q may vary but match
            that of indices for the same level. (x,y,w,h) where x,y are box center and w,h are width/height in feature grid cell unit 
    '''
    # output storage
    indices,anch=[],[]
    target_boxes,target_class_indices=[],[]
    
    nl,na=anchors.shape[:2] # number of levels and anchors
    nt=targets.shape[0] # number of targets
    # placeholder to store grid cell size as [1,1,W,H,W,H,1] where the first (W,H) is for (x,y) and the latter for (w,h)
    grid_cell_resolution=torch.ones(7, device=targets.device).long() 
    # example of anchor_indices
    # if there are 3 anchors and 2 targets, anchor_indices is
    # [[0, 0]
    #  [1, 1]
    #  [2, 2]]
    anchor_indices=torch.arange(na, device=targets.device).float().view(na,1).repeat(1, nt)  # naxnt
    # create a tensor where we have each anchor matches to every target 
    #   naxntx7 = naxntx6     naxntx1   (targets originally ntx6)
    # na x nt x 7 where 7 is for img-indx, class, x,y,w,h, anchor-indx, where x,y,w,h are normalized
    targets=torch.cat((targets.repeat(na, 1, 1), anchor_indices[:,:,None]), dim=2) # append anchors index to each target

    # 2D nearest neighbor grid indices including current cell (0,0)
    offsets=inside_grid_cell*torch.tensor([[0,0],[1,0],[0,1],[-1,0],[0,-1]], device=targets.device).float() # 5x2
    
    for i in range(nl):
        anchors_per_level=anchors[i] # na x 2

        # [1,1,W,H,W,H,1]
        grid_cell_resolution[2:6]=torch.tensor(prediction[i].shape,device=targets.device)[[3,2,3,2]] # WHWH -> xyxy size
        
        # multiplying targets by grid_cell_resolution to convert normalize location to location in feature cell grid
        # naxntx7 where 7 is for img-indx, class, x,y,w,h, anchor-indx 
        t=targets*grid_cell_resolution 
        if nt>0:
            # --- compute matching score and select only pairs of anchors-targets passing the test
            # match ratio in cell grid: na x nt x 2 / na x 1 x 2 where 2 for width height
            width_height_ratio=t[:,:,4:6]/anchors_per_level[:,None]
            # na x nt x 2 -> na x nt
            within_threshold=torch.maximum(width_height_ratio, 1./width_height_ratio).max(dim=2).values < matching_threshold
            # naxntx7 -> Nx7 where N is the number of pairs of anchors and targets whose width/height ratio within threshold
            t=t[within_threshold] #select only target-anchors that pass the threshold test
            # --- for the selected pairs, check whether the target locations are in the current grid cells or in neighboring grid cells
            # target box center location x, y in grid cell coordinates
            # the fractional part tells us where inside the grid cell the target is
            gxy=t[:, 2:4]  # Nx2 distance of targets from the top left corner of feature grid
            # target center location from grid size. To handle targets that are near the right or bottom edge of a grid cell
            # in that case, we want to consider neighboring cells on the right or below as positive candidates
            gxi=grid_cell_resolution[[2,3]]-gxy # Nx2 distance of targets from the bottom right corner of the feature grid
            # (gxy%1.<inside_grid_cell) check whether fractional part of target-locations (gxy) is inside the grid cell 
            # (gxy>1.) check whether target-location (gxy) fall into cell close to border (we not allow negative indices)
            j,k=((gxy%1.<inside_grid_cell)&(gxy>1.)).T # Nx2 -> 2xN -> N, N bool
            # (gxi%1. < inside_grid_cell) check whether fractional part of target-location (gxi) is inside neighboring cell 
            # (gxi>1.) check whether target-location (gxi) fall into cell close to the border (we not allow negative indices)
            # if (gxi<1.), targets will match previous condition, i.e, j,k
            l,m=((gxi%1. < inside_grid_cell)&(gxi>1.)).T # Nx2 -> 2xN > N, N bool
            # --- create mask to select cells (current and/or neighbors) that may contain target centers  
            # 5xN bool mask where 5 is for original-cell and 4 nearest-neighbor cells
            mask=torch.stack((torch.ones_like(j), j, k, l, m)) 
            # --- select target locations according the mask (of current and/or neighbor cells)
            t=t.repeat((5,1,1))[mask] # 5xNx7[5xN]  -> Qx7
            # --- select offsets according the mask (of current and/or neighbor cells)
            #              1xNx2                5x1x2              5xN -> Qx2
            offsets_=(torch.zeros_like(gxy)[None]+offsets[:,None])[mask]
        else:
            t=targets[0]
            offsets_=0

        # compute target locations
        grid_xy=t[:,2:4] # Qx2 target box center in feature grid coordinates
        grid_wh=t[:,4:6] # Qx2 target box width/height in feature grid coordinates
        # Qx2 - Qx2 -> Qx2
        grid_ijs=(grid_xy-offsets_).long() # like floor yields top-left integer grid cell indices
        grid_i, grid_j=grid_ijs.T # Qx2 -> 2xQ -> Q, Q

        # get image-index and object-class of these targets
        img_idx, obj_cls=t[:,:2].long().T # Qx2 -> 2xQ -> Q, Q

        anchor_idx=t[:,6].long()
        # Note grid_cell_resolution = [1,1,W,H,W,H,...]
        indices.append((img_idx, anchor_idx, grid_j.clamp_(0, grid_cell_resolution[3]-1),
                        grid_i.clamp_(0, grid_cell_resolution[2]-1)))
        anch.append(anchors_per_level[anchor_idx]) # anchors
        # grid_xy is the location of box center in feature-cell unit
        target_boxes.append(torch.cat((grid_xy, grid_wh), dim=1)) # Qx4 #.append(torch.cat((grid_xy-grid_ijs, grid_wh), dim=1) # Qx4 
        target_class_indices.append(obj_cls) # Q
    return indices, anch,target_class_indices, target_boxes

def matched_prediction_per_level_per_image(indices, anch, prediction, batch_index, stride):
    '''
    Get matched indices and prediction for each level. This is part of build_target2 in https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py#L1440
    Args:
        indices (list[tuple[Tensor]]): (image-index, anchor-indices, grid-y, grid-x) per level, each item is 1D tensor of size Q where 
            Q may vary per level
        anch (list[Tensor(float)]): Nx2 anchor width/height per level where N may vary per level and 2 is for width/height in feature-grid cell unit
        prediction (list[Tensor]): BxAxHxWxO prediction per level, where A is the number of anchors and O is the output dimension
        batch_index (int): image index in a batch, i.e., [0,B)
        stride (Tensor[float]): model stride per level
    Returns:
        pxyxys (Tensor): Qx4 predicted box location as x1,y1,x2,y2 in pixel units
        p_obj (Tensor): Qx1 predicted objectness logit
        p_cls (Tensor): QxNc predicted class logits, where Nc is the number of classes
        from_which_layer (Tensor[float]): Q levels to which each box matches
        all_img_idx (Tensor[long]): Q image/batch index of each box
        all_anch_idx (Tensor[long]): Q anchor index of each box
        all_gj (Tensor[long]): Q grid-cell y index of each box
        all_gi (Tensor[long]): Q grid-cell x index of each box
        all_anch (Tensor[float]): Qx2 anchors matched to each box
    '''
    pxyxys,p_cls,p_obj,from_which_layer=[],[],[],[]
    all_img_idx,all_anch_idx,all_gj,all_gi,all_anch=[],[],[],[],[]
    
    for level, pred_l in enumerate(prediction): 
        # pred_i is of size BxAxHxWxO, where A is the number of anchors and O is the output dimension

        # img_idx is image-index, anch_idx is anchor-index, gj is grid_y and gi is grid-x
        img_idx, anch_idx, gj, gi=indices[level]
        b_idx=(img_idx==batch_index)
        img_idx,anch_idx,gj,gi=img_idx[b_idx],anch_idx[b_idx],gj[b_idx],gi[b_idx]
        all_img_idx.append(img_idx); all_anch_idx.append(anch_idx)
        all_gj.append(gj); all_gi.append(gi)
        all_anch.append(anch[level][b_idx])
        from_which_layer.append(torch.ones(len(img_idx), device=pred_l.device) * level)

        # prediction [x,y,w,h,objectness,class_probs...]
        fg_pred=pred_l[img_idx,anch_idx,gj,gi] # len(img_idx)xO foreground/positive predictions
        p_obj.append(fg_pred[:,4:5]) # objectness len(img_idx)x1
        p_cls.append(fg_pred[:,5:]) # classes len(img_idx)xNc

        # this is the same as inference-branch in forward of Detection module 
        # we note that gi and gj is x, y in feature grid cell unit
        grid=torch.stack([gi,gj], dim=1) # len(img_idx)x2 = Qx2
        pxy=(fg_pred[:,:2].sigmoid()*2. - 0.5 + grid) *stride[level] # Qx2 stride converts feature grid unit to pixel unit
        pwh=(fg_pred[:,2:4].sigmoid()*2)**2 * anch[level][b_idx]*stride[level] # Qx2 stride converts feature grid unit to pixel unit
        pxywh=torch.cat([pxy,pwh], dim=-1) # Qx4
        pxyxy=xywh2xyxy(pxywh) # in pixel units
        pxyxys.append(pxyxy)

    # x1,y1,x2,y2 in pixel units
    pxyxys=torch.cat(pxyxys, dim=0)

    p_obj=torch.cat(p_obj, dim=0) # each Nx1 where N may vary
    p_cls=torch.cat(p_cls, dim=0) # each NxNc where N may vary and Nc is the number of classes
    from_which_layer=torch.cat(from_which_layer, dim=0) # each 1D tensor with varying size = each N for p_cls, p_obj
    all_img_idx=torch.cat(all_img_idx, dim=0) # each 1D tensor with varying size = each N for p_cls, p_obj
    all_anch_idx=torch.cat(all_anch_idx, dim=0) # each 1D tensor with varying size = each N for p_cls, p_obj
    all_gj=torch.cat(all_gj, dim=0) # each 1D tensor with varying size = each N for p_cls, p_obj
    all_gi=torch.cat(all_gi, dim=0) # each 1D tensor with varying size = each N for p_cls, p_obj
    all_anch=torch.cat(all_anch, dim=0) # each Nx2 where N may vary

    return pxyxys, p_obj, p_cls, from_which_layer, all_img_idx,all_anch_idx,all_gj,all_gi, all_anch


def determine_matching_targets(prediction, targets, indices, anch, stride, image_size, n_classes):
    '''
    This function answers the question `who is responsible for detecting what?`. It dynamically assigns anchors and feature-map cells to ground
    truth boxes so the network learns which predictions should fire for each object. Here, we compute IOU and class prediction losses to 
    find the best matching targets and anchors for each prediction per level for auxiliary head and select the best anchors and grid cells 
    that should be responsible for detecting ground-truth objects and returns formatted target tensors.
    This is a more documented version of build_target2 in https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py#L1440
    Args:
        prediction (list[Tensor[float]]): main-head output from model per level, each of size BxAxHxWxO,
            where A is the number of anchors, O is the output dimension (for x,y,w,h,obj,class..., see doc/network.pptx for details)
        targets (Tensor[float]): Ntx6 where Nt is the number of target boxes and 6 for image-index, class, x,y,h,w 
            (see doc/network.pptx for details)
        indices (list[tuple[Tensor]]): (image-index, anchor-indices, grid-y, grid-x) per level, each item is 1D tensor of size Q where 
            Q may vary per level
        anch (list[Tensor(float)]): Nx2 anchor width/height per level where N may vary per level and 2 is for width/height in feature-grid cell unit
        stride (Tensor[float]): model stride per level
        image_size (int): width or height, assuming width=height
        n_classes (int): number of classes
    Returns:
        matching_bs (list[Tensor[int64]]): 1D tensor of image indices in a batch per level, with value in [0,batch_size)
        matching_as (list[Tensor[int64]]): 1D tensor of anchor indices per level, with value in [0,n_anchors)
        matching_gjs (list[Tensor[int64]]): 1D tensor of grid indices along H/Y per level, with value in [0, feature-grid-height) at that level
        matching_gis (list[Tensor[int64]]): 1D tensor of grid indices along W/X per level, with value in [0, feature-grid-width) at that level
        matching_targets (list[Tensor[float]]): Wx6 targets per level, where W may vary per level and is the number of matched targets for each prediction
            (indicated by the aboved returned indices) and 6 for image-index, class, x,y,w,h (see doc/network.pptx)
        matching_anchs (list[Tensor[float]]): Wx2 per level, where W may vary per level and is the number of matched anchors and 2 for width/height defined
            in feature-grid cell unit
    '''

    # matched image index per level, i.e., if len(prediction)=4, there will be 4 sublists
    matching_bs=[[] for _ in prediction]  # each sublist stores values pertaining to each image in the batch
    matching_as=copy.deepcopy(matching_bs) # matched anchor index per level 
    matching_gjs=copy.deepcopy(matching_bs) # for grid_y
    matching_gis=copy.deepcopy(matching_bs) # grid_x
    matching_targets=copy.deepcopy(matching_bs) 
    matching_anchs=copy.deepcopy(matching_bs) 
    nl=len(prediction)
    for batch_index in range(prediction[0].shape[0]): # index into each image in the batch
        # batch_index is point to each image in a batch
        img_idx=targets[:,0]==batch_index
        # each is 6D for batch_idx, class, x, y, w, h where the last 4 in normalized space
        this_target=targets[img_idx] # boxes associated with this image N<=Nt
        if this_target.shape[0]==0: continue
    
        # Nx4 box center (x,y) and width,height in pixel unit
        target_xywh=this_target[:,2:]*image_size # imgs[batch_index].shape[1] # assume image width=height
        target_xyxy=xywh2xyxy(target_xywh) # Nx4 x1,y1,x2,y2 in pixel unit

        # get matched prediction, indices to features (batch item, anchor, grid), and anchors per level
        # pxyxys is Nx4 x1,y1,x2,y2 in pixel unit
        pxyxys, p_obj, p_cls, from_which_layer, all_b,all_a,all_gj,all_gi, all_anch=matched_prediction_per_level_per_image(indices, anch=anch,
                                                                     prediction=prediction, batch_index=batch_index, stride=stride)
        if pxyxys.shape[0]==0: continue
    
        # N x Q where N is number of target boxes and Q is the number of predicted boxes
        pair_wise_iou=torchvision.ops.box_iou(target_xyxy, pxyxys) # both in pixel units
        pair_wise_iou_loss=-torch.log(pair_wise_iou+1e-8) # smoothly but strongly penalize low matching 
    
        # for each target, find the k best match predictions
        top_k=torch.topk(pair_wise_iou, k=min(20,pair_wise_iou.shape[1]), dim=1).values # Nxk
        # SimOTA (simple optimal transport assignment) uses dynamic label assignment to determine how many anchors (matches)
        # are selected for each ground truth. In other words, determine how many boxes per target
        dynamic_ks=torch.clamp(top_k.sum(1).int(), min=1) # guarantees a minimum of 1 positive match/anchor per GT box
        
        # NxNc -> Nx1xNc -> NxQxNc
        #print(f'In loss.utils.determine_matching_targets this_target[:,1] {this_target[:,1]}, n_classes {n_classes}')
        gt_cls_per_img=F.one_hot(this_target[:,1].to(dtype=torch.int64), n_classes).float().unsqueeze(1).repeat(1, pxyxys.shape[0], 1)
        # NxNc -> 1xQxNc -> NxQxNc
        p_cls=p_cls.float().unsqueeze(0).repeat(this_target.shape[0],1,1).sigmoid()
        # Nx1 -> 1xQx1 ->  NxQx1
        p_obj=p_obj.unsqueeze(0).repeat(this_target.shape[0],1,1).sigmoid()
        # sqrt here is a heuristic trick to soften dymanic range of class probability, making low probability slightly higher
        pred_cls_per_img=(p_obj*p_cls).sqrt() 
        pred_logit_per_img=torch.log(pred_cls_per_img/(1.-pred_cls_per_img))
        # NxQxNc NxQxNc = NxQxNc -> NxQ
        pairwise_cls_loss=F.binary_cross_entropy_with_logits(pred_logit_per_img, gt_cls_per_img, reduction='none').sum(dim=-1)
    
        # NxQ = NxQ NxQ
        cost=(pairwise_cls_loss +3.*pair_wise_iou_loss)
    
        # NxQ for each target, set select flag to candidates that have low cost
        matching_matrix=torch.zeros_like(cost)
        for gt_idx in range(cost.shape[0]): # for each target, set candidate to 1 if cost is low
            # dynamic_ks[gt_idx].item() is number of top k and cost is NxQ
            _, pos_idx=torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[gt_idx][pos_idx]=1.
        del top_k, dynamic_ks
        # how many targets match for each candidate Q
        anchor_matching_gt=matching_matrix.sum(0) # NxQ-> Q
        # if candidate matches more than 1 target, select 1 best target for these candidates
        if (anchor_matching_gt>1).sum() >0: 
            # min cost with number of elements = (anchor_matching_gt>1).sum() 
            # cost[:,anchor_matching_gt>1] get cost only from candidates that match more than 1 target
            # then, for those candidates (matching more than 1 target), get index to targets with lowest cost
            _, cost_argmin=torch.min(cost[:,anchor_matching_gt>1], dim=0)
            matching_matrix[:, anchor_matching_gt>1]*=0.
            matching_matrix[cost_argmin, anchor_matching_gt>1]=1.
        # matching_matrix.sum(0)>0 determines which candidates match with some targets, L<Q
        match_some_targets_flag=matching_matrix.sum(0)>0 # boolean of len Q
        # matching_matrix[:, match_some_targets_flag] select only candidates that match with target
        # for each matching candidate, select index of matching target
        matched_gt_inds=matching_matrix[:, match_some_targets_flag].argmax(0) # L<Q
    
        # Nx6 -> Lx6 where L might > N
        this_target = this_target[matched_gt_inds] # target for each candidate in L candidates
        from_which_layer=from_which_layer[match_some_targets_flag] # L<Q
        all_b = all_b[match_some_targets_flag] # L<Q
        all_a = all_a[match_some_targets_flag] # L<Q
        all_gj = all_gj[match_some_targets_flag] # L<Q
        all_gi = all_gi[match_some_targets_flag] # L<Q
        # Lx2 < Qx2 where 2 is for width/height in feature grid cell
        all_anch = all_anch[match_some_targets_flag] 
    
        for l in range(nl):
            layer_mask = from_which_layer == l # boolean mask of length L<Q
            matching_bs[l].append(all_b[layer_mask]) # M<L<Q and M could be 0
            matching_as[l].append(all_a[layer_mask]) # M<L<Q and M could be 0
            matching_gjs[l].append(all_gj[layer_mask]) # M<L<Q and M could be 0
            matching_gis[l].append(all_gi[layer_mask]) # M<L<Q and M could be 0
            matching_targets[l].append(this_target[layer_mask]) # Mx6 where M<L<Q and M could be 0
            matching_anchs[l].append(all_anch[layer_mask]) # Mx2 where M<L<Q and M could be 0
    
    for level in range(nl):
        if len(matching_targets[level])==0:
            matching_bs[level] = torch.tensor([], device=targets.device, dtype=torch.int64)
            matching_as[level] = torch.tensor([], device=targets.device, dtype=torch.int64)
            matching_gjs[level] = torch.tensor([], device=targets.device, dtype=torch.int64)
            matching_gis[level] = torch.tensor([], device=targets.device, dtype=torch.int64)
            matching_targets[level] = torch.tensor([], device=targets.device, dtype=torch.int64)
            matching_anchs[level] = torch.tensor([], device=targets.device, dtype=torch.int64)
            continue
        # each level, each sublist below contains tensor per image in a batch so we concatenate them
        # to form batch per level
        # 1D scalar of size W where W is the sum of all M from each image
        matching_bs[level] = torch.cat(matching_bs[level], dim=0)  # W
        matching_as[level] = torch.cat(matching_as[level], dim=0)  # W
        matching_gjs[level] = torch.cat(matching_gjs[level], dim=0) # W
        matching_gis[level] = torch.cat(matching_gis[level], dim=0) # W
        matching_targets[level] = torch.cat(matching_targets[level], dim=0) # Wx6
        matching_anchs[level] = torch.cat(matching_anchs[level], dim=0) # Wx2

    return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs 