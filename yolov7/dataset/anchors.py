import torch

def best_possible_recall_metric(boxes_wh, anchors_wh, threshold):
    '''
    Compute the matching score between definition of anchors (width/height) and the real boxes (width/height)
    Args:
        boxes_wh (Tensor): Nx2 where N is the number of boxes and 2 for width/height in pixel units
        anchors_wh (Tensor): Mx2 where M is the number of anchors and 2 for width/height in pixel units
        threshold (float): maximum factor anchors allowed to be bigger/smaller than boxes, e.g. 4 for 4 times
            Also define minimum score of 1/threshold
    Returns:
        best_possible_recall (float) number of boxes with proper anchor match (depending on threshold)
        n_matched_anchor_wh (float) average number of anchors matched per box 
    '''
    # for each box width (height), compute its ratio with every anchor width (height) 
    ratios=boxes_wh[:,None]/anchors_wh[None] # Nx1x2 / 1xMx2 => NxMx2
    # compute the normalized similarity score for anchor matching.
    # torch.minimum(ratios, 1./ratios) computes minimum between all box-width/anchor-width, anchor-width/box-width
    # and the same for height where 1 means perfect match and zero means worse match
    # then between ratio of width and that of height, we compute the worse matching score, min(dim=-1).values 
    min_ratios=torch.minimum(ratios, 1./ratios).min(dim=-1).values #NxM
    best=min_ratios.max(dim=1).values # N tensor: for each box, get the highest matching scores
    # for each box, count the number anchors whose ratio/score with this box > 1/threshold
    # (i.e., allow anchors to be thresholdx bigger than box, e.g., if threshold=4, for each boxes, 
    # we count number anchors that at most 4 time bigger than the box in each direction )
    # then find average number of match anchor (width/height definition) for each box
    n_matched_anchor_wh=(min_ratios>1./threshold).float().sum(dim=1).mean() # NxM -> N -> scalar
    # check how many boxes got score > 1/threshold
    best_possible_recall=(best>1./threshold).float().mean() # N->scalar
    
    return best_possible_recall,n_matched_anchor_wh
