import torch
import numpy as np
from scipy.cluster.vq import kmeans

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

def check_anchor_matching(dataset, model, thr=4., imgsz=640):
    '''
    Check whether anchors is not thrx bigger or smaller than boxes in training data. If not, attempt to update it using genetic-evolved kmean
    Args:
        dataset: contains normalized box information [normalized box center (x,y) and normalized width/height (w,h)]. It must also
            contain width/heights of all images (used to denormalized boxes)
        model: contain detection module with anchors, anchor_grid, and stride information as input anchors to check and to update 
            if needed
        thr (float): largest factor that allows anchors to be bigger/smaller than boxes in training data
        imgz (int): input image size for traing model (not original image size)
    '''
    # get detection module
    module=model.model[-1]
    # make the maximum size = imgsz while keeping the aspect ratio consistent
    shapes=imgsz*dataset.image_sizes/dataset.image_sizes.max(1, keepdims=True) # Nx2 where N is the number of images
    scale=np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1)) # Nx1 augment scale
    # change normalized width,height to width/height in pixel unit using scaled_shape then stack along number of boxes to Nx2
    boxes_width_heights=torch.from_numpy(np.concatenate([s[None,:]*l[:,3:] for s, l in zip(shapes*scale, dataset.labels)], axis=0)).float()

    # nlx1xnax1x1x2 -> Mx2 width and height of each anchors
    anchors_width_heights=module.anchor_grid.clone().cpu().view(-1, 2)
    bpr,n_anchors=best_possible_recall_metric(boxes_wh=boxes_width_heights, anchors_wh=anchors_width_heights, threshold=thr)
    print(f'In dataset.anchors.check_anchor_matching anchors/target={n_anchors:.2f}, Best possible recall (BPR): {bpr:.4f}')

    if bpr>0.98: return
        
    # recompute anchors
    n_anchors=module.anchor_grid.numel()//2 # divide by 2 because this contain both width and height
    try: new_anchor_width_heights=kmean_anchors(dataset=dataset, n_anchors=n_anchors, img_size=imgsz, thr=thr, n_generations=1000, verbose=False)
    except Exception as e: print(f'Error: {e}')
    new_bpr=best_possible_recall_metric(boxes_wh=boxes_width_heights, anchors_wh=new_anchor_width_heights, threshold=thr)[0]
    
    if new_bpr<bpr: 
        print(f'In dataset.anchors.check_anchor_matching new anchors have lower bpr of {new_bpr} compared to {bpr}--used old one')
        return

    new_anchor_width_height_tensor=torch.from_numpy(new_anchor_width_heights).to(device=module.anchors.device, dtype=module.anchors.dtype)
    module.anchor_grid=new_anchor_width_height_tensor.clone().view_as(module.anchor_grid)
    print(module.anchors.shape, module.stride.shape)
    # normalize the anchors from image grid to feature cell grid
    module.anchors=new_anchor_width_height_tensor.clone().view_as(module.anchors)/module.stride.to(module.anchors.device).view(-1,1,1)
    check_anchor_order(module)
    print('In dataset.anchors.check_anchor_matching:: New anchor width/height has been estimated. Please update configuration file')

    
def kmean_anchors(dataset, n_anchors, img_size,thr=4.0, n_generations=1000, verbose=False): 
    '''
    Create kmean-evolved anchors from training data
    Args:
        dataset: loaded dataset
        n_anchors (int): number of anchors
        img_size (int): training data size
        thr (float): ratio of width/height between anchors and ground-truth boxes
        n_generations (int): the number of generations to evolve anchors using genetic algorithm
        verbose (bool): print result
    Returns:
        anchors (ndarray[float]): n_anchorsx2 where 2 is for width and height
    Example:
        anchors = kmean_anchors(dataset=dataset, n_anchors=n_anchors, img_size=imgsz, thr=thr, n_generations=1000, verbose=True)
    '''
    def print_results(k, reference_k, score_threshold, nk, img_size):
        '''
        Args:
            k (Tensor): Kx2 where K is the number of anchors and 2 is for width and height 
            reference_k (Tensor): Mx2 where M is the number of boxes and 2 for width and height
            score (float): score threshold
            nk (int): number of desired k
        '''
        
        scores, best_scores=metric(torch.from_numpy(k).float(), reference_k)
        # best possible recall and the number of anchors with scores > input-score
        bpr, n_matched_achors=(best_scores>score_threshold).float().mean(), nk*(scores>score_threshold).float().mean()
        print(f'threshold={score_threshold:.2f}: {bpr:.4f} best possible recall, {n_matched_achors:.2f} anchors passes the threshold')
        print(f'n_anchors={nk}, img_size={img_size}, metric_all={scores.mean():.3f}/{best_scores.mean():.3f} mean/best', end=', ')
        print(f'past_threshold {scores[scores>score_threshold].mean():.3f}-mean: ', end='')
        for i, x in enumerate(k): print(f'({int(round(x[0].item()))},{int(round(x[1].item()))})', end=', ' if i<len(k)-1 else '\n' )

    def metric(k, wh):
        '''
        Compute size matching score where 1 means best match and < 0 means worst match
        Args:
            k (Tensor): Kx2 where K is the number of boxes and 2 for width and height
            wh (Tensor): Mx2 where M is the number of reference boxes and 2 for width and height
        Returns:
            scores (Tensor): MxK size matching scores 
            best_scores (Tensor): M best matching scores for each box in k
        '''
        ratio=wh[:,None]/k[None] # MxKx2
        scores=torch.minimum(ratio, 1./ratio).min(dim=-1).values # MxK matching scores
        return scores, scores.max(dim=-1).values
    
    def anchor_fitness(k, reference_k, score_threshold):
        '''
        Args:
            k (Tensor): Kx2 where K is the number of boxes and 2 for width and height
            reference_k (Tensor): Mx2 where M is the number of reference boxes and 2 for width and height
            score_threshold (float): score threshold
        '''
        _, best=metric(torch.from_numpy(k).float(), reference_k)
        return (best*(best>score_threshold).float()).mean()
        
        
    score=1./thr
    # training data image size, keeping original aspect ratio between width and height 
    image_sizes=img_size*dataset.image_sizes/dataset.image_sizes.max(axis=1, keepdims=True) # Nx2 where N is the number of training data
    # convert normalized width and height stored as labels to width and height in pixel unit 
    original_boxes_width_height=np.concatenate([l[:, 3:5]*s[None,:] for l, s in zip(dataset.labels, image_sizes)], axis=0) # Mx2

    # filter
    small_boxes=(original_boxes_width_height<3.).any(axis=1).sum()
    if small_boxes>0: print(f'Warning: Extremely small objects found. {small_boxes} of {len(original_boxes_width_height)} labels are < 3 pixels in size ')
    filtered_boxes_width_height=original_boxes_width_height[(original_boxes_width_height>=2.).any(axis=1)] # we allow only boxes > 2 pixel width and height

    # Kmean
    print(f'Running Kmeans for {n_anchors} on {len(filtered_boxes_width_height)} boxes')
    std=filtered_boxes_width_height.std(axis=0) # 2 element sigmas for whitening of width and height
    k, dist=kmeans(filtered_boxes_width_height/std, n_anchors, iter=30) # n_anchorsx2 points and scalar mean distance
    assert len(k)==n_anchors, f'Error: scipy.cluster.vq.kmeans requested {n_anchors} points but returns only {len(k)}'
    k*=std[None,:] # n_anchors x 2
    filtered_boxes_width_height=torch.from_numpy(filtered_boxes_width_height).float()
    original_boxes_width_height=torch.from_numpy(original_boxes_width_height).float()

    #k = k[np.argsort(k.prod(1))]  # sort small to large
    print_results(k=k, reference_k=original_boxes_width_height, score_threshold=score, nk=n_anchors, img_size=img_size)

    # evolve
    fitness=anchor_fitness(k, reference_k=filtered_boxes_width_height, score_threshold=score)
    mutation_prob, sigma=0.9, 0.1
    for it in range(n_generations):
        v=np.ones_like(k)
        while (v==1).all(): # mutate until a change occurs (prevent duplicates)
            v=((np.random.random(k.shape)<mutation_prob)*np.random.random()*np.random.randn(*k.shape)*sigma+1.).clip(0.3, 3.0)
        kg=(k.copy()*v).clip(min=2.)
        fg=anchor_fitness(kg, reference_k=filtered_boxes_width_height, score_threshold=score)
        if fg>fitness:
            fitness, k=fg, kg.copy()
            if verbose: print_results(k=k, reference_k=original_boxes_width_height, score_threshold=score, nk=n_anchors, img_size=img_size)
    return k
