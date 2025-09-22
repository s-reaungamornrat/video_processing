import torch
import numpy as np

def rescale(img, out_min=0., out_max=1.):
    '''
    Rescale image intensity
    Args:
        img (Tensor/ndarray): HxWxC or CxHxW
    Returns:
        img (Tensor/ndarray): HxWxC or CxHxW
    '''
    in_min, in_max=img.min(), img.max()
    return (img-in_min)*(out_max-out_min)/(in_max-in_min) + out_min

def alpha_bending(foreground, background, alpha):
    '''
    Args:
        foreground (Tensor): NxCxHxW image to be on the foreground
        background (Tensor): NxCxHxW image to be on the background
        alpha (float): opacity of the foreground
    Returns:
        image (Tensor): NxCxHxW  alpha-blended image 
    '''
    assert all(x.ndim==4 for x in [foreground, background])
    background=background.to(torch.float32)
    foreground=foreground.to(torch.float32)
    C=max(background.shape[1], foreground.shape[1])
    
    foreground=rescale(foreground, out_max=255.)
    if foreground.shape[1]!=C: 
        foreground=torch.cat([foreground, torch.zeros_like(foreground), foreground], dim=1) # NxCxHxW where C is RBG channels
    if background.shape[1]!=C: 
        background=torch.cat([background, background, background], dim=1) # NxCxHxW where C is RBG channels
    
    return (background*(1.-alpha)+foreground*alpha).to(dtype=torch.uint8)