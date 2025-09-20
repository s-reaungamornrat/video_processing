import torch
import numpy as np

def xywh2xyxy(x):
    '''
    Convert x,y,w,h where (x,y) is the box center and (w,h) is the width and height to x1,y1,x2,y2 where
    (x1,y1) is the top left corner of the boxes and (x2,y2) is the bottom right corner
    Args:
        x (Tensor): Nx4 where N is the number of boxes and 4 for x,y,w,h
    '''
    if len(x)==0: return x
    y=x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:,0]=x[:,0] - x[:,2]/2 # top left x
    y[:,1]=x[:,1] - x[:,3]/2 # top left y
    y[:,2]=x[:,0] + x[:,2]/2 # bottom right x
    y[:,3]=x[:,1] + x[:,3]/2 # bottom right y
    return y
def normalized_xy2xy(x, w, h, shift_x=0., shift_y=0.):
    '''
    Convert normalized pixel unit to pixel unit by multiplying by width and height
    Args: 
        x (ndarray/Tensor): Nx2 where N is the number of pixels and 2 for x, y after normalized so they range from 0 to 1
        w (int/float): width of image
        h (int/float): height of image
        shift_x (int/float: offset along x
        shift_y (int/float): offset along y
    Returns:
        y (ndarray/Tensor): Nx2 where N is the number of pixels and 2 for x, y after denormalized so they range from 0 to w and 0 to height
    '''
    if len(x)==0: return x
    y=x.clone() if isinstance(x, torch.Tensor) else x.copy()
    y[:,0]=w*x[:,0]+shift_x
    y[:,1]=h*x[:,1]+shift_y 
    return y

def adjust_coords(input_image_size, boxes, original_image_size, size_factor, shift):
    '''
    Correct boxes coordinates to fit original image size. In other words, change boxe coordinates from being relative 
    to input_image_size to being relative to original image size. Inverse of dataset.coords.normalized_xy2xy
    Args:
        input_image_size (tuple[int]): size of input image to model, H, W
        boxes (Tensor[float]): Nx4 where N is the number of boxes and 4 for x1,y1,x2,y2 in pixel units
        original_image_size (tuple[int]): size of original image (H,W)
        size_factor (tuple[float]): original_image_size/input_image_size for H and W
        shift (tuple[float]): shift of boxes along x, y
    Returns: 
        adjusted_boxes (Tensor[float]): Nx4 where N is the number of boxes and 4 for x1,y1,x2,y2 in pixel units
    '''
    if shift is None: shift=(0.,0.)
    if size_factor is None: size_factor=[n/o for n,o in zip(input_image_size, original_image_size)]
    # assert size_factor[0]==size_factor[1], f'{size_factor} must be the same'

    boxes=boxes.clone()
    boxes[:,[0,2]]-=shift[0]
    boxes[:,[1,3]]-=shift[1]
    boxes[:,[0,2]]/=size_factor[1]
    boxes[:,[1,3]]/=size_factor[0]
    boxes[:,[0,2]]=torch.clamp(boxes[:,[0,2]], min=0, max=original_image_size[-1])
    boxes[:,[1,3]]=torch.clamp(boxes[:,[1,3]],min=0, max=original_image_size[0])

    return boxes

def xyxy2xywh(x):
    '''
    Convert Nx4 ndarray of [x1,y1,x2,y2] to [x,y,w,h] where x,y are the box center and (x1,y1) is top-left corner and
    (x2,y2) is the bottom right corner
    Args:
        x (ndarray/Tensor): Nx4 where N is the number of boxes and 4 is for [x1,y1,x2,y2]
    Returns:
        y (ndarray/Tensor): Nx4 where N is the number of boxes and 4 is for [x,y,w,h]
    '''
    if len(x)==0: return x
    y=x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:,0]=(x[:,0]+x[:,2])/2 # x center
    y[:,1]=(x[:,1]+x[:,3])/2 # y center
    y[:,2]=(x[:,2]-x[:,0]) # width
    y[:,3]=(x[:,3]-x[:,1]) # height
    assert (y[:,2:]>0).all(), 'width and hight must be positive'
    return y

def normalized_xywh2xyxy(x, w=None, h=None, shift_x=0., shift_y=0.):
    '''
    Convert [x,y,w,h] where (x,y) is the center of boxes and (w,h) is width and height to
    [x1,y1,x2,y2] where (x1,y1) is the top-left corner and (x2,y2) is the bottom right corner
    Args:
        x (ndarray/Tensor): Nx4 where N is the number of boxes and 4 for [x,y,w,h] with/without normalization
        w (int/float): width of the original image where the bounding box was annotated, used to denormalized input x,
            where pixel indices are normalized by width and height so they range from [0,1]. If None, x is defined in pixel units
        h (int/float): height of the original image where the bounding box was annotated, used to denormalized input x,
            where pixel indices are normalized by width and height so they range from [0,1]. If None, x is defined in pixel units
        shift_x (int/float): translation of the boxes along x axis
        shift_y (int/float): translation of the boxes along y axis 
    Returns:
        y (ndarray/Tensor): Nx4 where 4 is for [x1,y1,x2,y2] in pixel units of original images when annotation was defined
    '''
    if len(x)==0: return x
    y=x.clone() if isinstance(x, torch.Tensor) else x.copy()
    # top-left x = center-(width/2) -> denomalized -> shift by offset
    y[:,0]=(x[:,0]-x[:,2]/2)*w + shift_x
    # top-left y = center-(height/2) -> denomalized -> shift by offset
    y[:,1]=(x[:,1]-x[:,3]/2)*h + shift_y
    # bottom-right x = center+(width/2) -> denomalized -> shift by offset
    y[:,2]=(x[:,0]+x[:,2]/2)*w + shift_x
    # bottom-right y=center+(height/2) -> denomalized -> shift by offset
    y[:,3]=(x[:,1]+x[:,3]/2)*h + shift_y
    return y