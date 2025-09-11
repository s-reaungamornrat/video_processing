import math
import torch

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