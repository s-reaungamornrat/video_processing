import math
import torch

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