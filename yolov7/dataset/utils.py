import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageOps, ExifTags
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

from collections import OrderedDict

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation': break

def exif_size(img):
    '''
    Get corrected EXIF size
    Args:
        img (PIL.Image)
    Returns:
        size (tuple[int]): image size after corrected EXIF orientation of width/height
    '''
    size=img.size # width,height
    try:
        rotation=dict(img._getexif().items())[orientation]
        if rotation in [6, 8]: size=(size[1],size[0]) # rotating 270, 90 degrees respectively, high -> width and vice versa
    except: pass
    return size
    
def read_image(image_fpath, target_size, correct_exif, eps=1.e-4):
    '''
    Read an image from file and resize so its large size (either width or height) match target size. 
    Args:
        image_fpath (str): path to image file
        target_size (int): target width and height (YOLO assumes a square image)
        correct_exif (bool): whether to correct for EXIF orientation
        eps (float): zero testing
    Returns:
        image (np.ndarray): HxWxC RGB image where max(H,W)=target_size and H may not equal W
        (H,W) (tuple[int]): original image size
    '''
    # we use cv2 (rather than torchvision.io.decode_image) because it is faster and we can use
    # cv2.INTER_AREA which yields better downsampled image if downsampling is required
    if correct_exif:
        img = Image.open(image_fpath) # RGB HxWx3
        #print(f'In dataset.utols.read_image: img width/height {img.size} correct EXIF width/hight {ImageOps.exif_transpose(img).size}', end=',')
        img = np.array(ImageOps.exif_transpose(img) ) # rotates according to EXIF
        #print(' ndarray ', img.shape)
    else:
        img=cv2.imread(image_fpath)  # BGR HxWx3
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB HxWx3
        
    H, W=img.shape[:2] # image size
    ratio=target_size/max(H,W)
    if abs(ratio-1.)>eps:
        interp=cv2.INTER_AREA if ratio<1. else cv2.INTER_LINEAR
        img=cv2.resize(img, (int(W*ratio), int(H*ratio)), interpolation=interp)
    return img, (H,W)
    
def segments2boxes(segments):
    '''
    Convert segmentation pixel indices to bounding boxes
    Args:
        segments (list[ndarray]): list of Mx2 where M may vary representing probably pixel indices of label (i.e.,
            pixel indices associated with certain segmentation)
    Returns:
        xywh (ndarray/Tensor): Nx4 where N=len(segments) is the number of bounding box and 4 for bounding box center (x,y)
            and width and height
    '''
    boxes=[]
    for s in segments:
        x, y=s.T # convert Nx2 to 2xN and N for x, and N for y
        boxes.append([x.min(), y.min(), x.max(), y.max()])
    return xyxy2xywh(np.array(boxes))

def get_hash(files):
    '''
    Return a single has value of a list of files
    '''
    return sum(os.path.getsize(f) for f in files if f is not None and os.path.isfile(f))

def parse_label_files(image_filepaths, label_filepaths):
    '''
    Read all label files and store in a dict
    Args:
        image_filepaths (list[str]): path to image files
        label_filepaths (list[str]): path to label txt files corresponding to image_filepaths
    Returns:
        annotation (OrderedDict): {im_file: [label, size, segments], hash:int, info:tuple[int]}, where label is Nx5 
            (for N bounding boxes of [class, x, y, w, h]), size is image size after correcting EXIF, and segments is 
            list[Mx2 ndarray] where M may vary is representing normalized indices of segmentation
    '''
    annotation=OrderedDict()
    n_missing=n_found=n_empty=n_duplicate=0
    for i, (im_file, lb_file) in enumerate(zip(image_filepaths, label_filepaths)):
        try:
            img=Image.open(im_file)
            assert img.verify() is None, f'{im_file} did not pass verify()'
            size=exif_size(img)
            segments=[] # instance segments
            assert all(s>9 for s in size), f'image size {size}<10 pixels'
            assert img.format.lower() in img_formats, f'invalid image file format {img.format}'
    
            segments=None # class and normalized indices of instance segmentation
            if lb_file is None or not lb_file.is_file():
                n_missing+=1 # label missing
                label=np.zeros((0,5), dtype=np.float32)
            else:
                n_found+=1
                with open(lb_file, 'r') as f: content=f.read()
                # each object segmentation (pixel indices) are recorded within 1 line. Thus, end-of-line separates each object
                # the first element in the line is class and the remaining are pixel indices of segmentation (I think)
                items=[c.split() for c in content.strip().split('\n')]
                if any(len(x)>8 for x in items): # we only compute bounding boxes on object covers more than 7 pixels
                    classes=np.array([x[0] for x in items], dtype=np.float32) # first element is object class. This is 1D array
                    # remainings are normalized pixel indices of segmentation
                    # list of Nx2 ndarray of segmentation pixel indices where N may vary
                    segments=[np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in items] 
                    # print('segments ', [s.shape for s in segments])
                    # print('classes ', classes.shape)
                    # Nx5 where N is the number of boxes (segmentations) and 5 is [cls, x, y, w, h]
                    # where cls is object class, (x,y) is the center of bounding box, (w,h) is the bounding box width/height
                    label=np.concatenate((classes[:,None], segments2boxes(segments)), axis=1) 
                if len(label)>0:
                    assert label.shape[-1]==5, f'labels expect to encode class, x, y, w, h'
                    assert (label>=0).all(), 'labels must be non-negative'
                    assert (label[:,1:]<=1).all(), 'bounding boxes must defined in normalized space'
                    assert np.unique(label, axis=0).shape[0]==label.shape[0], 'duplicate labels/bounding boxes'
                else:
                    n_empty+=1 # missing label
                    label=np.zeros((0,5), dtype=np.float32)
            annotation[im_file]=[label, size, segments]
        except Exception as e:
            n_duplicate+=1
            print(f'Warning: Ignoring corrupted images and/or label {im_file}: {e}')
            
    if n_found==0: print(f'Warning: no labels found')
    annotation['hash']=get_hash(label_filepaths+image_filepaths)
    annotation['info']={'n_found':n_found, 'n_missing':n_missing, 'n_empty':n_empty, 'n_duplicate':n_duplicate, 'total_img_files':i+1}
    return annotation