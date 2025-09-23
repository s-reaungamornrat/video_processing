import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageOps, ExifTags
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

from collections import OrderedDict
from video_processing.yolov7.dataset.coords import xyxy2xywh

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
    
def read_image(image_fpath, target_size, correct_exif, eps=1.e-4, mode='linear'):
    '''
    Read an image from file and resize so its large size (either width or height) match target size. 
    Args:
        image_fpath (str): path to image file
        target_size (int): target width and height (YOLO assumes a square image)
        correct_exif (bool): whether to correct for EXIF orientation
        eps (float): zero testing
        mode (str): interpolation method, i.e., linear or nearest
    Returns:
        image (np.ndarray): HxWxC RGB image where max(H,W)=target_size and H may not equal W
        (H,W) (tuple[int]): original image size
    '''
    assert mode in ['linear', 'nearest']
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
    #print('target_size ', target_size, ' max(H,W) ', max(H,W))
    ratio=target_size/max(H,W)
    if abs(ratio-1.)>eps:
        if mode=='linear': interp=cv2.INTER_AREA if ratio<1. else cv2.INTER_LINEAR
        else: interp=cv2.INTER_NEAREST
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

def image_n_boxes_to_target_size(image, boxes, target_size, scale_up=True, color=(114,114,114), eps=1.e-4):
    '''
    Convert an image and boxes to target size. This function returns in-place box modification
    Args:
        image (ndarray[uint8]): image
        boxes (ndarray/Tensor): Nx4 where N is the number of boxes and 4 for x1,y1,x2,y2 in pixel unit
        target_size (int): target size of square image
    Returns:
        image (ndarray[uint8]): image after modification
        ratio (float): ratio of target_size/original-size for use in converting boxes back via division
        shift (tuple[float]): translation of boxes due to cropping and padding, i.e., (shift_x, shift_y) in pixel units
    '''
    image_size=image.shape[:2] # H,W

    # rescale image
    ratio=[target_size/i for i in image_size] # H W
    if not scale_up: ratio=[min(r, 1) for r in ratio]
    ratio=min(ratio) if np.random.rand()<0.5 else max(ratio) # 1 ratio for both so we preserve relative scale of each objects in image
    scaled_size=[int(s*ratio) for s  in image_size] # H, W
    if any(m!=n for m,n in zip(scaled_size,image_size)): image = cv2.resize(image, scaled_size[::-1], interpolation=cv2.INTER_LINEAR)
    # rescale boxes
    if len(boxes)>0:  # assuming pixel xyxy format
        box_widths, box_heights=(boxes[:,2:]-boxes[:,:2]).T  # Nx2 -> 2xN
        boxes[:,[0,1]]*=ratio
        boxes[:,2]=boxes[:,0]+ratio*box_widths
        boxes[:,3]=boxes[:,1]+ratio*box_heights

    #crop if scaled_image is bigger than target size
    cropw=croph=0
    if any(s>target_size for s in scaled_size):
        top_left=[int(np.ceil(s/2 - target_size/2)) for s in image.shape[:2]] # H, W
        image=image[top_left[0]:target_size+top_left[0], top_left[1]:target_size+top_left[1]]
        scaled_size=image.shape[:2]
        cropw,croph=top_left[1], top_left[0]
        if len(boxes)>0:  # assuming pixel xyxy format
            boxes[:,[0,2]]-=cropw
            boxes[:,[1,3]]-=croph
        
    # pad if scaled_image smaller than target size
    difference=[target_size-o for o in scaled_size]
    padw=padh=0
    if any(d>eps for d in difference):
        # divide padding into 2 sides
        padding=[d/2 for d in difference] # pady, padx
        top,bottom=int(round(padding[0]-0.1)), int(round(padding[0]+0.1))
        left,right=int(round(padding[1]-0.1)), int(round(padding[1]+0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        padw,padh=padding[1],padding[0]
    
        if len(boxes)>0:  # assuming pixel xyxy format
            boxes[:,[0,2]]+=padw
            boxes[:,[1,3]]+=padh

    # check whether boxes are still inside image
    if len(boxes)>0:  # assuming pixel xyxy format
        # clip pixel indices
        boxes[:,[0,2]]=boxes[:,[0,2]].clip(min=0, max=image.shape[1]-1) # width
        boxes[:,[1,3]]=boxes[:,[1,3]].clip(min=0, max=image.shape[0]-1) # height
    shift_x, shift_y=padw-cropw, padh-croph
    return image, ratio, (shift_x, shift_y)

def inverse_image_n_boxes_to_target_size(boxes, ratio, shift_x, shift_y):
    '''
    Inverse operations applied to boxes in image_n_boxes_to_target_size
    Args:
        boxes (ndarray/Tensor): Nx4 where N is the number of boxes and 4 for x1,y1,x2,y2 in pixel units
        ratio (float): ratio of target_size/original-size for use in converting boxes back via division
        shift_x (float): translation of boxes due to cropping and padding along x or width in pixel units
        shift_y (float): translation of boxes due to cropping and padding along y or height in pixel units
    Returns:
        boxes  (ndarray/Tensor): Nx4 where N is the number of boxes and 4 for x1,y1,x2,y2 in pixel units
    '''
    boxes=boxes.clone() if isinstance(boxes, torch.Tensor) else np.copy(boxes)
    boxes[:,[0,2]]-=shift_x
    boxes[:,[1,3]]-=shift_y
    width,height=(boxes[:,2:]-boxes[:,:2]).T
    width/=ratio
    height/=ratio
    boxes[:,[0,1]]/=ratio
    boxes[:,2]=boxes[:,0]+width
    boxes[:,3]=boxes[:,1]+height
    return boxes