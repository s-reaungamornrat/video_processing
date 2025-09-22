import cv2
import random
import numbers
import numpy as np

from video_processing.yolov7.dataset.coords import normalized_xywh2xyxy

def augment_hsv(img, hgain=0.5,sgain=0.5,vgain=0.5):
    '''
    Modify HSV range
    Args:
        img (ndarray[uint8]): image
    Returns:
        out (ndarray[uint8]): image after adjusting HSV
    '''
    assert img.dtype==np.uint8, f'image must be unsigned int 8 but got {img.dtype}'
    random_gains=np.random.uniform(-1,1,3)*[hgain,sgain,vgain]+1. 
    hue, sat, val=cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    dtype=img.dtype
    
    x=np.arange(0,256,dtype=img.dtype)
    lut_hue=((x*random_gains[0])%180).astype(img.dtype) # OpenCV scale Hue from 0-360 to 0-180 degrees
    lut_sat=np.clip(x*random_gains[1], 0, 255).astype(img.dtype)
    lut_val=np.clip(x*random_gains[2], 0, 255).astype(img.dtype)
    img_hsv=cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(img.dtype)
    out=cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return out

def are_box_candidates(box1,box2,width_height_thr=2, aspect_ratio_thr=20, area_thr=0.1, eps=1e-16):
    '''
    Check whether box2, which is box1 after transformation, are still box candidates
    Args:
        box1 (ndarray[float]): Nx4 where N is the number of boxes and 4 is for x1,y1,x2,y2 in pixel units
        box2 (ndarray[float]): Nx4 where N is the number of boxes and 4 is for x1,y1,x2,y2 in pixel units
        width_height_thr (float): minimum width and height of valid boxes
        aspect_ratio_thr (float): maximum aspect ratio of valid boxes where aspect ratio is width/height and height/width
        area_thr (float): minimum area of valid boxes
    Returns:
        output (ndarray[bool]): N bool array and True for valid boxes and False for invalid boxes
    '''
    w1,h1=box1[:,2]-box1[:,0],box1[:,3]-box1[:,1]
    w2,h2=box2[:,2]-box2[:,0],box2[:,3]-box2[:,1]
    aspect_ratio=np.maximum(w2/(h2+eps), h2/(w2+eps))
    return (w2>width_height_thr)&(h2>width_height_thr)&(w2*h2/(w1*h1+eps) > area_thr)&(aspect_ratio<aspect_ratio_thr)

def random_affine(img, targets, degrees=0., translate=0.2, scale=0.9, padding_value=(114,114,114)):
    '''
    Args:
        img (ndarray[uint8]): image
        targets (ndarray[float32]): Nx5 where N is the number of boxes and 5 for class, x1,y1,x2,y2 in pixel units
        degrees (float): rotation angle in degree
        translate (float): offset to half pixel ratio to width and height
        scale (float): offset to 1 scaling
        padding_value (tuple[uint8]): padding value
    Returns:
        targets (ndarray[float32]): Nx5 where N is the number of boxes and 5 for class, x1,y1,x2,y2 in pixel units
    '''
    height, width=img.shape[:2]

    # move origin to the center of image instead of top-left
    C=np.eye(3)
    C[0,2]=-img.shape[1]/2 
    C[1,2]=-img.shape[0]/2
    # rotation and scale
    angle_=random.uniform(-degrees, degrees)
    scale_=random.uniform(1-scale, 1.1+scale)
    R=np.eye(3)
    R[:2]=cv2.getRotationMatrix2D(angle=angle_, center=(0,0), scale=scale_)
    # translation
    T=np.eye(3)
    T[0,2]=random.uniform(0.5-translate, 0.5+translate)*width # in pixel unit
    T[1,2]=random.uniform(0.5-translate, 0.5+translate)*height # in pixel unit
    # compose transform
    M=T@R@C
    # transform image
    img=cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=padding_value)
    
    # if no targets, return
    if len(targets)==0: return img, targets

    # transform targets
    xy=np.ones((len(targets)*4, 3))
    # Nx3 -> [x,y,1]
    xy[:,:2]=targets[:,[1,2,3,4,1,4,3,2]].reshape(len(targets)*4, 2) # x1,y1,x2,y2,x1,y2,x2,y1
    xy=xy@M.T
    xy=xy[:,:2].reshape(len(targets),8) # remove 1
    x=xy[:,[0,2,4,6]] # x1,x2,x1,x2
    y=xy[:,[1,3,5,7]] # y1,y2,y2,y1
    tfm_xyxy=np.vstack([x.min(1),y.min(1),x.max(1),y.max(1)]).T # Nx4
    # clip pixel indices
    tfm_xyxy[:,[0,2]]=tfm_xyxy[:,[0,2]].clip(min=0, max=width-1)
    tfm_xyxy[:,[1,3]]=tfm_xyxy[:,[1,3]].clip(min=0, max=height-1)

    # filter box candidates
    valid=are_box_candidates(box1=targets[:,1:5],box2=tfm_xyxy )
    targets=targets[valid] # Nx5 where N is the number of valid boxes and 5 is for class,x1,y1,x2,y2 in pixel units
    targets[:,1:5]=tfm_xyxy[valid]

    return img, targets
    
def data_to_target_size(image, labels, target_size, color=(114,114,114), scale_up=False, eps=1.e-4):
    '''
    Change image size as well as label to match target size
    Args:
        image (ndarray): HxWx3 uint8 image array
        labels (ndarray/Tensor): Nx5 where N is the number of boxes and 5 is for class, (x, y, w, h) in normalized space
        target_size (int/tuple): number of pixels in x and y (i.e., square image)
        color (tuple[int]): default padding value
        scale_up (bool): whether to allow scale up
    Returns:
        image (ndarray): HxWx3 uint8 image array
        labels (ndarray/Tensor): Nx5 where N is the number of boxes and 5 is for class, (x1, y1, x2, y2) in pixel unit
        shift (tuple[float]): shift along x and y
    '''
    image_size=image.shape[:2] # H W
    if isinstance(target_size, numbers.Number): target_size=[target_size, target_size]
    
    ratio=[o/i for i, o in zip(image_size, target_size)] # H W
    if not scale_up: ratio=[min(r, 1) for r in ratio]
    ratio=min(ratio) if np.random.rand()<0.5 else max(ratio) # 1 ratio for both so we preserve relative scale of each objects in image
    scaled_size=[int(s*ratio) for s  in image_size] # H, W
    if any(m!=n for m,n in zip(scaled_size,image_size)): image = cv2.resize(image, scaled_size[::-1], interpolation=cv2.INTER_LINEAR)
    # crop if scaled_image is bigger than target size
    cropw=croph=0
    if any(s>t for s, t in zip(scaled_size, target_size)):
        top_left=[int(s/2 - t/2) for t, s in zip(target_size, image.shape[:2])] # H, W
        image=image[top_left[0]:target_size[0]+top_left[0], top_left[1]:target_size[1]+top_left[1]]
        scaled_size=image.shape[:2]
        cropw,croph=top_left[1], top_left[0]
        
    # pad if scaled_image smaller than target size
    difference=[t-o for t, o in zip(target_size, scaled_size)]
    padw=padh=0
    if any(d>eps for d in difference):
        # divide padding into 2 sides
        padding=[d/2 for d in difference] # pady, padx
        top,bottom=int(round(padding[0]-0.1)), int(round(padding[0]+0.1))
        left,right=int(round(padding[1]-0.1)), int(round(padding[1]+0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        padw,padh=padding[1],padding[0]
    
    # modify bounding box location
    shift_x=padw-cropw
    shift_y=padh-croph
    if labels.size>0:  # normalized xywh to pixel xyxy format
        labels[:, 1:]= normalized_xywh2xyxy(labels[:, 1:], w=ratio * image_size[1], h=ratio * image_size[0], shift_x=shift_x, shift_y=shift_y)

    # clip pixel indices
    labels[:,[1,3]]=labels[:,[1,3]].clip(min=0, max=image.shape[1]-1) # width
    labels[:,[2,4]]=labels[:,[2,4]].clip(min=0, max=image.shape[0]-1) # height
    
    return image, labels, (shift_x, shift_y)