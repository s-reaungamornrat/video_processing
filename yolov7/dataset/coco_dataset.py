import os
import random
from pathlib import Path

import torch
import numpy as np

from video_processing.yolov7.dataset.coords import xyxy2xywh
from video_processing.yolov7.dataset.modification import data_to_target_size, random_affine, augment_hsv
from video_processing.yolov7.dataset.utils import read_image, parse_label_files

class LoadImagesAndLabels(torch.utils.data.Dataset):
    
    def __init__(self, data_dirpath, image_paths, img_size=640, augment=False, hyp=None, n_data=None, correct_exif=True,
                 padding_value=(114,114,114)):
        '''
        This is the simplified version of LoadImagesAndLabels in https://github.com/WongKinYiu/yolov7/blob/main/utils/datasets.py#L353
        Args:
            data_dirpath (str): path to the main folder containing images/ and labels/
            image_paths (str): path to txt file listing all paths to images in images/, each path must contain images/
            img_size (int): image size, assuming square image so only need to know number of pixels in one direction
            rect (bool): rectangular training
        '''
        self.img_size=img_size
        self.augment=augment
        self.hyp=hyp
        self.data_dirpath=Path(data_dirpath)
        self.correct_exif=correct_exif
        self.padding_value=padding_value
        
        with open(image_paths, "r") as file: content = file.read().split('\n')   
            
        image_filepaths, label_filepaths=[],[]
        for i, file in enumerate(content):
            if n_data is not None and i>n_data-1: break
            relative_image_fpath=file[file.find('images'):]
            relative_label_fpath=os.path.splitext(relative_image_fpath.replace('images', 'labels'))[0]+".txt"
            image_fpath=self.data_dirpath/relative_image_fpath
            if not image_fpath.is_file(): continue
            #assert image_fpath.is_file(), f'{image_fpath} does not exist'
            label_fpath=self.data_dirpath/relative_label_fpath
            image_filepaths.append(image_fpath)
            label_filepaths.append(label_fpath if label_fpath.is_file() else None)

        cache_path=Path(label_filepaths[0]).parent.with_suffix('.cache') # store preparsing labels from last time
        print(f'In dataset.coco_dataset.__init__ save cache to {cache_path} cache_path.is_file() {cache_path.is_file()}')
        if cache_path.is_file():
            cache, exists=torch.load(cache_path, weights_only=False), True
        else: # we parse label file and store it in cache for quick load next time
            print(f'In dataset.coco_dataset.__init__ image_filepaths {image_filepaths[:5]}')
            cache=parse_label_files(image_filepaths=image_filepaths, label_filepaths=label_filepaths)
            torch.save(cache, cache_path)
            
        if 'info' in cache: cache.pop('info');
        if 'hash' in cache: cache.pop('hash')
        labels, shapes, self.segments=zip(*cache.values())
        self.labels=labels # list of [class, (x,y,w,h)] where the later are normalized and (x,y) is the box center and (w,h) is width/height
        self.image_sizes=np.array(shapes, dtype=np.float64) # Nx2 where N is the number of images and 2 for width, height in this order
        self.image_filepaths, self.label_filepaths=[],[]
        for im_fpath in cache.keys():
            label_fpath=Path(os.path.splitext(relative_image_fpath.replace('images', 'labels'))[0]+".txt")
            self.image_filepaths.append(im_fpath)
            self.label_filepaths.append(label_fpath if label_fpath.is_file() else None)
            
    def __len__(self):
        return len(self.image_filepaths)

    def __getitem__(self, idx):
        '''
        Args:
            image (Tensor): CxHxW uint8 image
            labels_out (Tensor): Nx6 where N is the number of boxes and 6 is for [0, object-class, (x,y,w,h)] where
                (x,y) is the normalized center of boxes and (w,h) is the normalized width and height
            image_fpath (str): image file path
            shapes (tuple): ((H0,W0), (factor_w, factor_h), (shift_x, shift_y)) where H0,W0 are original image size,
                            (factor_w, factor_h) is the ratio of new/original for width and height and 
                            (shift_x, shift_y) is the shift of pixel indices
        '''

        # preventing index out of bound
        idx%=len(self.image_filepaths)
        
        # load image
        image,(H0,W0)=read_image(self.image_filepaths[idx], target_size=self.img_size, correct_exif=self.correct_exif)
        height, width=image.shape[:2]

        # check image image is not RGB, make it RGB
        if image.ndim==2: image=np.dstack([image]*3) 
            
        # convert data to match target size and change normalized x,y,w,h to x1,y1,x2,y2 in pixel unit
        img, labels, shift=data_to_target_size(image=image.copy(), labels=self.labels[idx].copy(), target_size=self.img_size, 
                                         scale_up=self.augment and np.random.rand()>0.1)
        shapes=(H0,W0), ((height/H0, width/W0),shift) # # for COCO mAP rescaling
        # print(f'In dataset.coco_dataset.__getitem__ img {img.shape}')
        # print(f'In dataset.coco_dataset.__getitem__ labels x1 {any(x<0 for x in labels[:,1].tolist())} y1 {any(x<0 for x in labels[:,2].tolist())}')
        # print(f'In dataset.coco_dataset.__getitem__ labels x2 {any(x>=img.shape[1] for x in labels[:,3].tolist())} y2 {any(x>=img.shape[0] for x in labels[:,4].tolist())}')
        
        if self.augment:
            img,labels=random_affine(img=img, targets=labels, degrees=self.hyp['degrees'], translate=self.hyp['translate'], 
                                     scale=self.hyp['scale'], padding_value=self.padding_value)
            img=augment_hsv(img, hgain=self.hyp['hsv_h'],sgain=self.hyp['hsv_s'],vgain=self.hyp['hsv_v'])
        
        if len(labels)>0:
            # convert x1,y1,x2,y2 in pixel unit to x,y,w,h where (x,y) is the center of box and (w,h) is width and height
            labels[:,1:]=xyxy2xywh(labels[:,1:])
            # normalize so width and height is between 0-1
            labels[:,[1,3]]/=img.shape[1] # normalize x direction by width
            labels[:,[2,4]]/=img.shape[0] # normalize y direction by height
        
        if self.augment:
            if np.random.rand()<self.hyp['flipud']: # flip up down
                img=np.flipud(img)
                if len(labels)>0: labels[:,2]=1-labels[:,2]
            if np.random.rand()<self.hyp['fliplr']: # flip left right
                img=np.fliplr(img)
                if len(labels)>0: labels[:,1]=1-labels[:,1]

        # Nx6 where N is the number of boxes and 6 for (index, object class, x, y, w, h)
        # index is the index of images in a batch with which the boxes are associated with
        # x,y,w,h with (x,y) the normalized center of boxes and (w,h) the normalized width and height of boxes
        labels_out=torch.zeros((len(labels), 6))
        if len(labels)>0: labels_out[:,1:]=torch.from_numpy(labels)
            
        # image numpy HxWx3 to torch 3xHxW
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
            
        return torch.from_numpy(img), labels_out, self.image_filepaths[idx], shapes
        
    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # each img, label, path are tuple of item in each batch
        for i, l in enumerate(label):
            l[:, 0] = i  # add target-image index so we know which boxes associated with which boxes
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
        