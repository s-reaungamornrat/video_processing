import os
import sys
import cv2
import yaml
import numbers

import torch
import torchvision
from torchvision.ops.boxes import masks_to_boxes

import numpy as np
from PIL import Image

from video_processing.yolov7.dataset.utils import read_image, image_n_boxes_to_target_size, exif_size
from video_processing.yolov7.dataset.coords import xyxy2xywh
from video_processing.yolov7.dataset.modification import random_affine, augment_hsv

class PennFudanDataset(torch.utils.data.Dataset):

    def __init__(self, root, image_dirname, mask_dirname, hyp, indices=None, img_size=1280, augment=False, correct_exif=True,
                 padding_value=(114,114,114)):
        '''
        Input:
            root (str): path to data folder
            image_dirname (str): name of subfolder containing images
            mask_dirname (str): name of subfolder containing masks
            hyp (dict): hyperparameters for data augmentation
            indices (sequence): indices to data used
            img_size (int): input image size
        '''
        assert isinstance(img_size, numbers.Number), f'img_size must be integer since the model assume a square input image'
        
        self.augment=augment
        self.img_size=img_size
        self.hyp=hyp
        self.correct_exif=correct_exif
        self.padding_value=padding_value
        self.image_dirpath=os.path.join(root, image_dirname).encode('utf-8')
        self.mask_dirpath=os.path.join(root, mask_dirname).encode('utf-8')

        self.image_fnames=np.asarray(sorted(s.encode('utf-8') for s in os.listdir(self.image_dirpath.decode('utf-8')))) # decode s.decode('utf-8')
        self.mask_fnames=np.asarray(sorted(s.encode('utf-8') for s in os.listdir(self.mask_dirpath.decode('utf-8')))) # decode s.decode('utf-8')
        if indices is None: indices=np.arange(len(self.image_fnames))
        if max(indices)<len(self.image_fnames):
            self.image_fnames=[self.image_fnames[idx] for idx in indices]
            self.mask_fnames=[self.mask_fnames[idx] for idx in indices]

        # make sure that files are ordered consistently and record image size
        image_sizes=[]
        for im_file, msk_file in zip(self.image_fnames, self.mask_fnames):
            im_file=im_file.decode('utf-8')
            msk_file=msk_file.decode('utf-8')
            assert all(os.path.splitext(im_file)[0]==os.path.splitext(x)[0].replace('_mask', '') for x in [msk_file])
            # get image size
            image_sizes.append(exif_size(Image.open(os.path.join(self.image_dirpath.decode('utf-8'), im_file))))
            
        self.image_sizes=np.array(image_sizes, dtype=np.float64) # Nx2 where N is the number of images and 2 for width, height in this order
        
    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        '''
        Returns:
            image (Tensor[uint8]): CxHxW image
            targets (Tensor[float]): Nx6 where N is the number of boxes, 6 for [image-idx, class-idx, x,y,w,h],
                where (x,y) is the box center in normalized space and (w,h) is width/height in normalized space
        '''

        idx=idx%len(self.image_fnames)

        image_fpath=os.path.join(self.image_dirpath.decode('utf-8'), self.image_fnames[idx].decode('utf-8'))
        mask_fpath=os.path.join(self.mask_dirpath.decode('utf-8'), self.mask_fnames[idx].decode('utf-8'))

        image,(H0,W0)=read_image(image_fpath, target_size=self.img_size, correct_exif=self.correct_exif)
        mask,_=read_image(mask_fpath, target_size=self.img_size, correct_exif=self.correct_exif, mode='nearest')
        
        ## --- determine class-index, image-index, bounding boxes in x1,y1,x2,y2 format in pixel unit
        mask=torch.from_numpy(mask)[None] # CxHxW -> CxYxX
        # instances are encoded with different colors
        obj_ids=torch.unique(mask)
        # first id is the background, so remove it
        obj_ids=obj_ids[1:]
        n_objs=len(obj_ids)
        # first we split the color-encoded mask into a set of binary masks
        masks=(mask==obj_ids[:,None,None]).to(dtype=torch.uint8) # LxHxW or LxYxX where L is the number of objects
        class_idx=torch.ones(n_objs,dtype=torch.long)[:,None] # Nx1 where N is the number of boxes. Note there is only 1 class
        image_id=torch.tensor([idx]).repeat(n_objs)[:,None] # Nx1
        boxes=masks_to_boxes(masks) # Nx4 where N is the number of objects and 4 for x-min,y-min,x-max,y-max
 
        ## --- adjust image and boxes to target size. We actually can start by operating on mask and then get boxes from mask
        # but we implement this way in case, for some data, we do not have masks
        image, ratio, (shift_x, shift_y)=image_n_boxes_to_target_size(image, boxes, target_size=self.img_size, scale_up=self.augment,
                                                                      color=(114,114,114), eps=1.e-4)
        targets=torch.cat([class_idx, boxes], dim=-1)
        if self.augment:
            # we should have just passed the boxes but since we already have this function from yolov7 which was written first for coco
            # we are just going to format the targets and pass them to the function
            image,targets=random_affine(img=image, targets=targets.numpy(), degrees=self.hyp['degrees'], translate=self.hyp['translate'], 
                                     scale=self.hyp['scale'], padding_value=self.padding_value)
            image=augment_hsv(image, hgain=self.hyp['hsv_h'],sgain=self.hyp['hsv_s'],vgain=self.hyp['hsv_v'])
        if len(targets)>0:
            # convert x1,y1,x2,y2 in pixel unit to x,y,w,h where (x,y) is the center of box and (w,h) is width and height
            targets[:,1:]=xyxy2xywh(targets[:,1:])
            # normalize so width and height is between 0-1
            targets[:,[1,3]]/=image.shape[1] # normalize x direction by width
            targets[:,[2,4]]/=image.shape[0] # normalize y direction by height
        
        if self.augment:
            if np.random.rand()<self.hyp['flipud']: # flip up down
                image=np.flipud(image)
                if len(targets)>0: targets[:,2]=1-targets[:,2]
            if np.random.rand()<self.hyp['fliplr']: # flip left right
                image=np.fliplr(image)
                if len(targets)>0: targets[:,1]=1-targets[:,1]

        # Nx6 where N is the number of boxes and 6 for (index, object class, x, y, w, h)
        # index is the index of images in a batch with which the boxes are associated with
        # x,y,w,h with (x,y) the normalized center of boxes and (w,h) the normalized width and height of boxes
        labels=torch.zeros((len(targets), 6))
        if len(targets)>0: labels[:,1:]=torch.from_numpy(targets) if not isinstance(targets, torch.Tensor) else targets
            
        # image numpy HxWx3 to torch 3xHxW
        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        return torch.from_numpy(image), labels, ratio, (shift_x, shift_y)

    @staticmethod
    def collate_fn(batch):
        images, labels, ratios, shifts = zip(*batch)  # each img, label, path are tuple of item in each batch
        for i, l in enumerate(labels):
            l[:, 0] = i  # add target-image index so we know which boxes associated with which images
        return torch.stack(images, 0), torch.cat(labels, 0), ratios, shifts
        