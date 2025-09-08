import os

import torch
import torchvision
import numpy as np

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T

def get_transform(is_train):#, image_mean, image_std, min_size, max_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR):
    '''
    Input:
    is_train (bool): whether to get transform for training data augmentation
    image_mean (tuple): mean intensity, e.g., model.transform.image_mean
    image_std (tuple): intensity std, e.g., model.transform.image_std
    min_size (int): minimum image size, e.g., model.transform.min_size
    max_size (int): maximum image size, e.g., model.transform.max_size
    '''
    # We note that resizing and image normalization are done within forward function inside the model, 
    # see line#96 in https://github.com/pytorch/vision/blob/main/torchvision/models/detection/generalized_rcnn.py
    transforms=[]
    if is_train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        #transforms.append(T.RandomShortestSize(min_size=min_size, max_size=max_size, interpolation=interpolation))
    #else: transforms.append(T.Resize(interpolation=interpolation, max_size=max_size))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    #transforms.append(T.Normalize(mean=image_mean, std=image_std))
    return T.Compose(transforms)


class PennFudanDataset(torch.utils.data.Dataset):

    def __init__(self, root, image_dirname, mask_dirname, annotation_dirname, transforms, indices=None):
        '''
        Input:
            root (str): path to data folder
            image_dirname (str): name of subfolder containing images
            mask_dirname (str): name of subfolder containing masks
            annotation_dirname (str): name of subfolder containing annotations
            transforms (callable): image transformation for preprocessing and augmentation 
            indices (sequence): indices to data used
        '''

        self.transforms=transforms
        self.image_dirpath=os.path.join(root, image_dirname).encode('utf-8')
        self.mask_dirpath=os.path.join(root, mask_dirname).encode('utf-8')
        self.annotation_dirpath=os.path.join(root, annotation_dirname).encode('utf-8')


        self.image_fnames=np.asarray(sorted(s.encode('utf-8') for s in os.listdir(self.image_dirpath.decode('utf-8')))) # decode s.decode('utf-8')
        self.mask_fnames=np.asarray(sorted(s.encode('utf-8') for s in os.listdir(self.mask_dirpath.decode('utf-8')))) # decode s.decode('utf-8')
        self.bbox_fnames=np.asarray(sorted(s.encode('utf-8') for s in os.listdir(self.annotation_dirpath.decode('utf-8')))) # decode s.decode('utf-8')
        if max(indices)<len(self.image_fnames):
            self.image_fnames=[self.image_fnames[idx] for idx in indices]
            self.mask_fnames=[self.mask_fnames[idx] for idx in indices]
            self.bbox_fnames=[self.bbox_fnames[idx] for idx in indices]

        # make sure that files are ordered consistently
        for im, msk, bb in zip(self.image_fnames, self.mask_fnames, self.bbox_fnames):
            im=im.decode('utf-8')
            msk=msk.decode('utf-8')
            bb=bb.decode('utf-8')
            assert all(os.path.splitext(im)[0]==os.path.splitext(x)[0].replace('_mask', '') for x in [bb, msk])

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):

        idx=idx%len(self.image_fnames)
        #print(f'In PennFudanDataset.__getitem__ idx {idx} fname {self.image_fnames[idx].decode("utf-8")}')

        image_fpath=os.path.join(self.image_dirpath.decode('utf-8'), self.image_fnames[idx].decode('utf-8'))
        mask_fpath=os.path.join(self.mask_dirpath.decode('utf-8'), self.mask_fnames[idx].decode('utf-8'))
        bbox_fpath=os.path.join(self.annotation_dirpath.decode('utf-8'), self.bbox_fnames[idx].decode('utf-8'))

        # read image/mask
        image=read_image(image_fpath) # CxHxW -> CxYxX
        mask=read_image(mask_fpath) # CxHxW -> CxYxX

        # instances are encoded with different colors
        obj_ids=torch.unique(mask)
        # first id is the background, so remove it
        obj_ids=obj_ids[1:]
        num_objs=len(obj_ids)

        # split the color-encoded mask into a set of binary masks
        masks=(mask==obj_ids[:,None,None]).to(dtype=torch.uint8) # LxHxW or LxYxX where L is the number of objects

        # get bounding box coordinates for each mask
        boxes=masks_to_boxes(masks) # Nx4 where N is the number of objects and 4 for x-min,y-min,x-max,y-max

        # there is only one class
        labels=torch.ones((num_objs),dtype=torch.long)
        image_id=idx
        area=(boxes[:, 2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]) # N

        # suppose all instances are not crowd
        iscrowd=torch.zeros((num_objs,), dtype=torch.long)

        # Wrap sample and targets into torchvision tv_tensors
        img=tv_tensors.Image(image) # CxHxW uint8

        target={}
        target['boxes']=tv_tensors.BoundingBoxes(boxes, format='XYXY', canvas_size=F.get_size(img))
        target['masks']=tv_tensors.Mask(masks) # LxHxW
        target['labels']=labels # 1D labels of size N
        target['image_id']=image_id # int
        target['area']=area # 1D area of size N
        target['iscrowd']=iscrowd # 1D tensor of size N

        if self.transforms is not None: img, target = self.transforms(img, target)

        return img, target

if __name__ == "__main__":
    
    def find_lib(dirpath, lib_name='mask_rcnn'):
        '''
        Find the relative path to the library from the input dirpath
        Input:
            dirpath (str): current path where the script is
            lib_name (str): the name of the library, typically folder name
        '''
        pathlen=len(dirpath.split(os.path.sep))
        for i in range(pathlen):
            if lib_name in os.listdir(dirpath): return '../'*i
            dirpath=os.path.dirname(dirpath)
            
    sys.path.append(find_lib(os.getcwd(), lib_name='mask_rcnn'))
    from mask_rcnn.utils.utils import display_image_overlay


    dataset=PennFudanDataset(root='data/PennFudanPed', image_dirname='PNGImages', mask_dirname='PedMasks', annotation_dirname='Annotation', transforms=None)
    print(f'There are {len(dataset)} data in the dataset')

    image, target=dataset[len(dataset)-1]
    print('image ', image.shape, type(image), image.dtype)
    print('target ', {k:(v if k!='masks' else v.shape) for k, v in target.items()})
    annotation={'bbox':[[box.squeeze()[2*i:(2*i+2)] for i in range(2)] for box in target['boxes'].split(1, dim=0)],
                'label':[f'{l}' for l in target['labels']]}
    # does not work here since we ignore background labels
    display_image_overlay(image=image, mask=torch.argmax(target['masks'], dim=0, keepdim=True), annotation=annotation)
    display_image_overlay(image=image, mask=target['masks'].sum(dim=0, keepdim=True), annotation=annotation)
         