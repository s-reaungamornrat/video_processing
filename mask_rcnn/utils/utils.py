import os,re
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import cycle

def read_annotation(annotation_fpath):
    '''
    Read annotation file of PennFudanPed dataset
    Input:
    annotation_fpath (str): path to annotation
    Output:
    annotation (dict): dict of sequences of object numbers, labels, and bounding boxes, as {obj_num:[], label:[], bbox:[] }
    '''
    assert os.path.isfile(annotation_fpath), f'{annotation_fpath} does not exist'
    
    with open(annotation_fpath, 'r') as f:  annotation_txt=f.read().split('\n')
    
    # read annotation
    annotation={'obj_num':[], 'label':[], 'bbox':[] }
    for line in annotation_txt:
        if re.findall('^Original label', line):
            idx=np.nonzero([word.isnumeric() for word in line.split()])[0].item()
            obj_n=int(line.split()[idx])
            label=line.split()[idx+1].strip('"')
        elif re.findall('^Bounding box', line):
            bbox=[[int(x.strip(' ( ) ')) for x in term.split(',')] for term in line.split(':')[-1].split('-')]
            annotation['obj_num'].append(obj_n)
            annotation['label'].append(label)
            annotation['bbox'].append(bbox)
    return annotation

def display_image_overlay(image, mask=None, annotation=None, axis_off=True):
    '''
    Display each image and mask overlay by annotation
    Input:
    image (tensor): CxHxW or CxYxX uint8
    mask (tensor): 1xHxW or 1xYxX uint8
    annotation (dict): dict containing sequences of bounding box (under key bbox) and label (under key label)
    '''
    if annotation is not None: assert all(key in annotation for key in ['bbox', 'label'])
    
    # bounding box colors
    colours = cycle(['r', 'g', 'b', 'm', 'c'])
    
    # text format for label overlaid on images
    font = {'family': 'serif',
          'color':  'cyan',
          'weight': 'normal',
          'size': 10,
          }
    
    fig, axs = plt.subplots(1, 2 if mask is not None else 1, figsize=(16, 8))
    if mask is not None: axs[0].imshow(image.permute(1,2,0)) # HxWxC / YxXxC
    else: axs.imshow(image.permute(1,2,0)) # HxWxC / YxXxC
    if annotation is not None:
        for i, (bbox, color) in enumerate(zip(annotation['bbox'], colours)):
            width,height=[e-s for e, s in zip(bbox[-1], bbox[0])]
            # Create a Rectangle patch
            rect = patches.Rectangle([x-1 for x in bbox[0]], width,height, linewidth=2,edgecolor=color, facecolor='none')
            # Add the patch to the Axes
            axs[0].add_patch(rect)
            axs[0].text(bbox[0][0],bbox[-1][1], annotation['label'][i], fontdict=font)
            if axis_off: axs[0].set_axis_off()
    
    if mask is not None: axs[1].imshow(mask.permute(1,2,0)) # HxWxC / YxXxC
    if annotation is not None:
        for i, (bbox, color) in enumerate(zip(annotation['bbox'], colours)):
            width,height=[e-s for e, s in zip(bbox[-1], bbox[0])]
            # Create a Rectangle patch
            rect = patches.Rectangle([x-1 for x in bbox[0]], width,height, linewidth=2,edgecolor=color, facecolor='none')
            # Add the patch to the Axes
            axs[1].add_patch(rect)
            axs[1].text(bbox[0][0],bbox[-1][1], annotation['label'][i], fontdict=font)
            if axis_off: axs[1].set_axis_off()
    plt.show(block=True)
     