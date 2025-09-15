import yaml
import math
import numbers
from copy import deepcopy

import torch
import torch.nn as nn

from video_processing.yolov7.utils.general import make_divisible
from video_processing.yolov7.models.common import check_anchor_order
from video_processing.yolov7.models.common import Conv, ReOrg, IAuxDetect, SPPCSPC, Concat

class Model(nn.Module):
    
    def __init__(self, cfg='yolor.yaml', ch=3,nc=None, anchors=None, verbose=False):
        super(Model, self).__init__()
        
        assert ch is not None and isinstance(ch, numbers.Number)
        
        if isinstance(cfg, dict): self.yaml=cfg
        else:
            import yaml
            self.yaml_file=cfg
            with open(cfg) as f: self.yaml=yaml.load(f, Loader=yaml.SafeLoader) # model dict
        # get the number of input channels
        self.yaml['ch']=self.yaml.get('ch', ch)
        if nc is not None:
            if 'nc' in self.yaml: 
                assert nc==self.yaml['nc'], f'number of class in cfg {self.yaml["nc"]} mismatch with model arg nc {nc}'
            else: self.yaml['nc']=nc
        assert isinstance(self.yaml['nc'], numbers.Number)
        self.model, self.save=parse_model(deepcopy(self.yaml), ch=[ch], verbose=verbose)
        self.names=[f'{i}' for i in range(self.yaml['nc'])] # default name

        # determine the model stride dynamically and then 
        if verbose: print('ch_in ', ch)
        self.__setup_stride_and_anchors(ch_in=ch, verbose=verbose)
        # initialize bias of detection heads requiring knowledge of stride since we need to approximate
        # the number of grid cell per images
        self._initialize_aux_biases(verbose=verbose) 

    def __setup_stride_and_anchors(self, ch_in, verbose=False):
        '''
        Dynamically determine strides based on model architecture dynamically generated through parse_model function. The strides 
        are determined per scale/level. The stride for each level then is used to normalize anchor definition from the image pixel unit to 
        the feature-map grid cell unit. In other words, strides are used to convert anchors to grid unit per level/scale.
        Args:
            ch_in (int): the number of input channels, e.g., 3 for RGB
        '''
        # Build strides, anchors
        m=self.model[-1] # detector
        # we assuming input image size since stride of the model will not change if input size change
        image_size=256 # 2x min stride ... assuming input image size
        # output is a sequence of length 8 but the same tensor sizes for the first 4 and the last 4
        # output size  [torch.Size([1, 6, 32, 32, 85]), torch.Size([1, 6, 16, 16, 85]), torch.Size([1, 6, 8, 8, 85]), torch.Size([1, 6, 4, 4, 85])]
        outputs=self.forward(torch.zeros(1, ch_in, image_size, image_size))[:4] 
        if verbose: print('outputs ', [v.shape for v in outputs])
        # assuming images having the same width and height
        m.stride=torch.tensor([image_size/v.shape[-2] for v in outputs])
        if verbose: print('m.stride ', m.stride)
        check_anchor_order(m)
        if verbose:
            print('m.anchors in pixels ', m.anchors.shape, m.anchors)
            print('m.stride ', m.stride.shape, m.stride)
        # **** anchors define LxAx2 where 2 is for width and height while stride is defined for height and width ****
        # this is not correct if stride does not have equal value for width and height
        # Here we convert anchor definition from pixel unit to feature map space where each stridexstride represent 1 grid cell
        m.anchors/=m.stride.view(-1, 1, 1) # LxAx2 /= Lx1x1 turns into grid cells for each corresponding scale/level
        if verbose: print('m.anchors in grid cells ', m.anchors.shape, m.anchors)
        self.stride=m.stride
        
    def _initialize_aux_biases(self, cf=None, verbose=False):  
        '''
        Initialize bias in detection and auxiliary heads so the baseline assumption of objectness probability and class probability are low,
        reflecting the fact that most grid cells (images) contain backgrounds. Also, this helps improve training stability and guide the model
        with proper bias priors
        Args:
            cf (Tensor): class frequency if known. It will be used to initialize bias for class prediction based on class imbalance,
                probably can get it from 
                `cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.`
        
        '''
        module=self.model[-1]
        for i, (mi, mi2, stride) in enumerate(zip(module.m, module.m2, module.stride)):
            if verbose: print(i, '-'*50)
            # from na*(nc+5)=2A(n+5) to na x (nc+5)=2A x (nc+5) where na is the 2*A for A number of anchors and 2 for height and width
            # nc is the number of classes and 5 for bounding box location and bounding box objectness
            # Note: the second dimension nc+5 actually is [x, y, w, h, objectness, class1, class2, ...., class_nc]
            b=mi.bias.view(module.na, -1) 
            if verbose: print('b ', b.shape, b.min().item(), b.max().item())
            # we need to make sure that at the start of training objectness is low and that class prediction is also low
            # so we initialize biases so that the networks starts with low confidence predictions for objectness and class 
            # (close to background/no object)
            b.data[:,4]+=math.log(8/(64/stride)**2) # start with low objectness score, assuming 8 objects per 640x640 image
            # start with low class confidence prediction. The former assumes small uniform probability and the latter takes into account class imbalance
            b.data[:,5:]+=math.log(0.6/(module.nc-0.99)) if cf is None else torch.log(cf/cf.sum()) 
            mi.bias=torch.nn.Parameter(b.view(-1), requires_grad=True)
            if verbose: print('mi.bias ', mi.bias.shape, mi.bias.min().item(), mi.bias.max().item())
            # na*(nc+5)=2A(n+5) to na x (nc+5)=2A x (nc+5)
            b2=mi2.bias.view(module.na, -1)
            if verbose: print('b2 ', b2.shape, b2.min().item(), b2.max().item())
            b2.data[:,4]+=math.log(8/(640/stride)**2) 
            b2.data[:,5:]+=math.log(0.6/(module.nc-0.99)) if cf is None else torch.log(cf/cf.sum())
            mi2.bias=torch.nn.Parameter(b2.view(-1), requires_grad=True)
            if verbose: print('mi2.bias ', mi2.bias.shape, mi2.bias.min().item(), mi2.bias.max().item())
        
    def forward(self, x, verbose=False):
        '''
        Args:
            x (Tensor): BxCxHxW where C is the number of input channels, e.g., 3 for RGB
        Returns:
            output (list[Tensor]): sequence of BxAxHxWxO where A is the number of anchors, 
                O=Nc+5 where Nc is the number of classes and 5 is bounding box location and objectness core of the bounding box
        '''
        return self.forward_once(x, verbose=verbose)
        
    def forward_once(self, x, verbose=False):
        '''
        Args:
            x (Tensor): BxCxHxW where C is the number of input channels, e.g., 3 for RGB
        Returns:
            output (list[Tensor]): sequence of BxAxHxWxO where A is the number of anchors, 
                O=Nc+5 where Nc is the number of classes and 5 is bounding box location and objectness core of the bounding box
        '''
        y=[] # outputs
        for i, m in enumerate(self.model):
            if verbose: print(i, m.type, m.f, '-'*100)
            if m.f!=-1: # if not from previous layer
                x=y[m.f] if isinstance(m.f, numbers.Number) else [x if j==-1 else y[j] for j in m.f] # from earlier layers
            if verbose: print('x ', x.shape if isinstance(x, torch.Tensor) else [v.shape for v in x] )
            x=m(x) #run
            if verbose: print('x ', x.shape if isinstance(x, torch.Tensor) else [v.shape for v in x])
            y.append(x if m.i in self.save else None) # save output
            if verbose: print('y ', [v.shape if isinstance(v, torch.Tensor) else v for v in y])
        return x

def parse_model(d, ch, verbose=False):
    '''
    d (dict): model configuration
    ch (list[int]): input channel, e.g., ch=[3] for RGB
    '''
    anchors, nc, depth_multiple, width_multiple=d['anchors'], d['nc'], d['depth_multiple'],d['width_multiple']
    n_anchors=(len(anchors[0])//2) if isinstance(anchors, list) else anchors # number of anchors
    n_outputs=n_anchors*(nc+5) # number of outputs = anchors*(num_classes+5)

    layers, save, c2=[],[], ch[-1] # layers, savelist, ch out
    #all_n=[]
    for i, (f, n, m, args) in enumerate(d['backbone']+d['head']): # from, number, module, args
        if verbose:
            print(i, '-'*100)
            print('args ', args)
        m=eval(m) if isinstance(m, str) else m # turn string to module from video_processing.yolov7.models.common
        # make sure that elements in args are int --> though some are None
        for j, a in enumerate(args):
            try: args[j]=eval(a) if isinstance(a,str) else a 
            except: pass
        if verbose: print('args ', args, '\nn ', n) 
        # n is always 1
        #n=max(round(n*depth_multiple), 1) if n>1 else n # depth gain
        #all_n.append(n)
        if m in [Conv, SPPCSPC]:
            ch_in, ch_out=ch[f], args[0]
            if ch_out!=n_outputs: ch_out=make_divisible(ch_out*width_multiple, 8)
            args=[ch_in, ch_out, *args[1:]]
            if verbose: print(f'For Conv, SPPCSPC args: {args} from ch[f] {ch[f]} and f {f}')
            if m in [SPPCSPC]:
                args.insert(2, n) # number of repeats?
                if verbose: print(f'For SPPCSPC n {n}, args {args}')
                n=1
        elif m is Concat:
            if verbose: print(f'For Concat ch {ch} f {f}', end=', ')
            ch_out = sum([ch[x] for x in f])
            if verbose: print(f'ch_out {ch_out}')
        elif m in [IAuxDetect]:
            if verbose: print(f'For IAuxDetect ch {ch} f {f}')
            args.append([ch[x] for x in f])
            if verbose: print(f'args {args}')
        elif m is ReOrg:
            if verbose: print(f'For ReOrg ch {ch} f {f}', end=', ')
            ch_out = ch[f] * 4
            if verbose: print(f' ch_out {ch_out}')
        else:
            if verbose: print(f'For else ch {ch} f {f}', end=', ')
            ch_out=ch[f]
            if verbose: print(f' ch_out {ch_out}')
        if verbose: print(f'Final args {args} n {n}')
        m_=m(*args) # module
        if verbose: print('m_ ', m_)
        t=str(m)
        t=t[t.rfind('.')+1:].strip("'>") #str(m)[8:-2].replace('__main__.','') # module type
        n_params=sum(x.numel() for x in m_.parameters()) # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, n_params # index, from, type, n_params
        if verbose: print(f'i {i} f {f}')
        save.extend(x%i for x in ([f] if isinstance(f, int) else f) if x!=-1) # append to save list
        if verbose: 
            print('t ', t,  ' np ', n_params)
            print('save ', save)
        layers.append(m_)
        if i==0: ch=[]
        ch.append(ch_out)
        if verbose: print('ch ', ch)

    #print('all_n ', np.unique(all_n))
    return nn.Sequential(*layers), sorted(save)
