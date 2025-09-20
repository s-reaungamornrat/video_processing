import torch
import torch.nn as nn

from video_processing.yolov7.utils.general import make_grid

def check_anchor_order(m):
    '''
    Check whether anchors have been ordered in the same order as stride. For example, if strides ordered from small to large,
    the anchors should be the same. Otherwise, correct the order of anchors by fliping the ordering. This is inplace function,
    where it changes data members of the input module m
    Args:
        m (callable): detection module, e.g., yolov7.models.common.IAuxDetect
    '''
    # check anchor order against stride order 
    # nlx1xna//2x1x1x2 -> Lx1xAx1x1x2 where 2 is for width,height  (this exact order)
    # compute anchor areas
    areas=m.anchor_grid.prod(dim=-1).view(-1) # nl*(na//2) -> L*A
    diff_areas=areas[-1]-areas[0]
    diff_strdes=m.stride[-1]-m.stride[0] # Note ***stride is equal for both width and height***
    if diff_strdes.sign()!=diff_areas.sign(): # if not same ordering reorder them
        m.anchor_grid=m.anchor_grid.flip(dims=[0])
        m.anchors=m.anchors.flip(dims=[0])

def autopad(k, p=None): 
    '''
    Compute padding so that the output has the same size as input after operation.
    Note: this does not really make the output having the same size as input =\
    Args:
        k (int or sequence): kernel size
        p (int or sequence): padding
    From https://github.com/WongKinYiu/yolov7/blob/main/models/common.py#L23
    '''
    if p is not None: return p
    return k//2 if isinstance(k, int) else [x//2 for x in k]

class Conv(nn.Module):
    '''
    Example
        conv=Conv(3, 7, 3, 1, None, 1, nn.ReLU)
        x=torch.rand(2, 3, 19,19)
        y=conv(x) # 2x7x19x19

        conv=Conv(4,12,3,1,None,2, True)
        x=torch.rand(2,4,18,19)
        y=conv(x) # 2x12x18x19
    '''
    def __init__(self, ch_in, ch_out, kernel=1, stride=1, padding=None, groups=1, activation=True):
        super(Conv, self).__init__()
        self.conv=nn.Conv2d(ch_in, ch_out, kernel_size=kernel, stride=stride, 
                            padding=autopad(k=kernel, p=padding), groups=groups, bias=False)
        self.bn=nn.BatchNorm2d(ch_out)
        self.activation=nn.SiLU() if (isinstance(activation, bool) and activation) else \
        (activation if isinstance(activation, nn.Module) else nn.Identity())
    def forward(self, x): return self.activation(self.bn(self.conv(x)))
    def fuseforward(self, x): return self.activation(self.conv(x))

class ReOrg(nn.Module):
    '''
    Reorganize tensor from BxCxHxW to Bx4Cx(H/2)x(W/2) by sampling along H and W in even and odd indices
    Example:
        m=ReOrg()
        x=torch.rand(2,3,10,12)
        y=m(x) # 2x12x5x6
    '''
    def __init__(self):
        super(ReOrg, self).__init__()
    def forward(self, x):
        '''
        Args:
            x (Tensor): BxCxHxW
        Returns
            y (Tensor): Bx4Cx(H/2)x(W/2)        
        '''
        x0=x[...,::2,::2]# sample from even positions along H and W, 
        x1=x[...,1::2,::2]# sample from odd along H and even along W
        x2=x[...,::2,1::2]# sample from even along H and odd along W
        x3=x[...,1::2,1::2]# sample from odd along H and W
        return torch.cat([x0,x1,x2,x3], dim=1) # BxCxHxW -> Bx4Cx(H/2)x(W/2)

class ImplicitA(nn.Module):
    '''
    Affine transform to channel dimension of 4D tensors. Only element-wise addition or translation
    Example:
        x=torch.rand(2,9,10,10)
        im=ImplicitA(9)
        y=im(x)
    '''
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel=channel
        self.mean=mean
        self.std=std
        self.implicit=nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)
    def forward(self, x):
        assert x.ndim==self.implicit.ndim, f'dimension of input must be {self.implicit.ndim}'
        return self.implicit+x

class ImplicitM(nn.Module):
    '''
    Affine transform to channel dimension of 4D tensors. Only element-wise multiplication/scaling
    Example:
        x=torch.rand(2,9,10,10)
        im=ImplicitM(9)
        y=im(x)
    '''
    def __init__(self, channel, mean=1., std=0.02):
        super(ImplicitM, self).__init__()
        self.channel=channel
        self.mean=mean
        self.std=std
        self.implicit=nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)
    def forward(self, x):
        return self.implicit*x

class Concat(nn.Module):
    '''
    Example:
        cat=Concat(dim=1)
        x=[torch.rand(2, 1, 7,9) for _ in range(2)]
        y=cat(x) # 2x2x7x9
    '''
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.d=dim
    def forward(self, x):
        return torch.cat(x, dim=self.d)

class SPPCSPC(nn.Module):
    # from CrossStagePartialNetworks
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    '''
    Example:
        module=SPPCSPC(16, 32, e=0.5, k=(5,9,13))
        x=torch.rand(2, 16, 13, 23)
        y=module(x) # 2x32x13x23
    '''
    def __init__(self, ch_in, ch_out, e=0.5, k=(5,9,13)):
        '''
        Args:
            e (float): factor to devide the channels
            k (sequence): max-pooling kernels
        '''
        super(SPPCSPC, self).__init__()
        c_=int(2*ch_out*e) # hidden channels
        self.cv1=Conv(ch_in, c_, kernel=1, stride=1) # will maintain original H W
        self.cv2=Conv(ch_in, c_, kernel=1, stride=1) # will maintain original H W
        self.cv3=Conv(c_, c_, kernel=3, stride=1)  # will maintain original H W
        self.cv4=Conv(c_, c_, kernel=1, stride=1) # will maintain original H W
        # maxpool with also maintain H W since we pad by kernel//2 and kernel is odd
        self.m=nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x//2) for x in k])
        self.cv5=Conv(4*c_, c_, kernel=1, stride=1) # will maintain original H W
        self.cv6=Conv(c_, c_, kernel=3, stride=1) # will maintain original H W
        self.cv7=Conv(2*c_, ch_out, kernel=1, stride=1) # will maintain original H W
    def forward(self, x):
        x1=self.cv4(self.cv3(self.cv1(x)))
        y1=self.cv6(self.cv5(torch.cat([x1]+[m(x1) for m in self.m], dim=1)))
        y2=self.cv2(x)
        return self.cv7(torch.cat((y1,y2), dim=1))
        
class IAuxDetect(nn.Module):
    # class variables shared across all instances
    stride=None # compute during build
    export = False # onnx export
    end2end=False
    include_nms=False
    concat=False
    def __init__(self, nc=80, anchors=(), ch=()):# detection layer
        '''
        Args:
            nc (int): number of object classes
            anchors (list[list[int]]): 3 pairs of anchor width/heights for small, medium, and large bounding boxes per level,
                e.g., [[19,27,  44,40,  38,94], 
                       [96,68,  86,152,  180,137], 
                       [140,301,  303,264,  238,542], 
                       [436,615,  739,380,  925,792]]
            ch (list[int]): list of input channels for each level for its m and m2 modules, e.g., 
                [256, 512, 768, 1024, 320, 640, 960, 1280] where the first 4 are input channels for 
                each level of m and the remainings are the same for m2
        '''
        super(IAuxDetect, self).__init__()
        self.nc=nc # number of classes
        self.no=nc+5 # number of output per anchors
        self.nl=len(anchors) # number of detection layers
        self.na=len(anchors[0])//2 # number of anchors
        self.grid=[torch.zeros(1)]*self.nl # init grid
        a=torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a) # shape nl, na, 2 
        # nl 1 na 1 1 2 
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        print(f'In IAxDetect nl: {self.nl} na: {self.na}')
        print(f'In IAxDetect anchors: {self.anchors.shape} {self.nl}x{self.na}x{2}')
        print(f'In IAxDetect anchor_grid: {self.anchor_grid.shape} {self.nl}x1x{self.na}x1x1x{2}')
        self.m=nn.ModuleList(nn.Conv2d(x, self.no*self.na, 1) for x in ch[:self.nl]) # output conv
        self.m2=nn.ModuleList(nn.Conv2d(x, self.no*self.na, 1) for x in ch[self.nl:]) # output conv

        self.ia=nn.ModuleList(ImplicitA(x) for x in ch[:self.nl])
        self.im=nn.ModuleList(ImplicitM(self.no*self.na) for _ in ch[:self.nl])
    def forward(self, x, verbose=False):
        '''
        Args:
            x (list[Tensor])
        Returns:
            z (Tensor[float]): Bx(AHW)xO output predictions where B is batch size, A is the number of anchors, O is output dimension
                of x,y,w,h,obj,class_1, ..., class_nc with (x,y,w,h) in pixel units where (x,y) is the box center and (w,h) is width,height
            x (list[Tensor]): sequence of outputs from multiresolution main heads, (followed by multiresolution auxillary heads if training), 
                each is BxAxHxWxO and sorted from high-resolution features to low-resolution features (i.e., features with large HW first)
        '''
        ## see https://github.com/WongKinYiu/yolov7/blob/main/models/yolo.py#L116
        z=[]
        self.training|=self.export
        for i in range(self.nl):
            if verbose:
                print(i, '-'*100)
                print('\tx[i] ', x[i].shape,  ' detector.m[i] ', detector.m[i])
            x[i]=self.m[i](self.ia[i](x[i]))
            x[i]=self.im[i](x[i])
            if verbose: print('\tx[i] ', x[i].shape)
            bs,_,ny,nx=x[i].shape
            # BxAxHxWxO where A=number anchors and O is number of classes+5 (where 5 is for bbox coordinate and objectness score) 
            x[i]=x[i].view(bs, self.na, self.no, ny, nx).permute(0,1,3,4,2).contiguous() 
            if verbose:print('\tx[i] ', x[i].shape,'\n\ti+nl ', i+self.nl)
        
            if verbose: print('\tx[i+self.nl] ', x[i+self.nl].shape,  ' self.m2[i] ', self.m2[i])
            x[i+self.nl]=self.m2[i](x[i+self.nl])
            if verbose: print('\tx[i+self.nl] ', x[i+self.nl].shape)
            # BxAxHxWxO where A=number anchors and O is number of classes+5 (where 5 is for bbox coordinate and objectness score) 
            x[i+self.nl]=x[i+self.nl].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if verbose: print('\tx[i+self.nl] ', x[i+self.nl].shape)
            if not self.training: # inference
                if verbose:
                    print('\tself.grid[i].shape ', self.grid[i].shape)
                    print('\tself.grid[i].shape[2:4] ', self.grid[i].shape[2:4], ' x[i].shape[2:4] ', x[i].shape[2:4])
                if self.grid[i].shape[2:4]!=x[i].shape[2:4]:
                    # feature map grid
                    self.grid[i]=make_grid(nx=nx, ny=ny).to(x[i].device) # 1x1xHxWx2
                if verbose: print('\tself.grid[i].shape ', self.grid[i].shape )
                y=x[i].sigmoid()
                if verbose: 
                    print('\tx[i] ', x[i].min().item(), x[i].max().item())
                    print('\ty ', y.shape, y.min().item(), y.max().item() )
                    print('\tstride ', self.stride[i])
                if not torch.onnx.is_in_onnx_export():
                    # xy coordinates of bounding boxes
                    #               BxAxHxWx2        1x1xHxWx2  
                    y[...,0:2]=(2.*y[...,0:2] -0.5 + self.grid[i])*self.stride[i]
                    # width and height of bounding boxes
                    #              BxAxHxWx2             1xAx1x1x2
                    y[...,2:4]=( (2.*y[...,2:4])**2. ) * self.anchor_grid[i]
                else:
                    # split the 4th dim of BxAxHxWxO into BxAxHxWx2 of xy, BxAxHxWx2 of width/height, 
                    # and BxAxHxWx(Nc+1) of objectness and Nc for number of classes
                    xy,wh,confidence=y.split((2,2,self.nc+1), dim=4)
                    if verbose: print('xy ', xy.shape, ' wh ', wh.shape, ' confidence ', confidence.shape)
                    xy=xy*(2.* self.stride[i])+(self.stride[i]*(self.grid[i]-0.5))
                    wh=wh**2. * (4.*self.anchor_grid[i].data)
                    y=torch.cat([xy,wh,confidence], dim=4)
                    
                z.append(y.view(bs, -1, self.no)) # BxAxHxWxO -> Bx(AHW)xO
                
        return x if self.training else (torch.cat(z, 1), x[:self.nl])
        