import copy
import math
import torch

class ModelEMA:
    '''
    Model exponential moving average, keeps a moving average of model state_dict (parameters and buffers)
    This class is senstitive to where it is initialized (in the sequence of model initialization, GPU assignment and 
    distributed training wrapper)
    '''
    def __init__(self, model, decay=0.9999, updates=0):
        self.ema=copy.deepcopy(model).eval()
        self.updates=updates # tracking the number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay=lambda x: decay*(1.-math.exp(-x/2000.))
        for p in self.ema.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates+=1
        decay=self.decay(self.updates)
        model_state_dict=model.state_dict()
        #print('update ema')
        for k, v in self.ema.state_dict().items():
            if not v.dtype.is_floating_point: continue
            #print('({:.3f},{:.3f})'.format(v.min().item(), v.max().item()), end=',')
            v=decay*v+(1.-decay)*model_state_dict[k].detach()
            #print('({:.3f},{:.3f})'.format(v.min().item(), v.max().item()))
    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        '''
        Copy attribute from model to ema
        Args:
            include (sequence[str]): attributes to be included
            exclude (sequence[str]): attributes to be excluded
        '''
        # update EMA attribute
        for k, v in model.__dict__.items():
            if (len(include)>0 and k not in include) or k.startswith('_') or k in exclude: continue
            setattr(self.ema, k, v)