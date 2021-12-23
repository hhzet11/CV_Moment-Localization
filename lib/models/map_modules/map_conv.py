from torch import nn
import torch.nn.functional as F
from models.map_modules import get_padded_mask_and_weight

class MapConv(nn.Module):

    def __init__(self, cfg):
        super(MapConv, self).__init__()
        print("-----------------HELLO-----------------")
        input_size = cfg.INPUT_SIZE
        hidden_sizes = cfg.HIDDEN_SIZES
        kernel_sizes = cfg.KERNEL_SIZES
        strides = cfg.STRIDES
        paddings = cfg.PADDINGS
        dilations = cfg.DILATIONS
        self.convs = nn.ModuleList()
        assert len(hidden_sizes) == len(kernel_sizes) \
               and len(hidden_sizes) == len(strides) \
               and len(hidden_sizes) == len(paddings) \
               and len(hidden_sizes) == len(dilations)
        channel_sizes = [input_size]+hidden_sizes
        for i, (k, s, p, d) in enumerate(zip(kernel_sizes, strides, paddings, dilations)):
            self.convs.append(nn.Conv2d(channel_sizes[i], channel_sizes[i+1], k, s, p, d))

    def forward(self, x, mask):
        padded_mask = mask
        for i, pred in enumerate(self.convs):
            '''
            if i%2==0:
                x_identity=x

            if i%2==1:
                x=x+x_identity
            '''

            if i%2==0:
                x_identity=x
                if i==0: 
                    x1_identity=x
                elif i==2: 
                    x2_identity=x
                elif i==4: 
                    x3_identity=x                
                elif i==6: 
                    x4_identity=x            
            if i==1:
                x=x+x1_identity
            elif i==3:
                x=x+x2_identity
                x=x+x1_identity
            elif i==5:
                x=x+x3_identity
                x=x+x2_identity
                x=x+x1_identity
            elif i==7:
                x=x+x4_identity
                x=x+x3_identity
                x=x+x2_identity
                x=x+x1_identity
            
            

            x = F.relu(pred(x))
            
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, pred)
            x = x * masked_weight
        return x