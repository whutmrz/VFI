from functools import partial
import torch.nn as nn

from our_model import feature_extractor
from our_model import flow_estimation

'''==========Model config=========='''
def init_model_config(F=32, W=7, depth=[2, 2, 2, 4, 4]):
    '''This function should not be modified'''
    return { 
        'embed_dims':[F, 2*F, 4*F, 8*F, 16*F],
        'motion_dims':[0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'num_heads':[8*F//32, 16*F//32],
        'mlp_ratios':[4, 4],
        'qkv_bias':True,
        'norm_layer':partial(nn.LayerNorm, eps=1e-6), 
        'depths':depth,
        'window_sizes':[W, W]
    }, {
        'embed_dims':[F, 2*F, 4*F, 8*F, 16*F],
        'motion_dims':[0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'depths':depth,
        'num_heads':[8*F//32, 16*F//32],
        'window_sizes':[W, W],
        'scales':[4, 8, 16],
        'hidden_dims':[4*F, 4*F],
        'c':F
    }

def init_our_model_config(F=32, W=6, depth=[4, 2, 1]):
    '''This function should not be modified'''
    return {
        'embed_dims':[F, 2*F, 4*F, 8*F],
        'motion_dims':[0, 0, 2*32//depth[0], 8*F//depth[-2], 16*F//depth[-1]],
        'num_heads':8,
        'mlp_ratios':4,
        'qkv_bias':True,
        'norm_layer':partial(nn.LayerNorm, eps=1e-6),
        'depths':depth,
        'window_sizes':W
    }, {
        'embed_dims':[F, 2*F, 4*F, 8*F],
        'motion_dims':[0, 0, 2 * 32 // depth[0], 4 * 32 // depth[1], 8 * 32 // depth[2]],
        'depths':depth,
        'num_heads':[8*F//32, 16*F//32],
        'window_sizes':[W, W],
        'scales':[2, 4, 8],
        'hidden_dims':[4*F, 4*F, 4*F],
        'c':F
    }

MODEL_CONFIG = {
    'LOGNAME': 'EMA-VFI',
    'MODEL_TYPE': (feature_extractor, flow_estimation),
    'MODEL_ARCH': init_model_config(
        F = 32,
        W = 7,
        depth = [2, 2, 2, 4, 4]
    )
}

OUR_MODEL_CONFIG = {
    'LOGNAME': 'MFVFI',
    'MODEL_TYPE': (feature_extractor, flow_estimation),
    'MODEL_ARCH': init_our_model_config(
        F = 32,
        W = 6,
        depth = [4, 2, 1]
    )
}

# MODEL_CONFIG = {
#     'LOGNAME': 'ours_small',
#     'MODEL_TYPE': (feature_extractor, flow_estimation),
#     'MODEL_ARCH': init_model_config(
#         F = 16,
#         W = 7,
#         depth = [2, 2, 2, 2, 2]
#     )
# }