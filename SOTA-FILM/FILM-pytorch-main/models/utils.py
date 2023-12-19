
import torch
import torch.nn as nn
import torch.nn.functional as F

from .options import Options

class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """
    def __init__(self, dims, divisor = 16):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def build_image_pyramid(image, options):

    levels = options.pyramid_levels
    pyramid = []
    pool = nn.AvgPool2d(kernel_size=2, stride=2)
    for i in range(0, levels):
        pyramid.append(image)
        if i < levels-1:
            image = pool(image)
    return pyramid

def warp(image, flow):
    warped = F.grid_sample(image, torch.permute(flow, (0, 2, 3, 1)), align_corners=False)
    return warped

def multiply_pyramid(pyramid, scalar):
    return [ torch.permute(torch.permute(image, (1, 2, 3, 0)) * scalar, [3, 0, 1, 2]) for image in pyramid]

def flow_pyramid_synthesis(residual_pyramid):
    flow = residual_pyramid[-1]
    flow_pyramid = [flow]
    for residual_flow in reversed(residual_pyramid[:-1]):
        level_size = residual_flow.shape[1:3]
        flow = F.interpolate(flow, scale_factor=2)
        flow = residual_flow + flow
        flow_pyramid.append(flow)
    return list(reversed(flow_pyramid))

def pyramid_warp(feature_pyramid, flow_pyramid):
    warped_feature_pyramid = []
    for features, flow in zip(feature_pyramid, flow_pyramid):
        warped_feature_pyramid.append(warp(features, flow))
    return warped_feature_pyramid

def concatenate_pyramids(pyramid1, pyramid2):
    result = []
    for features1, features2 in zip(pyramid1, pyramid2):
        result.append(torch.concat([features1, features2], axis=1))
    return result
