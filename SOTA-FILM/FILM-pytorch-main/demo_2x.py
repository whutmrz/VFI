import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave

'''==========import from our code=========='''
sys.path.append('.')
from models import interpolator as film_net_interpolator
from models.interpolator import FILM_interpolator
from models import options as film_net_options
from utils import *
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def create_model():
    options = film_net_options.Options()
    return film_net_interpolator.create_model(options)

path = 'checkpoint_dir/test/checkpoint_3.pt'
parser = argparse.ArgumentParser()
options = film_net_options.Options()
model = FILM_interpolator(options).cuda()
checkpoint = torch.load(path)
# for key in list(checkpoint['state_dict'].keys()):
#     checkpoint['state_dict'][key.replace('module.','')] = checkpoint['state_dict'].pop(key)
model.load_state_dict(checkpoint['state_dict'])
# path = 'ckpt/checkpoint_40.pkl'
# parser = argparse.ArgumentParser()
# options = film_net_options.Options()
# model = FILM_interpolator(options).cuda()
# checkpoint = torch.load(path)
# for key in list(checkpoint.keys()):
#     checkpoint[key.replace('module.','')] = checkpoint.pop(key)
# model.load_state_dict(checkpoint)
model.eval()

print(f'=========================Start Generating=========================')
calc_psnr = PeakSignalNoiseRatio().cuda()
calc_ssim = StructuralSimilarityIndexMeasure().cuda()
# I0 = cv2.imread('example/im1.png')
# I2 = cv2.imread('example/im3.png')
I0 = cv2.imread('E:/WHU/VFI-master/Dataset/LPBFdataset/00025/0030/im1.png')
I2 = cv2.imread('E:/WHU/VFI-master/Dataset/LPBFdataset/00025/0030/im3.png')
GT = cv2.imread('E:/WHU/VFI-master/Dataset/LPBFdataset/00025/0030/im2.png')

I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
GT_ = (torch.tensor(GT.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
time = torch.tensor([[0.5, 0.5]]).cuda()
padder = InputPadder(I0_.shape, divisor=32)
I0_= padder.pad(I0_)
I2_= padder.pad(I2_)
inputs = {'x0': I0_, 'x1': I2_, 'y': GT_,
                   'time': time}
# inputs = to_gpu(inputs)
with torch.no_grad():
    pred = model(inputs)
mid = padder.unpad(pred['image'])
psnr = calc_psnr(mid, GT_)
ssim = calc_ssim(mid, GT_)
mid = mid.squeeze(0)
mid = (mid.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
images = [I0[:, :, ::-1], mid[:, :, ::-1], I2[:, :, ::-1]]
cv2.imwrite('E:/WHU/VFI-master/test/350W-1400/film.png', mid)
# mimsave('example/out_2x.gif', images, fps=3)
# mimsave('E:/WHU/VFI-master/Dataset/LPBFdataset/00025/0029/out_2x.gif', images, fps=3)
print('psnr={:.2f}, ssim={:.2f}'.format(psnr, ssim))
print(f'=========================Done=========================')