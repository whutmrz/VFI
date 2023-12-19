import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave

'''==========import from our code=========='''
sys.path.append('.')
from util import InputPadder
import models
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch.nn.functional as F
from warp import warp

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MFVFI', type=str)
args = parser.parse_args()
assert args.model in ['ours', 'ours_small', 'MFVFI'], 'Model not exists!'

'''==========Model setting=========='''
TTA = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.__dict__['flownets'](None).to(device)
model.load_state_dict(torch.load(f'ckpt/flownet.pkl'))
model.eval()
model.to(torch.device("cuda"))
calc_psnr = PeakSignalNoiseRatio().cuda()
calc_ssim = StructuralSimilarityIndexMeasure().cuda()
print(f'=========================Start Generating=========================')

# I0 = cv2.imread('example/im1.png')
# I2 = cv2.imread('example/im3.png')
I0 = cv2.imread('E:/WHU/VFI-master/Dataset/LPBFdataset/00025/0030/im1.png')
I2 = cv2.imread('E:/WHU/VFI-master/Dataset/LPBFdataset/00025/0030/im3.png')
GT = cv2.imread('E:/WHU/VFI-master/Dataset/LPBFdataset/00025/0030/im2.png')

I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
GT_ = (torch.tensor(GT.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

padder = InputPadder(I0_.shape, divisor=32)
I0_, I2_ = padder.pad(I0_, I2_)
h, w = I0_.size()[-2:]
input = torch.cat((I0_, I2_), dim=1)
output = model(input)
output = F.interpolate(output, (h, w))
mid = warp(input[:, :3], output)
mid = padder.unpad(mid)
# psnr = []
# for j in range(pred.shape[0]):
#     psnr.append(-10 * math.log10(((GT_[j] - pred[j]) * (GT_[j] - pred[j])).mean().cpu().item()))
# psnr = float(np.array(psnr).mean())
psnr = calc_psnr(mid, GT_)
ssim = calc_ssim(mid, GT_)
mid = mid.squeeze(0)
mid = (mid.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
images = [I0[:, :, ::-1], mid[:, :, ::-1], I2[:, :, ::-1]]
cv2.imwrite('E:/WHU/VFI-master/test/350W-1400/flownet.png', mid)
# mimsave('example/out_2x.gif', images, fps=3)
# mimsave('E:/WHU/VFI-master/test/350W-1200/flownet.gif', images, fps=3)
print('psnr={:.2f}, ssim={:.2f}'.format(psnr, ssim))
print(f'=========================Done=========================')