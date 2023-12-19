import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave
import model
'''==========import from our code=========='''
sys.path.append('.')
from padder import InputPadder
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch.nn.functional as F

parser = argparse.ArgumentParser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
flowComp = model.UNet(6, 4)
flowComp.to(device)
ArbTimeFlowIntrp = model.UNet(20, 5)
ArbTimeFlowIntrp.to(device)
validationFlowBackWarp = model.backWarp(320, 224, device)
validationFlowBackWarp = validationFlowBackWarp.to(device)
dict1 = torch.load('ckp/SuperSloMo0.ckpt')
ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
flowComp.load_state_dict(dict1['state_dictFC'])

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
I0_, I2_= padder.pad(I0_, I2_)
frameindex = torch.tensor([0])
flowOut = flowComp(torch.cat((I0_, I2_), dim=1))
F_0_1 = flowOut[:, :2, :, :]
F_1_0 = flowOut[:, 2:, :, :]

fCoeff = model.getFlowCoeff(frameindex, device)

F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

g_I0_F_t_0 = validationFlowBackWarp(I0_, F_t_0)
g_I1_F_t_1 = validationFlowBackWarp(I2_, F_t_1)

intrpOut = ArbTimeFlowIntrp(torch.cat((I0_, I2_, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
V_t_0 = F.sigmoid(intrpOut[:, 4:5, :, :])
V_t_1 = 1 - V_t_0

g_I0_F_t_0_f = validationFlowBackWarp(I0_, F_t_0_f)
g_I1_F_t_1_f = validationFlowBackWarp(I2_, F_t_1_f)

wCoeff = model.getWarpCoeff(frameindex, device)

Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
# inputs = to_gpu(inputs)

mid = padder.unpad(Ft_p)
psnr = calc_psnr(mid, GT_)
ssim = calc_ssim(mid, GT_)
mid = mid.squeeze(0)
mid = (mid.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
images = [I0[:, :, ::-1], mid[:, :, ::-1], I2[:, :, ::-1]]
cv2.imwrite('E:/WHU/VFI-master/test/350W-1400/superslomo.png', mid)
# mimsave('example/out_2x.gif', images, fps=3)
# mimsave('E:/WHU/VFI-master/Dataset/LPBFdataset/00025/0029/out_2x.gif', images, fps=3)
print('psnr={:.2f}, ssim={:.2f}'.format(psnr, ssim))
print(f'=========================Done=========================')