import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave

'''==========import from our code=========='''
sys.path.append('.')
from model.RIFE import Model
from padder import InputPadder
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours_t', type=str)
parser.add_argument('--n', default=3, type=int)
args = parser.parse_args()
assert args.model in ['ours_t', 'ours_small_t', 'MFVFI'], 'Model not exists!'

'''==========Model setting=========='''
TTA = True
path = 'ckpt'
model = Model(-1)
model.load_model(path)
model.eval()
model.device()
calc_psnr = PeakSignalNoiseRatio().cuda()
calc_ssim = StructuralSimilarityIndexMeasure().cuda()

print(f'=========================Start Generating=========================')

I0 = cv2.imread('E:/WHU/VFI-master/Dataset/GTs/350W-1400/0001.png')
I2 = cv2.imread('E:/WHU/VFI-master/Dataset/GTs/350W-1400/0004.png')

I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

padder = InputPadder(I0_.shape, divisor=32)
I0_, I2_ = padder.pad(I0_, I2_)

images = [I0[:, :, ::-1]]
preds = model.multi_inference(I0_, I2_, TTA=TTA, time_list=[(i+1)*(1./args.n) for i in range(args.n - 1)], fast_TTA=TTA)
for i, pred in enumerate(preds):
    pred = (padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)[:, :, ::-1]
    images.append(pred)
    cv2.imwrite('E:/WHU/VFI-master/test/multi/{}.png'.format(i+1), pred)
images.append(I2[:, :, ::-1])
mimsave('example/out_Nx.gif', images, fps=args.n)

print(f'=========================Done=========================')