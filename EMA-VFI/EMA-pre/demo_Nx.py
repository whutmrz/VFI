import cv2
import math
import os
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MFVFI', type=str)
parser.add_argument('--n', default=9, type=int)
parser.add_argument('--begin', default=21, type=int)
args = parser.parse_args()
assert args.model in ['ours_t', 'ours_small_t', 'MFVFI'], 'Model not exists!'

'''==========Model setting=========='''
TTA = True
if args.model == 'ours_small_t':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small_t'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
if args.model == 'ours_t':
    cfg.MODEL_CONFIG['LOGNAME'] = 'EMA-VFI'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )
else:
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'MFVFI'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_our_model_config(
    )
model = Model(-1)
model.load_model()
model.eval()
model.device()
calc_psnr = PeakSignalNoiseRatio().cuda()
calc_ssim = StructuralSimilarityIndexMeasure().cuda()

print(f'=========================Start Generating=========================')
path = 'E:/WHU/VFI-master/Dataset/GTs/350W-1400'
n = args.n
begin = args.begin
I0 = cv2.imread(os.path.join(path, f'00{begin}.png'))
I2 = cv2.imread(os.path.join(path, f'00{begin+n+1}.png'))
I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
images = []
for i in range(n):
    img = cv2.imread(os.path.join(path, f'00{begin+i+1}.png'))
    img = (torch.tensor(img.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    images.append(img)
# I0 = cv2.imread('E:/WHU/VFI-master/Dataset/GTs/350W-1400/0021.png')
# I2 = cv2.imread('E:/WHU/VFI-master/Dataset/GTs/350W-1400/0030.png')
# mid1 = cv2.imread('E:/WHU/VFI-master/Dataset/GTs/350W-1400/0022.png')
# mid2 = cv2.imread('E:/WHU/VFI-master/Dataset/GTs/350W-1400/0023.png')
# mid3 = cv2.imread('E:/WHU/VFI-master/Dataset/GTs/350W-1400/0024.png')
# mid4 = cv2.imread('E:/WHU/VFI-master/Dataset/GTs/350W-1400/0025.png')
# mid5 = cv2.imread('E:/WHU/VFI-master/Dataset/GTs/350W-1400/0026.png')
# mid6 = cv2.imread('E:/WHU/VFI-master/Dataset/GTs/350W-1400/0027.png')
# mid7 = cv2.imread('E:/WHU/VFI-master/Dataset/GTs/350W-1400/0028.png')
# mid8 = cv2.imread('E:/WHU/VFI-master/Dataset/GTs/350W-1400/0029.png')

# I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
# I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
# mid1_ = (torch.tensor(mid1.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
# mid2_ = (torch.tensor(mid2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
# mid3_ = (torch.tensor(mid3.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
# mid4_ = (torch.tensor(mid4.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

padder = InputPadder(I0_.shape, divisor=32)
I0_, I2_ = padder.pad(I0_, I2_)

video = [I0[:, :, ::-1]]
def make_inference(I0, I1, n):
    global model
    with torch.no_grad():
        middle = model.our_inference(I0, I1, TTA=TTA, fast_TTA=TTA)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2)
        second_half = make_inference(middle, I1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]
# preds = model.multi_inference(I0_, I2_, TTA=TTA, time_list=[(i+1)*(1./args.n) for i in range(args.n - 1)], fast_TTA=TTA)
preds = make_inference(I0_, I2_, n)

for i, pred in enumerate(preds):
    pred = pred.squeeze(0)
    mid = (padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)[:, :, ::-1]
    video.append(mid)
    cv2.imwrite('E:/WHU/VFI-master/test/multi/ours/9_frame/{}.png'.format(i+1), mid)
psnrs = []
ssims = []
psnr_sum = 0
ssim_sum = 0
for i in range(n):
    psnr = calc_psnr(padder.unpad(preds[i]), images[i])
    ssim = calc_ssim(padder.unpad(preds[i]), images[i])
    psnr_sum += psnr
    ssim_sum += ssim
    print('frame_num={}, psnr={:.2f}, ssim3={:.2f}'.format(i+1, psnr, ssim))
print('psnr_avg={:.2f}, ssim_avg={:.2f}'.format(psnr_sum/n, ssim_sum/n))
# psnr1 = calc_psnr(padder.unpad(preds[0]), mid1_)
# psnr2 = calc_psnr(padder.unpad(preds[1]), mid2_)
# psnr3 = calc_psnr(padder.unpad(preds[2]), mid3_)
# psnr4 = calc_psnr(padder.unpad(preds[3]), mid4_)
# ssim1 = calc_ssim(padder.unpad(preds[0]), mid1_)
# ssim2 = calc_ssim(padder.unpad(preds[1]), mid2_)
# ssim3 = calc_ssim(padder.unpad(preds[2]), mid3_)
# ssim4 = calc_ssim(padder.unpad(preds[3]), mid4_)
video.append(I2[:, :, ::-1])
# mimsave('E:/WHU/VFI-master/test/multi/EMA-VFI/extract_6/out_Nx.gif', video, fps=args.n)
# print('psnr1={:.2f}, ssim1={:.2f}'.format(psnr1, ssim1))
# print('psnr2={:.2f}, ssim2={:.2f}'.format(psnr2, ssim2))
# print('psnr3={:.2f}, ssim3={:.2f}'.format(psnr3, ssim3))
# print('psnr4={:.2f}, ssim4={:.2f}'.format(psnr4, ssim4))
print(f'=========================Done=========================')