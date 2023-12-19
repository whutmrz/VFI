import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import os
import math
import numpy as np
import cv2
from testdataset import SelfDataset
from torch.utils.data import DataLoader
from util import InputPadder
from tqdm import tqdm
import models
from warp import warp
import torch.nn.functional as F

def evaluate(model, val_data):
    calc_psnr = PeakSignalNoiseRatio().cuda()
    calc_ssim = StructuralSimilarityIndexMeasure().cuda()
    TTA = True
    psnr = []
    ssim = 0
    for i, imgs in enumerate(tqdm(val_data)):
        imgs = imgs.to(device, non_blocking=True) / 255.
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]
        padder = InputPadder(imgs.shape, divisor=64)
        imgs, gt = padder.pad(imgs, gt)
        img1, img2 = imgs[:, 0:3], imgs[:, 3:]
        h, w = img1.size()[-2:]
        with torch.no_grad():
            pred = model(imgs)
            output = F.interpolate(pred, (h, w))
            mid = warp(img1, output)
            mid = padder.unpad(mid)
            gt = padder.unpad(gt)
            # psnr += calc_psnr(pred, gt)
            ssim += calc_ssim(mid, gt)
        for j in range(gt.shape[0]):
            psnr.append(-10 * math.log10(((gt[j] - mid[j]) * (gt[j] - mid[j])).mean().cpu().item()))
        # if i == len(val_data) - 1:
        #     # for j in range(gt.shape[0]):
        #     pred[-1] = pred[-1] * 255.
        #     img = (pred[-1].squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
        #     cv2.imwrite('example/test_1/' + str(nr_eval) + '_' + '.png', img)
    psnr = float(np.array(psnr).mean())
    ssim = (float(ssim) / len(val_data))
    return psnr, ssim

if __name__ == "__main__":
    device = torch.device("cuda")
    dataset_val = SelfDataset('test', 'E:/WHU/VFI-master/Dataset/experiments', test_num='23')
    val_data = DataLoader(dataset_val, batch_size=1, pin_memory=True, num_workers=1)
    model = models.__dict__['flownets'](None).to(device)
    model.load_state_dict(torch.load(f'ckpt/flownet.pkl'))
    model.eval()
    model.to(torch.device("cuda"))
    psnr, ssim = evaluate(model, val_data)
    print('psnr={:.2f}, ssim={:.2f}'.format(psnr, ssim))