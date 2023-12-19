import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import os
import math
import numpy as np
import cv2
from testdataset import SelfDataset
from torch.utils.data import DataLoader
from models import interpolator as film_net_interpolator
from models.interpolator import FILM_interpolator
from models import options as film_net_options
from utils import *
from tqdm import tqdm

def evaluate(model, val_data):
    calc_psnr = PeakSignalNoiseRatio().cuda()
    calc_ssim = StructuralSimilarityIndexMeasure().cuda()
    TTA = True
    psnr = []
    ssim = 0
    for i, imgs in enumerate(tqdm(val_data)):
        imgs['x0'] = imgs['x0'].to(device, non_blocking=True) / 255.
        imgs['x1'] = imgs['x1'].to(device, non_blocking=True) / 255.
        imgs['y'] = imgs['y'].to(device, non_blocking=True) / 255.
        imgs['time'] = imgs['time'].to(device, non_blocking=True)
        padder = InputPadder(imgs['x0'].shape, divisor=32)
        imgs['x0'] = padder.pad(imgs['x0'])
        imgs['x1'] = padder.pad(imgs['x1'])
        imgs['y'] = padder.pad(imgs['y'])
        gt = imgs['y']
        with torch.no_grad():
            pred = model(imgs)
            mid = padder.unpad(pred['image'])
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
    def create_model():
        options = film_net_options.Options()
        return film_net_interpolator.create_model(options)
    path = 'checkpoint_dir/test/checkpoint_3.pt'
    options = film_net_options.Options()
    model = FILM_interpolator(options).cuda()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    psnr, ssim = evaluate(model, val_data)
    print('psnr={:.2f}, ssim={:.2f}'.format(psnr, ssim))