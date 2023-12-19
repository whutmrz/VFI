import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import os
import math
import numpy as np
import cv2
from testdataset import SelfDataset
from torch.utils.data import DataLoader
from padder import InputPadder
from tqdm import tqdm
import model
import torch.nn.functional as F
import torch.nn as nn
from math import log10

def evaluate(model, val_data):
    MSE_LossFn = nn.MSELoss()
    calc_psnr = PeakSignalNoiseRatio().cuda()
    calc_ssim = StructuralSimilarityIndexMeasure().cuda()
    TTA = True
    psnr = 0
    ssim = 0
    with torch.no_grad():
        for i, (validationData, validationFrameIndex) in enumerate(tqdm(val_data)):
            frame0, frame1, frameT = validationData[:, 0:3], validationData[:, 3:6], validationData[:, 6:]
            I0 = frame0.to(device) / 255.
            I1 = frame1.to(device) / 255.
            IFrame = frameT.to(device) / 255.
            padder = InputPadder(I0.shape, divisor=32)
            I0, I1, IFrame = padder.pad(I0, I1, IFrame)
            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]

            fCoeff = model.getFlowCoeff(validationFrameIndex, device)

            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            g_I0_F_t_0 = validationFlowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = validationFlowBackWarp(I1, F_t_1)

            intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0 = F.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1 = 1 - V_t_0

            g_I0_F_t_0_f = validationFlowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = validationFlowBackWarp(I1, F_t_1_f)

            wCoeff = model.getWarpCoeff(validationFrameIndex, device)

            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                        wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
            MSE_val = MSE_LossFn(Ft_p, IFrame)
            psnr += (10 * log10(1 / MSE_val.item()))
            ssim += calc_ssim(Ft_p, IFrame)
        psnr = (psnr / len(val_data))
        ssim = (float(ssim) / len(val_data))
    return psnr, ssim

if __name__ == "__main__":
    device = torch.device("cuda")
    dataset_val = SelfDataset('test', 'E:/WHU/VFI-master/Dataset/experiments', test_num='23')
    val_data = DataLoader(dataset_val, batch_size=1, pin_memory=True, num_workers=1)
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
    psnr, ssim = evaluate(model, val_data, )
    print('psnr={:.2f}, ssim={:.2f}'.format(psnr, ssim))