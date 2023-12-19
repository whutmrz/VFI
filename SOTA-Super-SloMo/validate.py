import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import model
import dataloader
from math import log10
import datetime
import cv2

from tensorboardX import SummaryWriter
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import click


# For parsing commandline arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset_root", type=str, default='E:/WHU/VFI-master/Dataset', required=True, help='path to dataset folder containing train-test-validation folders')
# parser.add_argument("--checkpoint_dir", type=str, default='E:/WHU/VFI-master/SOTA-Super-Slomo/ckp', required=True, help='path to folder for saving checkpoints')
# parser.add_argument("--checkpoint", type=str, help='path of checkpoint for pretrained model')
# parser.add_argument("--train_continue", type=bool, default=False, help='If resuming from checkpoint, set to True and set `checkpoint` path. Default: False.')
# parser.add_argument("--epochs", type=int, default=200, help='number of epochs to train. Default: 200.')
# parser.add_argument("--train_batch_size", type=int, default=6, help='batch size for training. Default: 6.')
# parser.add_argument("--validation_batch_size", type=int, default=10, help='batch size for validation. Default: 10.')
# parser.add_argument("--init_learning_rate", type=float, default=0.0001, help='set initial learning rate. Default: 0.0001.')
# parser.add_argument("--milestones", type=list, default=[100, 150], help='Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [100, 150]')
# parser.add_argument("--progress_iter", type=int, default=100, help='frequency of reporting progress and validation. N: after every N iterations. Default: 100.')
# parser.add_argument("--checkpoint_epoch", type=int, default=5, help='checkpoint saving frequency. N: after every N epochs. Each checkpoint is roughly of size 151 MB.Default: 5.')
# args = parser.parse_args()

writer = SummaryWriter('log')

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans_forward = transforms.ToTensor()
trans_backward = transforms.ToPILImage()
mean = [0.429, 0.431, 0.397]
mea0 = [-m for m in mean]
std = [1] * 3
if device != "cpu":
    trans_forward = transforms.Compose([trans_forward, transforms.Normalize(mean=mean, std=std)])
    trans_backward = transforms.Compose([transforms.Normalize(mean=mea0, std=std), trans_backward])

normalize = transforms.Normalize(mean=mean,
                                 std=std)
transform = transforms.Compose([transforms.ToTensor(), normalize])

flow = model.UNet(6, 4).to(device)
interp = model.UNet(20, 5).to(device)
back_warp = None

def load_models(checkpoint):
    states = torch.load(checkpoint, map_location='cpu')
    interp.load_state_dict(states['state_dictAT'])
    flow.load_state_dict(states['state_dictFC'])

validationset = dataloader.SuperSloMo(root='E:/WHU/VFI-master/Dataset/GTs', transform=transform, train=False)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False)
print(validationset)

###Create transform to display image from tensor

negmean = [x * -1 for x in mean]
revNormalize = transforms.Normalize(mean=negmean, std=std)
TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

L1_lossFn = nn.L1Loss()
MSE_LossFn = nn.MSELoss()
vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
vgg16_conv_4_3.to(device)
for param in vgg16_conv_4_3.parameters():
    param.requires_grad = False
calc_psnr = PeakSignalNoiseRatio().cuda()
calc_ssim = StructuralSimilarityIndexMeasure().cuda()

def validate():
    # For details see training.
    psnr = 0
    ssim = 0
    flag = 1
    load_models(checkpoint='E:/WHU/VFI-master/SOTA-Super-SloMo/data/Superslomo.ckpt')
    with torch.no_grad():
        for validationIndex, (validationData, validationFrameIndex) in enumerate(validationloader, 0):
            frame0, frameT, frame1 = validationData

            w0, h0 = frame0.size(3), frame0.size(2)
            w, h = (w0//32)*32, (h0//32)*32
            back_warp = model.backWarp(w, h, device).to(device)
            I0 = frame0.to(device)
            I1 = frame1.to(device)
            IFrame = frameT.to(device)

            flowOut = flow(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]

            fCoeff = model.getFlowCoeff(validationFrameIndex, device)

            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            g_I0_F_t_0 = back_warp(I0, F_t_0)
            g_I1_F_t_1 = back_warp(I1, F_t_1)

            intrpOut = interp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0 = F.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1 = 1 - V_t_0

            g_I0_F_t_0_f = back_warp(I0, F_t_0_f)
            g_I1_F_t_1_f = back_warp(I1, F_t_1_f)

            wCoeff = model.getWarpCoeff(validationFrameIndex, device)

            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                        wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
            # For tensorboard
            if (flag):
                retImg = torchvision.utils.make_grid(
                    [revNormalize(frame0[0]), revNormalize(frameT[0]), revNormalize(Ft_p.cpu()[0]),
                     revNormalize(frame1[0])], padding=10)
                flag = 0

            # psnr
            psnr += float(calc_psnr(Ft_p, IFrame))
            ssim += float(calc_ssim(Ft_p, IFrame))

    return (psnr / len(validationloader)), (ssim / len(validationloader))

# @click.command('Evaluate Model by converting a low-FPS video to high-fps')
# @click.argument('input', default='E:/WHU/VFI-master/SOTA-Super-SloMo/data/input.mp4')
# @click.option('--checkpoint', default='E:/WHU/VFI-master/SOTA-Super-SloMo/data/Superslomo.ckpt', help='Path to model checkpoint')
# @click.option('--output', default='E:/WHU/VFI-master/SOTA-Super-SloMo/data/output.mp4', help='Path to output file to save')
# @click.option('--batch', default=2, help='Number of frames to process in single forward pass')
# @click.option('--scale', default=4, help='Scale Factor of FPS')
# @click.option('--fps', default=120, help='FPS of output video')
if __name__ == '__main__':
    psnr, ssim = validate()
    print(psnr)
    print(ssim)
