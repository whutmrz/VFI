import os
import cv2
import math
import time
import torch
# import torch.distributed as dist
import numpy as np
import random
import argparse
from tqdm import tqdm

from lpbfdataset import SelfDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from padder import InputPadder
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from run import Network
import torch.nn.functional as F
from warp import warp
from torch.optim import AdamW
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# from torch.utils.data.distributed import DistributedSampler
# from config import *

device = torch.device("cuda")
exp = os.path.abspath('.').split('/')[-1]


def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000
        return 2e-4 * mul
    else:
        mul = np.cos((step - 2000) / (300 * args.step_per_epoch - 2000) * math.pi) * 0.5 + 0.5
        return (2e-4 - 2e-5) * mul + 2e-5


def train(model, local_rank, batch_size, data_path):
    step = 0
    nr_eval = 0
    best_psnr = 0
    best_ssim = 0
    best_psnr_epoch = 0
    best_ssim_epoch = 0
    loss_fn = torch.nn.MSELoss()
    dataset = SelfDataset('train', data_path)
    # sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=True, drop_last=True)
    args.step_per_epoch = train_data.__len__()
    dataset_val = SelfDataset('test', data_path)
    val_data = DataLoader(dataset_val, batch_size=batch_size, pin_memory=True, num_workers=1)
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    print('training...')
    model.train()
    time_stamp = time.time()
    for epoch in range(50):
        # sampler.set_epoch(epoch)
        for i, imgs in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            imgs = imgs.to(device, non_blocking=True) / 255.
            img1, img2, gt = imgs[:, 0:3], imgs[:, 3:6], imgs[:, 6:]
            learning_rate = get_learning_rate(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            pred = model(img1, img2, [0.5])
            # h, w = gt.size()[-2:]
            # flow = F.interpolate(flow, (h, w))
            # b = imgs.size(0) // 2
            # warped_img1 = warp(img1[:b], flow[:b])
            # warped_img2 = warp(img1[b:], flow[b:])
            # pred = torch.cat((warped_img1, warped_img2), dim=0)
            loss = loss_fn(pred[0], gt)
            loss.requires_grad_(True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss:{:.4e}'.format(epoch + 1, i, args.step_per_epoch,
                                                                             data_time_interval, train_time_interval,
                                                                             loss))
            step += 1
        nr_eval += 1
        if nr_eval % 1 == 0:
            best, best_epoch, best_ssim, best_ssim_epoch = evaluate(model, val_data, nr_eval, local_rank, best_psnr,
                                                                    best_psnr_epoch, best_ssim, best_ssim_epoch)
        # model.save_model(local_rank)

        # dist.barrier()


def evaluate(model, val_data, nr_eval, local_rank, best_psnr, best_psnr_epoch, best_ssim, best_ssim_epoch):
    # if local_rank == 0:
    #     writer_val = SummaryWriter('log/validate_EMAVFI')
    calc_psnr = PeakSignalNoiseRatio().cuda()
    calc_ssim = StructuralSimilarityIndexMeasure().cuda()
    psnr = []
    ssim = 0
    model.eval()
    for i, imgs in enumerate(tqdm(val_data)):
        imgs = imgs.to(device, non_blocking=True) / 255.
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]
        padder = InputPadder(imgs.shape, divisor=32)
        imgs, gt = padder.pad(imgs, gt)
        img1, img2 = imgs[:, 0:3], imgs[:, 3:]
        with torch.no_grad():
            pred = model(img1, img2, [0.5])
            # h, w = gt.size()[-2:]
            # flow = F.interpolate(flow, (h, w))
            # b = imgs.size(0) // 2
            # warped_img1 = warp(img1[:b], flow[:b])
            # warped_img2 = warp(img1[b:], flow[b:])
            # pred = torch.cat((warped_img1, warped_img2), dim=0)
            pred = padder.unpad(pred[0])
            gt = padder.unpad(gt)
            # psnr += calc_psnr(pred, gt)
            ssim += calc_ssim(pred, gt)
        for j in range(gt.shape[0]):
            psnr.append(-10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item()))
        if i == len(val_data) - 1:
            # for j in range(gt.shape[0]):
            pred[-1] = pred[-1] * 255.
            img = (pred[-1].squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
            cv2.imwrite('example/test_1/' + str(nr_eval) + '_' + '.png', img)
    psnr = float(np.array(psnr).mean())
    ssim = (float(ssim) / len(val_data))
    if psnr > best_psnr:
        best_psnr = psnr
        best_psnr_epoch = nr_eval
        torch.save(model.state_dict(), f'ckpt/softmaxsplatting.pkl')
    if ssim > best_ssim:
        best_ssim = ssim
        best_ssim_epoch = nr_eval
    if local_rank == 0:
        print('epoch=', str(nr_eval), 'psnr=', psnr, 'best_psnr=', best_psnr, 'best_psnr_epoch=', best_psnr_epoch)
        print('epoch=', str(nr_eval), 'ssim=', ssim, 'best_ssim=', best_ssim, 'best_ssim_epoch=', best_ssim_epoch)
        # writer_val.add_scalar('psnr', psnr, nr_eval)
    return best_psnr, best_psnr_epoch, best_ssim, best_ssim_epoch


def denorm255(x):
    out = (x + 1.0) / 2.0
    return out.clamp_(0.0, 1.0) * 255.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--data_path', default='E:/WHU/VFI-master/Dataset', type=str, help='data path of vimeo90k')
    args = parser.parse_args()
    # torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    if args.local_rank == 0 and not os.path.exists('log'):
        os.mkdir('log')
    # seed = 1234
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = True
    model = Network()
    model.to(device)
    train(model, args.local_rank, args.batch_size, args.data_path)