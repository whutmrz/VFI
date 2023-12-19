"""
Converts a Video to SuperSloMo version
"""
from time import time
import click
import cv2
import torch
from PIL import Image
import numpy as np
# import model
from torchvision import transforms
from torch.functional import F
from Trainer import Model
from benchmark.utils.padder import InputPadder

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans_forward = transforms.ToTensor()
trans_backward = transforms.ToPILImage()
model = Model(-1)
model.load_model()
model.eval()
model.device()
if device != "cpu":
    mean = [0.429, 0.431, 0.397]
    mea0 = [-m for m in mean]
    std = [1] * 3
    trans_forward = transforms.Compose([trans_forward, transforms.Normalize(mean=mean, std=std)])
    trans_backward = transforms.Compose([transforms.Normalize(mean=mea0, std=std), trans_backward])


def setup_back_warp(w, h):
    global back_warp
    with torch.set_grad_enabled(False):
        back_warp = model.backWarp(w, h, device).to(device)


def interpolate_batch(frames, factor):
    # frame0 = torch.stack(frames[:-1])
    # frame1 = torch.stack(frames[1:])
    frame_buffer = []
    for i in range(0, len(frames)-1):
        frame0 = frames[i]
        frame1 = frames[i+1]
        i0 = frame0.to(device)
        i1 = frame1.to(device)
        for i in range(1, factor):
            ft_p = model.inference(i0, i1, TTA=True, fast_TTA=True)
            frame_buffer.append(ft_p)

    return frame_buffer


def load_batch(video_in, batch_size, batch):
    if len(batch) > 0:
        batch = [batch[-1]]

    for i in range(batch_size):
        ok, frame = video_in.read()
        if not ok:
            break
        frame = (torch.tensor(frame.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

        padder = InputPadder(frame.shape, divisor=32)
        frame = padder.pad(frame)[0]
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = Image.fromarray(frame)
        # frame = frame.resize((w, h), Image.ANTIALIAS)
        # frame = frame.convert('RGB')
        # frame = trans_forward(frame)
        batch.append(frame)

    return batch


def denorm_frame(frame, padder, w0, h0):
    # frame = frame.cpu()
    # frame = trans_backward(frame)
    # frame = frame.resize((w0, h0), Image.BILINEAR)
    # frame = frame.convert('RGB')
    frame = padder.unpad(frame).detach().cpu().numpy().transpose(1, 2, 0) * 255.0
    frame = frame.astype(np.uint8)
    return frame[:, :, ::-1]


def convert_video(source, dest, factor, batch_size=2, output_format='mp4v', output_fps=30):
    vin = cv2.VideoCapture(source)
    count = vin.get(cv2.CAP_PROP_FRAME_COUNT)
    w0, h0 = int(vin.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vin.get(cv2.CAP_PROP_FRAME_HEIGHT))

    codec = cv2.VideoWriter_fourcc(*output_format)
    vout = cv2.VideoWriter(dest, codec, float(output_fps), (w0, h0))
    padder = InputPadder([h0, w0], divisor=32)
    done = 0
    batch = []
    while True:
        batch = load_batch(vin, batch_size, batch)
        if len(batch) == 1:
            break
        done += len(batch) - 1

        intermediate_frames = interpolate_batch(batch, factor)
        # intermediate_frames = list(zip(*intermediate_frames))

        for fid, iframe in enumerate(intermediate_frames):
            vout.write(denorm_frame(batch[fid][0], padder, w0, h0))
            for frm in iframe:
                vout.write(denorm_frame(frm, padder, w0, h0))

        try:
            yield len(batch), done, count
        except StopIteration:
            break

    vout.write(denorm_frame(batch[0][0], padder, w0, h0))

    vin.release()
    vout.release()


@click.command('Evaluate Model by converting a low-FPS video to high-fps')
@click.argument('input', default='E:/WHU/VFI-master/Dataset/Video/extracted.mp4')
@click.option('--checkpoint', default='E:/WHU/VFI-master/SOTA-Super-SloMo/data/Superslomo.ckpt', help='Path to model checkpoint')
@click.option('--output', default='E:/WHU/VFI-master/Dataset/Video/interpolated.mp4', help='Path to output file to save')
@click.option('--batch', default=2, help='Number of frames to process in single forward pass')
@click.option('--scale', default=2, help='Scale Factor of FPS')
@click.option('--fps', default=10, help='FPS of output video')
def main(input, checkpoint, output, batch, scale, fps):
    avg = lambda x, n, x0: (x * n/(n+1) + x0 / (n+1), n+1)
    t0 = time()
    n0 = 0
    fpx = 0
    for dl, fd, fc in convert_video(input, output, int(scale), int(batch), output_fps=int(fps)):
        fpx, n0 = avg(fpx, n0, dl / (time() - t0))
        prg = int(100*fd/fc)
        eta = (fc - fd) / fpx
        print('\rDone: {:03d}% FPS: {:05.2f} ETA: {:.2f}s'.format(prg, fpx, eta) + ' '*5, end='')
        t0 = time()

if __name__ == '__main__':
    main()