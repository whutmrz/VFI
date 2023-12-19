import torch.utils.data as data
import os, random, cv2, torch
import numpy as np
from glob import glob
import torch.nn.functional as F
# adopted from XVFI (https://github.com/JihyongOh/XVFI)

class Custom_Train(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.t = 0.5
        self.framesPath = []
        f = open(os.path.join(args.train_data, 'train.txt'),
                 'r')  # './datasets/vimeo_triplet/sequences/tri_trainlist.txt'
        while True:
            scene_path = f.readline().split('\n')[0]
            if not scene_path: break
            frames_list = sorted(glob(os.path.join(args.train_data, 'LPBFdataset', scene_path, '*.png')))
            self.framesPath.append(frames_list)
        f.close
        self.nScenes = len(self.framesPath)
        if self.nScenes == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + args.train_data + "\n"))
        print("nScenes of Vimeo train triplet : ", self.nScenes)
    
    def __getitem__(self, idx):
        candidate_frames = self.framesPath[idx]
        """ Randomly reverse frames """
        frameRange = [0, 2, 1]

        # frames : (C, T, H, W)
        frames = frames_loader_train(self.args, candidate_frames, frameRange)  # including "np2Tensor [-1,1] normalized"
        
        outputs = {'x0': frames[:,0,:,:], 'x1': frames[:,1,:,:], 'y': frames[:,2,:,:] ,'time': np.expand_dims(np.array(0.5, dtype=np.float32), 0)}
        # padder0 = InputPadder(outputs['x0'].shape, divisor=32)
        # padder1 = InputPadder(outputs['x1'].shape, divisor=32)
        # padder2 = InputPadder(outputs['y'].shape, divisor=32)
        # outputs['x0'] = padder0.pad(outputs['x0'])
        # outputs['x1'] = padder1.pad(outputs['x1'])
        # outputs['y'] = padder2.pad(outputs['y'])
        return outputs

    def __len__(self):
        return self.nScenes


class Custom_Test(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.t = 0.5
        self.framesPath = []
        f = open(os.path.join(args.train_data, 'test.txt'),
                 'r')  # './datasets/vimeo_triplet/sequences/tri_trainlist.txt'
        while True:
            scene_path = f.readline().split('\n')[0]
            if not scene_path: break
            frames_list = sorted(glob(os.path.join(args.train_data, 'LPBFdataset', scene_path, '*.png')))
            self.framesPath.append(frames_list)
        f.close
        self.nScenes = len(self.framesPath)
        if self.nScenes == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + args.train_data + "\n"))
        print("nScenes of Vimeo train triplet : ", self.nScenes)

    def __getitem__(self, idx):
        candidate_frames = self.framesPath[idx]
        """ Randomly reverse frames """
        # if (random.randint(0, 1)):
        #     frameRange = [0, 2, 1]
        # else:
        #     frameRange = [2, 0, 1]
        # frames : (C, T, H, W)
        frameRange = [0, 2, 1]
        frames = frames_loader_train(self.args, candidate_frames, frameRange)  # including "np2Tensor [-1,1] normalized"

        outputs = {'x0': frames[:, 0, :, :], 'x1': frames[:, 1, :, :], 'y': frames[:, 2, :, :],
                   'time': np.expand_dims(np.array(0.5, dtype=np.float32), 0)}
        # padder0 = InputPadder(outputs['x0'].shape, divisor=32)
        # padder1 = InputPadder(outputs['x1'].shape, divisor=32)
        # padder2 = InputPadder(outputs['y'].shape, divisor=32)
        # outputs['x0'] = padder0.pad(outputs['x0'])
        # outputs['x1'] = padder1.pad(outputs['x1'])
        # outputs['y'] = padder2.pad(outputs['y'])
        return outputs

    def __len__(self):
        return self.nScenes

class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """
    def __init__(self, dims, divisor = 16):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    def pad(self, inputs):
        return F.pad(inputs, self._pad, mode='replicate')

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def frames_loader_train(args, candidate_frames, frameRange):
    frames = []
    for frameIndex in frameRange:
        frame = cv2.imread(candidate_frames[frameIndex])
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    (ih, iw, c) = frame.shape
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)

    if args.need_patch:  ## random crop
        ps = args.patch_size
        ix = random.randrange(0, iw - ps + 1)
        iy = random.randrange(0, ih - ps + 1)
        frames = frames[:, iy:iy + ps, ix:ix + ps, :]

    # if random.random() < 0.5:  # random horizontal flip
    #     frames = frames[:, :, ::-1, :]

    frames = frames[:, :, :,:] # (T, H, W, 3) , H and W should be divided by 2**(pyramid_levels)
    # No vertical flip

    """rot = random.randint(0, 3)  # random rotate
    frames = np.rot90(frames, rot, (1, 2))"""

    """ np2Tensor [-1,1] normalized """
    frames = RGBframes_np2Tensor(frames, 3) # (C, T, H, W)

    return frames

def RGBframes_np2Tensor(imgIn, channel):
    ## input : T, H, W, C
    if channel == 1:
        # rgb --> Y (gray)
        imgIn = np.sum(imgIn * np.reshape([65.481, 128.553, 24.966], [1, 1, 1, 3]) / 255.0, axis=3,
                       keepdims=True) + 16.0

    # to Tensor
    ts = (3, 0, 1, 2)  # dimension order should be [C, T, H, W]
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)
    
    # normalization [-1,1]
    imgIn = (imgIn / 255.0 - 0.5) * 2

    return imgIn

def create_training_dataset(args, augmentation_fns=None):
    data_train = Custom_Train(args)
    dataloader = data.DataLoader(data_train, batch_size=args.batch_size, drop_last=True, shuffle=False, num_workers=1, pin_memory=True)
    return dataloader

def create_testing_dataset(args, augmentation_fns=None):
    data_test = Custom_Test(args)
    dataloader = data.DataLoader(data_test, batch_size=args.batch_size, drop_last=True, shuffle=False, num_workers=1, pin_memory=True)
    return dataloader
