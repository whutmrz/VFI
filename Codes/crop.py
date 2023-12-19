import numpy as np
import shutil
import cv2
import os
from tqdm import tqdm

def EnsureExistFloder(path):
    # 'data/frames'
    # ['data', 'frames']
    sonpath = path.split("/")
    for i in range(len(sonpath)):
        path = ''
        for sp in sonpath[:i + 1]:
            path += (sp + '/')
        if sp != ".":
            if bool(1 - os.path.exists(path)):
                print("[utils::EnsureExistFloder] Invalid Existence of %s, creating it." % str(path))
                os.mkdir(path)

def CropPics(FramePath: str, SavePath: str, H0: int, H1: int, W0: int, W1: int):
    print("[uitls::CropPics] \n -> FramePath: %s. \n -> VideoPath: %s. \n -> Resolution: %d x %d"
        % (FramePath, SavePath, W1-W0, H1-H0))
    EnsureExistFloder(SavePath)
    framelist = os.listdir(FramePath)
    with tqdm(total=len(framelist)) as pbar:
        pbar.set_description('[uitls::CropPics]')
        for file in framelist:
            frame = cv2.imread(os.path.join(FramePath, file))
            rtn = cv2.imwrite(os.path.join(SavePath,file), frame[W0:W1,H0:H1,:3])
            pbar.update(1)

CropPics('E:/WHU/VFI-master/Dataset/LPBFdataset_large/00001', 'E:/WHU/VFI-master/Dataset/LPBFdataset_large/00001_test', 280, 980, 100, 800)
