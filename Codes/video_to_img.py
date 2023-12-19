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


def GetFrame(videoPath, savePath, gapFrame=10):
    EnsureExistFloder(savePath)
    cap = cv2.VideoCapture(videoPath)
    assert cap.isOpened()
    numFrame = 0
    t = 0
    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                # frame = np.rot90(frame)
                #cv2.imshow('video', frame)
                numFrame += 1
                #print(numFrame)
                if ((numFrame+1)%gapFrame==0):
                    t += 1
                    newPath = os.path.join(savePath, '{:04d}.png'.format(t))
                    cv2.imencode('.png', frame)[1].tofile(newPath)
        else:
            break
    print("[uitls::GetFrame] Done. \n -> Video path: %s. \n -> Save path: %s"
        % (videoPath, savePath))

GetFrame('E:/WHU/VFI-master/Dataset/Video/original.mp4', 'E:/WHU/VFI-master/Dataset/Video/extracted')