import cv2
import os
from tqdm import tqdm
import numpy as np
import shutil

def mvfiles2dir(original_path, target_path):
    filenames = os.listdir(original_path)
    for fn in filenames:
        shutil.move(os.path.join(original_path, fn), os.path.join(target_path, fn))
    shutil.rmtree(original_path)

def EnsureExistFloder(path):
    # 'data/frames'
    # ['data', 'frames']
    sonpath = path.split("/")
    for i in range(len(sonpath)):
        path = ''
        for sp in sonpath[:i+1]:
            path += (sp+'/')
        if sp!=".":
            if bool(1-os.path.exists(path)):
                print("[utils::EnsureExistFloder] Invalid Existence of %s, creating it." % str(path))
                os.mkdir(path)

# Function: Get frames from video with interval(gapFrame)
def GetFrame(videoPath, savePath, gapFrame=1):
    EnsureExistFloder(savePath)
    cap = cv2.VideoCapture(videoPath)
    numFrame = 0
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
                    newPath = os.path.join(savePath, '{:03d}.png'.format(numFrame))
                    cv2.imencode('.png', frame)[1].tofile(newPath)
        else:
            break
    print("[uitls::GetFrame] Done. \n -> Video path: %s. \n -> Save path: %s" 
        % (videoPath, savePath))

def WriteVideo(FramePath: str, VideoPath: str, FPS: int, Width: int, Height: int):
    print("[uitls::WriteVideo] \n -> FramePath: %s. \n -> VideoPath: %s. \n -> FPS: %d, resolution: %d x %d" 
        % (FramePath, VideoPath, FPS, Width, Height))
    out = cv2.VideoWriter(VideoPath, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (Width, Height))
    framelist = os.listdir(FramePath)
    framelist.sort() # key=lambda x:int(x[5:-4])
    with tqdm(total=len(framelist)) as pbar:
        pbar.set_description('[uitls::WriteVideo]')
        for file in framelist:
            frame = cv2.imread(os.path.join(FramePath, file))
            assert(frame.shape[0]==Height)
            assert(frame.shape[1]==Width)
            out.write(frame[:,:,:3])
            pbar.update(1)

    out.release()
    print("[uitls::WriteVideo] Done.")

def WriteVideo_Concat(FramePath0: str, FramePath1: str,  VideoPath: str, FPS: int, Height: int, Width: int):
    print("[uitls::WriteVideo] \n -> FramePath0: %s, FramePath1: %s.\n -> VideoPath: %s. \n -> FPS: %d, resolution: %d x %d" 
        % (FramePath0, FramePath1, VideoPath, FPS, Width, Height))
    out = cv2.VideoWriter(VideoPath, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (Width, Height))
    framelist0 = os.listdir(FramePath0)
    framelist1 = os.listdir(FramePath1)
    framelist0.sort() # key=lambda x:int(x[5:-4])
    with tqdm(total=len(framelist0)) as pbar:
        pbar.set_description('[uitls::WriteVideo]')
        for ite in range(len(framelist0)):
            frame0 = cv2.imread(os.path.join(FramePath0, framelist0[ite]))
            assert(frame0.shape[0]==Height/2)
            assert(frame0.shape[1]==Width)
            frame1 = cv2.imread(os.path.join(FramePath1, framelist1[ite//2]))
            assert(frame1.shape[0]==Height/2)
            assert(frame1.shape[1]==Width)
            frame = np.concatenate([frame1, frame0], 0)
            out.write(frame[:,:,:3])
            pbar.update(1)
    out.release()
    print("[uitls::WriteVideo] Done.")

def CropPics(FramePath: str, SavePath: str, H0: int, H1: int, W0: int, W1: int):
    print("[uitls::CropPics] \n -> FramePath: %s. \n -> VideoPath: %s. \n -> Resolution: %d x %d" 
        % (FramePath, SavePath, W1-W0, H1-H0))
    EnsureExistFloder(SavePath)
    framelist = os.listdir(FramePath)
    with tqdm(total=len(framelist)) as pbar:
        pbar.set_description('[uitls::CropPics]')
        for file in framelist:
            frame = cv2.imread(os.path.join(FramePath, file))
            rtn = cv2.imwrite(os.path.join(SavePath,file), frame[H0:H1,W0:W1,:3])
            pbar.update(1)