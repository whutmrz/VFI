import os
import cv2
import torch
import numpy as np
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder

model = Model(-1)
model.load_model()
model.eval()
model.device()
TTA = True
def your_interpolation_algorithm(img1, img2):
    I0 = cv2.imread(img1)
    I2 = cv2.imread(img2)
    I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    padder = InputPadder(I0_.shape, divisor=32)
    I0_, I2_ = padder.pad(I0_, I2_)
    with torch.no_grad():
        mid = padder.unpad(model.our_inference(I0_, I2_, TTA=TTA, fast_TTA=TTA))[0]
    mid = (mid.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return mid
# 输入文件夹路径
input_folder_path = 'E:/WHU/VFI-master/Dataset/data/350W-1400'
input_image_files = sorted(os.listdir(input_folder_path))

# 输出文件夹路径
merged_folder_path = 'E:/WHU/VFI-master/Dataset/data/350W-1400-interpolated'
if not os.path.exists(merged_folder_path):
    os.makedirs(merged_folder_path)

merged_image_files = []
flag = 0
for i in range(len(input_image_files) - 1):
    input_image_path = os.path.join(input_folder_path, input_image_files[i])
    image1 = cv2.imread(input_image_path)
    cv2.imwrite(os.path.join(merged_folder_path, input_image_files[i]), image1)
    # 调用您的插帧算法处理图片，生成插入的帧
    interpolated_frame = your_interpolation_algorithm(input_image_path,
                                                      os.path.join(input_folder_path, input_image_files[i + 1]))
    flag += 2
    # 生成帧的文件名，确保在原始两帧之间
    interpolated_filename = '{:03d}.jpg'.format(int(flag))  # 生成帧的文件名，根据实际情况进行修改
    interpolated_filepath = os.path.join(merged_folder_path, interpolated_filename)  # 生成帧的文件路径
    cv2.imwrite(interpolated_filepath, interpolated_frame)  # 保存插入的帧到合并后的文件夹

    merged_image_files.append(input_image_files[i])  # 将原始图片添加到合并后的图片文件列表中
    merged_image_files.append(interpolated_filename)  # 将插入的帧文件名添加到合并后的图片文件列表中

# 添加最后一张原始图片
merged_image_files.append(input_image_files[-1])
image2 = cv2.imread(os.path.join(input_folder_path, input_image_files[-1]))
cv2.imwrite(os.path.join(merged_folder_path, input_image_files[-1]), image2)

# 打印合并后的图片文件列表
print("Merged Image Files:", merged_image_files)
