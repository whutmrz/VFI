# -*- coding: utf-8 -*-
import shutil
import os
# 复制图像到另一个文件夹
# 文件所在文件夹
file_dir = 'E:/WHU/VFI-master/Dataset/GTs/350W-1400'
# 创建一个子文件存放文件
target_dir = 'E:/WHU/VFI-master/Dataset/test/350W-1400-half'

file_list = os.listdir(file_dir)

if bool(1-os.path.exists(target_dir)):
    os.mkdir(target_dir)

for image in file_list:
    image_inf = os.path.splitext(image)
    name = image_inf[0]

    if int(name)%2 == 1:
        shutil.copy(os.path.join(file_dir, image), os.path.join(target_dir, image))
