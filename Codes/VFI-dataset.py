# -*- coding:utf8 -*-

import os
import shutil

class BatchRename():
    '''
    批量重命名文件夹中的图片文件
    '''

    def __init__(self, path):
        self.path = path  # 表示需要命名处理的文件夹

    def rename(self):
        filelist = os.listdir(self.path)  # 获取文件路径
        total_num = len(filelist)  # 获取1文件长度（个数）
        i = 1  # 表示文件的命名是从开始的

        for item in filelist:
            if item.endswith('.png'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), 'im' +
                                str(i) + '.png')  # 处理后的格式也为jpg格式的，当然这里可以改成png格式
                try:
                    os.rename(src, dst)
                    # print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print('total %d to rename & converted %d jpgs' % (total_num, i))


if __name__ == '__main__':
   local_path = 'E:/WHU/VFI-master/Dataset/experiments/00025'
   filelist = os.listdir(local_path)
   num_img = len(filelist)
   num_folder = int(num_img / 3)
   flag = 1
   for i in range(num_folder):
       folder_path = os.path.join(local_path, "%04d"%(i+1))
       if not os.path.exists(folder_path):
           os.makedirs(folder_path)
       for file in filelist:
           file_name0 = os.path.splitext(file)[0]
           if file_name0 == "%04d"%(flag) or file_name0 == "%04d"%(flag+1) or file_name0 == "%04d"%(flag+2):
               shutil.move(os.path.join(local_path, file), folder_path)

       flag += 3
       demo = BatchRename(folder_path)
       demo.rename()
