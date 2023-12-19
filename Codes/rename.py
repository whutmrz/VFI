# -*- coding:utf8 -*-

import os


class BatchRename():
    '''
    批量重命名文件夹中的图片文件
    '''

    def __init__(self):
        self.path = 'E:/WHU/VFI-master/unet-pytorch-main/img'  # 表示需要命名处理的文件夹

    def rename(self):
        filelist = os.listdir(self.path)  # 获取文件路径
        total_num = len(filelist)  # 获取1文件长度（个数）
        i = 1  # 表示文件的命名是从开始的

        for item in filelist:
            if item.endswith('.png'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path),
                                str("%04d"%i) + '.png')  # 处理后的格式也为jpg格式的，当然这里可以改成png格式
                try:
                    os.rename(src, dst)
                    # print ('converting %s to %s ...' % (src, dst))
                    i += 1
                except:
                    continue
        print('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()