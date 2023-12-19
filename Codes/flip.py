'''
Python批量实现图像水平、垂直翻转
函数功能：扩大数据量
'''
import PIL.Image as img
import os

path_old = r'E:\WHU\VFI-master\Dataset\GTs\350W-1400'
path_new = r'E:\WHU\VFI-master\Dataset\GTs\350W-1400'
filelist = os.listdir(path_old)
total_num = len(filelist)
print(total_num)

for i,subdir in enumerate(filelist):
    sub_dir = path_old + '/' + subdir

    im = img.open(sub_dir)
    # ng = im.transpose(img.ROTATE_180) #旋转 180 度角。
    ng = im.transpose(img.FLIP_LEFT_RIGHT) #左右对换。
    #     ng = im.transpose(img.FLIP_TOP_BOTTOM)  # 上下对换。
    # ng=im.transpose(img.FLIP_LEFT_RIGHT) #左右对换。
    # ng=im.transpose(img.FLIP_TOP_BOTTOM) #上下对换。
    # ng=im.transpose(Image.ROTATE_270) #旋转 270 度角。
    #ng = im.rotate(180)  # 逆时针旋转 45 度角。
    ng = ng.convert('RGB')# 彩图必须要加上这个！！
    ng.save(path_new + '/' + "f-"+subdir)

print('done')