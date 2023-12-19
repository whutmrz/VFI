from utils.utils import *

'''
Prepare dateset.
Remove files by hands.
'''
laser_settings = [
    '150W-600','150W-800','150W-1000','150W-1200','150W-1400',
    '200W-600','200W-800','200W-1000','200W-1200','200W-1400',
    '250W-600','250W-800','250W-1000','250W-1200','250W-1400',
    '300W-600','300W-800','300W-1000','300W-1200','300W-1400',
    '350W-600','350W-800','350W-1000','350W-1200','350W-1400'
    ]
for laser_setting in laser_settings:
    print('\n' + '='*10 + laser_setting + '='*10)
    GetFrame('Dataset/%s.avi' % laser_setting, 'Dataset/%s/raw' % laser_setting)
    CropPics('Dataset/%s/raw' % laser_setting, 'Dataset/%s/crop' % laser_setting, 300, 500, 700, 1000)
    # WriteVideo('Dataset/%s/crop' % laser_settings, "Dataset/%s/crop.mp4" % laser_settings, 30, 200, 300)
    shutil.rmtree('Dataset/%s/raw' % laser_setting)
    mvfiles2dir('Dataset/%s/crop/' % laser_setting, 'Dataset/%s' % laser_setting)

# GetFrame('LPBF/CROP.mp4', 'LPBF/CROP_0.5/', 2)
# WriteVideo('LPBF/CROP_0.5/', "LPBF/CROP_0.5.mp4", 30, 288, 1280)

# GetFrame('SOTA-Super-SloMo/data/output.mp4', 'SOTA-Super-SloMo/data/output/')

# WriteVideo_Concat('LPBF/CROP/', 'LPBF/CROP_0.5/', "LPBF/CROP_CONCAT_6FPS.mp4", 6, 288*2, 1280)

# WriteVideo('Dataset/GTs/350W-1000/', "Doc/350w-1000-gt.mp4", 10, 200, 300)
# WriteVideo('Dataset/Labels/350W-1000/', "Doc/350w-1000-label.mp4", 10, 200, 300)
