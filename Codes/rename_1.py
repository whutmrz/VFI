import os

# 指定文件夹路径
folder_path = 'E:/WHU/VFI-master/Dataset/data/350W-1400'

# 获取文件夹中所有文件的列表
file_list = sorted(os.listdir(folder_path))

# 重命名计数器
count = 1

# 遍历文件列表并重命名文件
for filename in file_list:
    # 构造新的文件名
    new_filename = '{:03d}'.format(count) + os.path.splitext(filename)[1]

    # 构造原文件及新文件的完整路径
    old_filepath = os.path.join(folder_path, filename)
    new_filepath = os.path.join(folder_path, new_filename)

    # 重命名文件
    os.rename(old_filepath, new_filepath)

    # 更新计数器
    count += 2