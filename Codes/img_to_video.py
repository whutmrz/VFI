from utils.utils import WriteVideo

input_path = 'E:/WHU/VFI-master/Dataset/test_vfi'
output_path = 'E:/WHU/VFI-master/Dataset/Video/original.mp4'
FPS = 10
Width = 300
Height = 200

WriteVideo(input_path, output_path, FPS, Width, Height)