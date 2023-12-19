import cv2

# 加载彩色图像
color_image = cv2.imread('C:/Users/Mr.Z/Desktop/1.png')

# 将彩色图像转换为灰度图像
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# 保存黑白图像
cv2.imwrite('output.png', gray_image)
