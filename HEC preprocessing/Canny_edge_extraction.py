import cv2
import os
# import Image
# 读取图像
# input_folder = 'F:/xinguanshujuhuizong/xinguanzhifanghua'
# output_folder = 'F:/xinguanshujuhuizong/zhifanglunkuo'
# for filename in os.listdir(input_folder):
filename_1 = '14_002_5_0027.jpeg'
img = cv2.imread(filename_1)
# image = cv2.imread('zhifang.jpeg', 0)

# 应用高斯滤波，减少噪声影响
gaussian = cv2 .GaussianBlur(img, (5, 5), 0)

# 使用Canny算法进行边缘检测
edges = cv2.Canny(gaussian, threshold1=10, threshold2=35)

# 显示结果
cv2.imshow("Image", edges)
cv2.imwrite('yuantulunkuo.jpg',edges)
# 等待用户按下任意键后关闭窗口
cv2.waitKey(0)
# 关闭窗口
cv2.destroyAllWindows()
