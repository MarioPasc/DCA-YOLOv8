# 直方图均衡化：遍历图像每个像素的灰度，算出每个灰度的概率（n/MN-n是每个灰度的个数，MN是像素总数），用L-1乘以所得概率得到新的灰度

import cv2
import numpy as np
import os
def pix_gray(img_gray):
    h = img_gray.shape[0]
    w = img_gray.shape[1]

    gray_level = np.zeros(256)
    gray_level2 = np.zeros(256)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            gray_level[img_gray[i, j]] += 1  # 统计灰度级为img_gray[i,j的个数

    for i in range(1, 256):
        gray_level2[i] = gray_level2[i - 1] + gray_level[i]  # 统计灰度级小于img_gray[i,j]的个数

    return gray_level2


# 直方图均衡化

def hist_gray(img_gary):
    h, w = img_gary.shape
    gray_level2 = pix_gray(img_gray)
    lut = np.zeros(256)
    for i in range(256):
        lut[i] = 255.0 / (h * w) * gray_level2[i]  # 得到新的灰度级
    lut = np.uint8(lut + 0.5)
    out = cv2.LUT(img_gray, lut)
    return out
input_folder = '14_002_5_0027.jpeg'
output_folder = 'result_zhifang.jpeg'
# for filename in os.listdir(input_folder):
#     filename_1 = os.path.join(input_folder+'/'+filename)
img = cv2.imread(input_folder)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
out_img = hist_gray(img_gray)
# output_path = os.path.join(output_folder, filename)
cv2.imwrite(output_folder, out_img)

# 直方图统计




# cv2.imshow(' imput', img_gray)
# out_img = hist_gray(img_gray)

