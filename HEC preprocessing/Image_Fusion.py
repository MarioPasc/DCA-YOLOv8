import cv2
import numpy as np
import os


input_1 = 'clahe.jpg'
input_2 = 'Cany.jpg'
img1 = cv2.imread(input_1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(input_2, cv2.IMREAD_GRAYSCALE)
img1 = img1.astype(np.float32)
img2 = img2.astype(np.float32)
result_img = np.clip( img1 + 0.3 * img2, 0, 255).astype(np.uint8)
# result_path = os.path.join(out_put, img_name1)
cv2.imwrite('result_fusion.jpeg', result_img)