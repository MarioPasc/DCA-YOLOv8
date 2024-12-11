import cv2
import numpy as np
import os
# Open the image.
img = cv2.imread('14_002_5_0027.jpeg')

# Trying 4 gamma values.
# for gamma in [2.0]:
#     # Apply gamma correction.
#     gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
#
#     # Save edited images.
#     cv2.imwrite('gamma_transformed' + str(gamma) + '.jpg', gamma_corrected)
gamma = 2.0
# input_folder = 'F:/xinguanshuju'
# output_folder = 'F:/xinguanjiama'
# for filename in os.listdir(input_folder):
#     filename_1 = os.path.join(input_folder+'/'+filename)
# img = cv2.imread(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
out_img = np.array(255 * (img / 255) ** gamma, dtype='uint8')
# out_img = hist_gray(img_gray)
# output_path = os.path.join(output_folder, filename)
cv2.imwrite('out_put_jiama.jpeg',out_img)
