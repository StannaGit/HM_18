
'''
Цифрова обробка зображень: фільтрація

Package                      Version
---------------------------- -----------
opencv-python                3.4.18.65
numpy                        1.23.5
pip                          23.1
matplotlib                   3.6.2

'''

import cv2
import numpy as np
from matplotlib import pyplot as plt


#----------------- 1: Усереднення зображення через згортку ---------
img = cv2.imread('cat.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
kernel = np.ones((10, 10), np.float32) /70
#------------------ виклик методу 2D Convolution ------------------
blur1 = cv2.filter2D(img, -1, kernel)
blur1_rgb = cv2.cvtColor(blur1, cv2.COLOR_BGR2RGB)
#-------------- відображення результату ----------------------------
plt.subplot(121), plt.imshow(img_rgb), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur1_rgb), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
cv2.imwrite('cat_filter2D.jpg', blur1)


#---------------- 2: Розмиття через згладжування ------------------
img = cv2.imread('cat.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#------------------- виклик методу - blur --------------------------
blur2 = cv2.blur(img, (15, 15))
blur2_rgb = cv2.cvtColor(blur2, cv2.COLOR_BGR2RGB)
#------------------- відображення результату -----------------------
plt.subplot(121), plt.imshow(img_rgb), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur2_rgb), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
cv2.imwrite('cat_blur.jpg', blur2)


#-------------------- 3: Гауссова фільтрація  -----------------------
img = cv2.imread('cat.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#------------------- виклик методу - GaussianBlur ------------------
blur3 = cv2.GaussianBlur(img, (15, 15), 0)
blur3_rgb = cv2.cvtColor(blur3, cv2.COLOR_BGR2RGB)
#------------------- відображення результату -----------------------
plt.subplot(121), plt.imshow(img_rgb), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur3_rgb), plt.title('GaussianBlur')
plt.xticks([]), plt.yticks([])
plt.show()
cv2.imwrite('cat_GaussianBlur.jpg', blur3)


#-------------------- 4: Медіанний фільтр  -------------------------
img = cv2.imread('cat.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#------------------- виклик методу -blur ---------------------------
blur4 = cv2.medianBlur(img, 15)
blur4_rgb = cv2.cvtColor(blur4, cv2.COLOR_BGR2RGB)
#------------------- відображення результату -----------------------
plt.subplot(121), plt.imshow(img_rgb), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur4_rgb), plt.title('Median Filtering')
plt.xticks([]), plt.yticks([])
plt.show()
cv2.imwrite('cat_medianBlur.jpg', blur4)


#------------------- 5: Двостороння фільтрація ---------------------
img = cv2.imread('cat.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#------------------- виклик методу -blur ---------------------------
blur5 = cv2.bilateralFilter(img, 20, 50, 50)
blur5_rgb = cv2.cvtColor(blur5, cv2.COLOR_BGR2RGB)
#------------------- відображення результату -----------------------
plt.subplot(121), plt.imshow(img_rgb), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur5_rgb), plt.title('Bilateral Filtering')
plt.xticks([]), plt.yticks([])
plt.show()
cv2.imwrite('cat_bilateralFilter.jpg', blur5)
