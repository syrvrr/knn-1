import cv2
from matplotlib import pyplot as plt
import numpy as np
# Perhitungan Gradient dengan Python
# Membaca Gambar
im = cv2.imread('andip.jpg')
im = np.float32(im) / 255.0

# Perhitungan Gradient
gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)

mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
plt.imshow(mag)
#plt.imshow(angle)
plt.title('Hasil Komputasi Magnitude')
plt.show()
