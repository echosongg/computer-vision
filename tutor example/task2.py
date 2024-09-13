import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from skimage.color import rgb2hsv
import math
# Task2

# Task2.1
im1 = cv2.imread("images/image1.jpg")
im1 = cv2.resize(im1, (1042, 720))
cv2.imwrite("images/image1.jpg", im1)
im2 = cv2.imread("images/image2.jpg")
im2 = cv2.resize(im2, (1042, 720))
cv2.imwrite("images/image2.jpg", im2)
im3 = cv2.imread("images/image3.jpg")
im3 = cv2.resize(im3, (1042, 720))
cv2.imwrite("images/image3.jpg", im3)
im4 = cv2.imread("images/image4.jpg")

# because the task2 does not ask to resize the image4, but for convenience in later task(task5) I resize it, and store it in a different name
# which will not modify the source image.
im4 = cv2.resize(im4, (1042, 720))
cv2.imwrite("images/image4_s.jpg", im4)
# Task2.2
# Task2.2.a
origin_im1 = cv2.imread("images/image1.jpg")
im1 = cv2.resize(origin_im1, (384, 256))

cv2.imshow("Origin Figure after resizing", im1)

# Task2.2.b
b, g, r = cv2.split(im1)
cv2.imshow("R Channel", r)
cv2.imshow("G Channel", g)
cv2.imshow("B Channel", b)

# im4 = Image.open("images/image1.jpg")
# im4 = np.asarray(im4)
# im5 = plt.imread("images/image1.jpg")


# Task2.2.c
hsb = cv2.calcHist([im1], [0], None, [256], [0, 255])
plt.title("Histogram for B Channel")
plt.plot(hsb)
plt.show()

hsg = cv2.calcHist([im1], [1], None, [256], [0, 255])
plt.title("Histogram for G Channel")
plt.plot(hsg)
plt.show()

hsr = cv2.calcHist([im1], [2], None, [256], [0, 255])
plt.title("Histogram for R Channel")
plt.plot(hsr)
plt.show()

# Task2.2.d
hsb_ec = cv2.equalizeHist(b)
hsb_e = cv2.calcHist([hsb_ec], [0], None, [256], [0, 255])
plt.plot(hsb_e)
plt.title("B Channel After histogram equalisation")
plt.show()

hsg_ec = cv2.equalizeHist(g)
hsg_e = cv2.calcHist([hsg_ec], [0], None, [256], [0, 255])
plt.plot(hsg_e)
plt.title("G Channel After histogram equalisation")
plt.show()

hsr_ec = cv2.equalizeHist(r)
hsr_e = cv2.calcHist([hsr_ec], [0], None, [256], [0, 255])
plt.plot(hsr_e)
plt.title("R Channel After histogram equalisation")
plt.show()

im1_e = cv2.merge([hsb_ec, hsg_ec, hsr_ec])
cv2.imshow("Figure after histogram equalisation", im1_e)

cv2.waitKey(0)
cv2.destroyAllWindows()
print("Task2 ends")