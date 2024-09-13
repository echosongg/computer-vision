from math import sqrt

from skimage.filters import sobel
import matplotlib.pyplot as plt
import cv2
from skimage.filters import sobel_h
from skimage.filters import sobel_v
import numpy as np
import math


sobel_x = np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_y = np.asarray([[1,2,1],[0,0,0],[-1,-2,-1]])
def addPadding(image, kernel_size):
    padding = kernel_size // 2
    padding_img = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2),dtype="float64")
    padding_img[padding:padding + image.shape[0], padding:padding+image.shape[1]] = image
    return padding_img

def my_Sobel_filter(image, sobel_x = sobel_x, sobel_y = sobel_y):
    padding_img = addPadding(image,sobel_x.shape[0])
    padding = sobel_x.shape[0] // 2
    filtered_img = np.zeros(padding_img.shape,dtype="float64")
    for row in range(0,image.shape[0]):
        for col in range(0,image.shape[1]):
            kernel_template = padding_img[row:row+sobel_x.shape[0],col:col+sobel_x.shape[0]]
            #print(kernel_template)
            sum_result = math.sqrt(np.sum(kernel_template * sobel_x)**2 + np.sum(kernel_template * sobel_y)**2)
            filtered_img[row+padding][col+padding] = sum_result
    filtered_img = filtered_img[padding:padding+image.shape[0],padding:padding+image.shape[1]]
    filtered_img = np.clip(filtered_img,0,1)
    return (filtered_img)
def my_Sobel_hist(image_xx,image_yy):
    result = []
    for row in range(0, img_xx.shape[0]):
        for col in range(0, img_xx.shape[1]):
            result.append(math.atan2(img_yy[row][col], img_xx[row][col]))
    return result



#img_net = sobel(img)
img = cv2.imread("images/image4_s.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = img / 255.0
img_my = my_Sobel_filter(img)
img_xx= cv2.Sobel(img,ddepth= -1 ,dy=0,dx=1,ksize=3)
img_yy= cv2.Sobel(img,ddepth=-1,dx=0,dy=1,ksize=3)

img_t = np.sqrt(np.square(img_xx) + np.square(img_yy))
img_t = np.clip(img_t,0,1)
#plt.imshow(img_net,cmap='gray')
#plt.show()
plt.title("Cat, my own sobel filter result")
plt.imshow(img_my,cmap='gray')
plt.show()
plt.title("Cat, opencv sobel filter result")
plt.imshow(img_t,cmap='gray')
plt.show()
result = my_Sobel_hist(img_xx,img_yy)
plt.title("histogram of gradient direction of Cat")
plt.hist(result)
plt.show()


img = cv2.imread("images/image3.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = img / 255.0
img_my = my_Sobel_filter(img)
img_xx= cv2.Sobel(img,ddepth= -1 ,dy=0,dx=1,ksize=3)
img_yy= cv2.Sobel(img,ddepth=-1,dx=0,dy=1,ksize=3)
plt.show()
img_t = np.sqrt(np.square(img_xx) + np.square(img_yy))
img_t = np.clip(img_t,0,1)
#plt.imshow(img_net,cmap='gray')
#plt.show()
plt.title("NewYork, my own sobel filter result")
plt.imshow(img_my,cmap='gray')
plt.show()
plt.title("NewYork, opencv sobel filter result")
plt.imshow(img_t,cmap='gray')
plt.show()
result = my_Sobel_hist(img_xx,img_yy)
plt.title("histogram of gradient direction of New York city")
plt.hist(result)
plt.show()
print("Task5 ends")