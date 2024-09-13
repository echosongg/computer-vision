import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from skimage.color import rgb2hsv

original_im = cv2.imread("images/image4.jpg")

length = original_im.shape[0]
width = original_im.shape[1]
min_edge = min(length, width)
shift_x = (length - min_edge) // 2
shift_y = (width - min_edge) // 2

crop_img = original_im[shift_x:shift_x + min_edge, shift_y:shift_y + min_edge, :]

resize_img = cv2.resize(crop_img, (512, 512))

gray_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("images/image4_gray.jpg", gray_img)
plt.imshow(gray_img, cmap="gray")
plt.title("the gray scale img with 512*512")
plt.show()
original_im = cv2.cvtColor(original_im,cv2.COLOR_BGR2RGB)
plt.imshow(original_im)
plt.title("the original image")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

noise = np.random.normal(0, 15, gray_img.shape)


plt.imshow(gray_img, cmap='gray')
plt.title("raw img before adding noise")
plt.show()

plt.title("random noise")
plt.imshow(noise, cmap='gray')
plt.show()

raw_noise_img = gray_img + noise


noise_img = np.clip(raw_noise_img, 0.0, 255.0)
noise_img = np.uint8(noise_img)

plt.title("Image with Gaussian noise")
plt.imshow(noise_img, cmap='gray')
plt.show()

hist_noise = cv2.calcHist([noise_img], [0], None, [256], [0, 255])
hist_raw = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

plt.title("gray scale histogram of raw gray scale image ")
plt.plot(hist_raw)
plt.show()

plt.title("gray scale histogram of image with noise")
plt.plot(hist_noise)
plt.show()


def get2dGaussianKernel(sigma, kernel_size=7, ktype=cv2.CV_32F):
    kernel_1d = cv2.getGaussianKernel(kernel_size, sigma, ktype)
    return kernel_1d * kernel_1d.T



def addPadding(image, kernel_size):
    padding = kernel_size // 2
    padding_img = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2),dtype="float64")
    padding_img[padding:padding + image.shape[0], padding:padding+image.shape[1]] = image
    return padding_img

padding_img = addPadding(gray_img, 7)

def my_Gauss_filter(noise_image, gaussian_kernel):
    padding_img = addPadding(noise_image,gaussian_kernel.shape[0])
    padding = gaussian_kernel.shape[0] // 2
    filtered_img = np.zeros(padding_img.shape,dtype="float64")
    for row in range(0,noise_image.shape[0]):
        for col in range(0,noise_image.shape[1]):
            kernel_template = padding_img[row:row+gaussian_kernel.shape[0],col:col+gaussian_kernel.shape[0]]
            #print(kernel_template)
            sum_result = np.sum(kernel_template * gaussian_kernel)
            filtered_img[row+padding][col+padding] = sum_result
    filtered_img = filtered_img[padding:padding+noise_image.shape[0],padding:padding+noise_image.shape[1]]
    filtered_img = np.clip(filtered_img,0,255)
    return np.uint8(filtered_img)

kernel10 = get2dGaussianKernel(10)
filter_img_10 = my_Gauss_filter(noise_img,kernel10)
plt.title("filter sigma  is 10")
plt.imshow(filter_img_10,cmap='gray')
plt.show()

kernel15 = get2dGaussianKernel(15)
filter_img_15 = my_Gauss_filter(noise_img,kernel15)
plt.title("filter sigma  is 15")
plt.imshow(filter_img_15,cmap='gray')
plt.show()

kernel20 = get2dGaussianKernel(20)
filter_img_20 = my_Gauss_filter(noise_img,kernel20)
plt.title("filter sigma  is 20")
plt.imshow(filter_img_20,cmap='gray')
plt.show()

kernel1 = get2dGaussianKernel(1)
filter_img_1 = my_Gauss_filter(noise_img,kernel1)
plt.title("filter sigma  is 1")
plt.imshow(filter_img_1,cmap='gray')
plt.show()

kernel01 = get2dGaussianKernel(0.1)
filter_img_01 = my_Gauss_filter(noise_img,kernel01)
plt.title("filter sigma  is 0.1")
plt.imshow(filter_img_01,cmap='gray')
plt.show()


built_in_res = cv2.GaussianBlur(noise_img,(7,7),1)
plt.title("result of built-in function(sigma = 1)")
plt.imshow(built_in_res,cmap='gray')
plt.show()
print("Task4 ends")