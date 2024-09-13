import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from skimage.color import rgb2hsv
import math
# Task 1
a = np.array([[1, 2, 3], [5, 2, 20]])
# print('a', a.shape)
b = a[1, :]
# print('b', b.shape)
f = np.random.randn(200, 1)
# print('f', f.shape)
g = f[f > 0]
# print('g', g.shape)
x = np.zeros(50) + 0.5
# print('x', x.shape)
y = 0.5 * np.ones([1, len(x)])
# print('y', y.shape)
z = x + y
# print('z', z.shape)
a = np.linspace(1, 200)
# print('a', a.shape, a)
b = a[::-1]
# print('b', b.shape, b)
b[b < 35] = 0
# print("===========================")
# print('a', a.shape, a)
# print('b', b.shape, b)
print("Task1 ends")

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




im2_1 = cv2.imread("images/Figure2-a.png")
# imm = plt.imread("images/Figure2-a.png")
# plt.imshow(imm)
# plt.show()
def cvRGB2HSV_pixel(b, g, r):
    normalized_b = b / 255.0
    normalized_g = g / 255.0
    normalized_r = r / 255.0
    # print(normalized_b)
    x_max = max(normalized_b, max(normalized_g, normalized_r))
    v = x_max
    x_min = min(normalized_b, min(normalized_g, normalized_r))
    c = x_max - x_min
    if c == 0:
        h = 0
    elif v == normalized_r:
        if normalized_g >= normalized_b:
            h = ((normalized_g - normalized_b) / c) * 60
        else:
            h = ((normalized_g - normalized_b) / c) * 60 + 360
    elif v == normalized_g:
        h = ((normalized_b - normalized_r) / c + 2) * 60
    elif v == normalized_b:
        h = ((normalized_r - normalized_g) / c + 4) * 60

    if v == 0:
        s = 0
    else:
        s = c / v

    H = (h / 2.0) / 180.0
    S = s
    V = v
    return (H, S, V)


def cvRGB2HSV(image):
    output = np.zeros(image.shape)
    for row in range(0, image.shape[0]):
        for col in range(0, image.shape[1]):
            output[row][col] = cvRGB2HSV_pixel(image[row][col][0], image[row][col][1], image[row][col][2])
    return output


# q = np.asarray([[[128, 100, 200]]])
# cv2.imwrite("images/q.jpg", q)
# q = cv2.imread("images/q.jpg")
# print(q)
# a = cv2.cvtColor(q, cv2.COLOR_BGR2HSV)
# print("a", a)
# print(cvRGB2HSV_pixel(128, 100, 200))
# print("=======================")
# print(im1[0][0][0])
figure_after_HSV = cvRGB2HSV(im2_1)
# p = q
# q = cv2.cvtColor(im2_1, cv2.COLOR_BGR2HSV)
# o = q
# print(q.shape)
# print(p.shape)
# print(p)
# r = q == p
# print(cvRGB2HSV(im1))
# print(cv2.cvtColor(im1,cv2.COLOR_BGR2HSV))
# print("=======")
h, s, v = cv2.split(figure_after_HSV)
# h1, s1, v1 = cv2.split(figure_after_HSV)
# plt.subplot(1, 2, 1)
# plt.title("origin h")
# plt.imshow(h1, cmap='gray')
# plt.subplot(1, 2, 2)
plt.title("Task 3.1 the H Channel")
plt.imshow(h, cmap='gray')
plt.show()

plt.title("Task 3.1 the V Channel")
plt.imshow(v, cmap='gray')
plt.show()

plt.title("Task 3.1 the S Channel")
plt.imshow(s, cmap='gray')
plt.show()

# Task 3.2
imb = cv2.imread("images/Figure2-b.png")
imb_hsv = cvRGB2HSV(imb)
imb_H, imb_S, imb_V = cv2.split(imb_hsv)
len_imb = imb_H.shape[1]
step_size = len_imb // 5 + 1
index = 0

plt.imshow(cv2.cvtColor(imb, cv2.COLOR_BGR2RGB))

loop = 0
while (index < len_imb):
    loop += 1
    acc_index = index + step_size
    if acc_index >= len_imb:
        acc_index = len_imb
    sub_H = imb_H[:, index:acc_index]
    sub_mean = round(np.average(sub_H), 3)
    plt.text(acc_index - 120, 280, str(sub_mean), fontsize=6)
    index += step_size

imb_rgb = cv2.cvtColor(imb, cv2.COLOR_BGR2RGB)
imb_normal = (imb_rgb / 255).astype('float64')
imb_hsv_in = rgb2hsv(imb_normal)
imb_H_in, imb_S_in, imb_V_in = cv2.split(imb_hsv_in)
check_for_abseq = imb_H == imb_H_in
check_for_eq = abs(imb_H - imb_H_in) < 0.000001
any_false = np.sum(check_for_eq == 0)
any_true = np.sum(check_for_eq == 1)
print("Compare H channel with built in function, is there any difference?", any_false)
check_for_abseq = imb_S == imb_S_in
check_for_eq = abs(imb_S - imb_S_in) < 0.000001
any_false = np.sum(check_for_eq == 0)
any_true = np.sum(check_for_eq == 1)
print("Compare S channel with built in function, is there any difference?", any_false)
check_for_abseq = imb_V == imb_V_in
check_for_eq = abs(imb_V - imb_V_in) < 0.000001
any_false = np.sum(check_for_eq == 0)
any_true = np.sum(check_for_eq == 1)
print("Compare V channel with built in function, is there any difference?", any_false)
index = 0
while (index < len_imb):
    acc_index = index + step_size
    if acc_index >= len_imb:
        acc_index = len_imb
    sub_H = imb_H_in[:, index:acc_index]
    sub_mean = round(np.average(sub_H), 3)
    plt.text(acc_index - 120, 680, str(sub_mean), fontsize=6)
    index += step_size
plt.title("Calculating the Avg of H Channel")
plt.show()
print("Task3 ends")




# Task4
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




#Task 5

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
print("task5 ends")