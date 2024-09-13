import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from skimage.color import rgb2hsv
import math



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
