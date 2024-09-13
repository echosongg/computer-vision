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