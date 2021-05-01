"""
    @Time    : 2021/2/15 13:00 
    @Author  : fate
    @Site    : 
    @File    : aa22.py
    @Software: PyCharm
"""
import tensorflow as tf
import numpy as np
import cv2
from imageio import imread, imsave
import scipy.io
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


# 加载内容图片，去掉黑白图片，处理成指定大小，暂时不进行归一化，像素值范围为0至255之间
def resize_and_crop(image, image_size):
    h = image.shape[0]
    w = image.shape[1]
    if h > w:
        image = image[h // 2 - w // 2: h // 2 + w // 2, :, :]
    else:
        image = image[:, w // 2 - h // 2: w // 2 + h // 2, :]
    image = cv2.resize(image, (image_size, image_size))
    return image


# 格式化训练图片
image_size = 256
X_data = []
paths = glob.glob('E:\\python\\painting\\trainA\\*.jpg')
for i in tqdm(range(len(paths))):
    path = paths[i]
    image = imread(path)
    if len(image.shape) < 3:
        continue
    X_data.append(resize_and_crop(image, image_size))
X_data = np.array(X_data)
batch_size = 4

# 将X_data打乱, 随机
data_index = np.arange(X_data.shape[0])
np.random.shuffle(data_index)
X_data = X_data[data_index]
for i in tqdm(range(X_data.shape[0] // batch_size)):
    X_batch = X_data[i * batch_size: i * batch_size + batch_size]
    pass
