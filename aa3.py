"""
    @Time    : 2021/2/16 11:03 
    @Author  : fate
    @Site    : 
    @File    : aa3.py
    @Software: PyCharm
"""
import zipfile
from imageio import imread, imsave
import cv2
from tqdm import tqdm

zip_file = zipfile.ZipFile('E:\\python\\train2014.zip')
zip_list = zip_file.namelist()  # 得到压缩包里所有文件
zip_list = zip_list[1:]


def resize_and_crop(image, image_size):
    h = image.shape[0]
    w = image.shape[1]
    if h > w:
        image = image[h // 2 - w // 2: h // 2 + w // 2, :, :]
    else:
        image = image[:, w // 2 - h // 2: w // 2 + h // 2, :]
    image = cv2.resize(image, (image_size, image_size))
    return image


X_data = []
for i in tqdm(range(len(zip_list))):
    path = zip_list[i]
    image = imread(path)
    if len(image.shape) < 3:
        continue
    X_data.append(resize_and_crop(image, 256))
print(X_data.shape)
