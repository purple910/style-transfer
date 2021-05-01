"""
    @Time    : 2021/3/6 19:16 
    @Author  : fate
    @Site    : 
    @File    : aa5.py
    @Software: PyCharm
"""

# 加载vgg19模型
import scipy.io

vgg = scipy.io.loadmat('./vggnet/imagenet-vgg-verydeep-19.mat')
vgg_layers = vgg['layers']
