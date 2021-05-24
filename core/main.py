"""
    @Time    : 2021/4/8 20:09 
    @Author  : fate
    @Site    : 
    @File    : main.py
    @Software: PyCharm
"""
import argparse  # 导入参数选择模块
import os

import numpy as np
import tensorflow as tf
from imageio import imread, imsave

import four
import one
import two
import customize
from vgg16 import main


def vgg16_style_transfer(content, model, result):
    FLAGS = tf.app.flags.FLAGS
    lst = list(FLAGS._flags().keys())
    for key in lst:
        FLAGS.__delattr__(key)
    tf.app.flags.DEFINE_string('loss_model', 'vgg_16', '')
    tf.app.flags.DEFINE_integer('image_size', 256, 'I')
    tf.app.flags.DEFINE_string("model_file", model, "")
    tf.app.flags.DEFINE_string("image_file", content, "")
    tf.app.flags.DEFINE_string("out_file", result, "")
    FLAGS = tf.app.flags.FLAGS
    main(FLAGS)


def vgg19_style_transfer(content, model, result):
    X_image = imread(content)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph(os.path.join(model, 'fast_style_transfer.meta'))
    saver.restore(sess, tf.train.latest_checkpoint(model))
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name('X:0')
    g = graph.get_tensor_by_name('transformer/g:0')
    gen_img = sess.run(g, feed_dict={X: [X_image]})[0]
    gen_img = np.clip(gen_img, 0, 255) / 255.
    imsave(result, gen_img)


def one_style_transfer(content, model, result, style):
    # 定义一个参数设置器
    parser = argparse.ArgumentParser()
    # 修改以下5个参数以开启训练
    parser.add_argument("--PATH_IMG", type=str, default=content)  # 参数：选择测试图像
    parser.add_argument("--LABEL_1", type=int, default=style[0])  # 参数：风格1
    # 固定参数
    parser.add_argument("--LABELS_NUMS", type=int, default=26)  # 参数：风格数量
    parser.add_argument("--PATH_MODEL", type=str, default=model)  # 参数：模型存储路径
    parser.add_argument("--PATH_RESULTS", type=str, default=result)  # 参数：测试结果存储路径
    parser.add_argument("--PATH_STYLE", type=str, default="static/img/style/png/")  # 参数：风格图片路径
    parser.add_argument("--ALPHA1", type=float, default=0.6)  # 参数：Alpha1，风格权重，默认为1
    args = parser.parse_args()  # 定义参数集合args

    one.get_image_matrix(args)


def two_style_transfer(content, model, result, style):
    parser = argparse.ArgumentParser()  # 定义一个参数设置器
    # 修改以下5个参数以开启训练
    parser.add_argument("--PATH_IMG", type=str, default=content)  # 参数：选择测试图像
    parser.add_argument("--LABEL_1", type=int, default=style[0])  # 参数：风格1
    parser.add_argument("--LABEL_2", type=int, default=style[1])  # 参数：风格2
    # 固定参数
    parser.add_argument("--LABELS_NUMS", type=int, default=26)  # 参数：风格数量
    parser.add_argument("--PATH_MODEL", type=str, default=model)  # 参数：模型存储路径
    parser.add_argument("--PATH_RESULTS", type=str, default=result)  # 参数：测试结果存储路径
    parser.add_argument("--PATH_STYLE", type=str, default="static/img/style/png/")  # 参数：风格图片路径
    parser.add_argument("--ALPHA1", type=float, default=0.5)  # 参数：Alpha1，风格权重，默认为0.25
    parser.add_argument("--ALPHA2", type=float, default=0.5)  # 参数：Alpha2，风格权重，默认为0.25
    args = parser.parse_args()  # 定义参数集合args
    two.get_image_matrix(args)


def four_style_transfer(content, model, result, style):
    # 设置参数
    parser = argparse.ArgumentParser()  # 定义一个参数设置器
    # 修改以下5个参数以开启训练
    parser.add_argument("--PATH_IMG", type=str, default=content)  # 参数：选择测试图像
    parser.add_argument("--LABEL_1", type=int, default=style[0])  # 参数：风格1
    parser.add_argument("--LABEL_2", type=int, default=style[1])  # 参数：风格2
    parser.add_argument("--LABEL_3", type=int, default=style[2])  # 参数：风格3
    parser.add_argument("--LABEL_4", type=int, default=style[3])  # 参数：风格4
    parser.add_argument("--LABELS_NUMS", type=int, default=26)  # 参数：风格数量
    parser.add_argument("--PATH_MODEL", type=str, default=model)  # 参数：模型存储路径
    parser.add_argument("--PATH_RESULTS", type=str, default=result)  # 参数：测试结果存储路径
    parser.add_argument("--PATH_STYLE", type=str, default="static/img/style/png/")  # 参数：风格图片路径
    parser.add_argument("--ALPHA1", type=float, default=0.25)  # 参数：Alpha1，风格权重，默认为0.25
    parser.add_argument("--ALPHA2", type=float, default=0.25)  # 参数：Alpha2，风格权重，默认为0.25
    parser.add_argument("--ALPHA3", type=float, default=0.25)  # 参数：Alpha3，风格权重，默认为0.25
    parser.add_argument("--ALPHA4", type=float, default=0.25)  # 参数：Alpha3，风格权重，默认为0.25
    args = parser.parse_args()  # 定义参数集合args
    four.get_image_matrix(args)


def custom_style_transfer(content, result, style, weights):
    # 设置参数
    parser = argparse.ArgumentParser()  # 定义一个参数设置器
    # 修改以下5个参数以开启训练
    parser.add_argument("--PATH_IMG", type=str, default=content)  # 参数：选择测试图像
    parser.add_argument("--LABEL", type=int, default=style)  # 参数：风格1
    parser.add_argument("--LABELS_NUMS", type=int, default=26)  # 参数：风格数量
    parser.add_argument("--PATH_MODEL", type=str, default='core/models/paintA_models/')  # 参数：模型存储路径
    parser.add_argument("--PATH_RESULTS", type=str, default=result)  # 参数：测试结果存储路径
    parser.add_argument("--PATH_STYLE", type=str, default="static/img/style/png/")  # 参数：风格图片路径
    parser.add_argument("--ALPHA", type=float, default=weights)  # 参数：Alpha1，风格权重，默认为0.25
    args = parser.parse_args()  # 定义参数集合args
    customize.get_image_matrix(args)


def fusion_style_transfer(content, result, style):
    if len(style) == 1:
        one_style_transfer(content, 'core/models/paintA_models/', result, style)
    elif len(style) == 2:
        two_style_transfer(content, 'core/models/paintA_models/', result, style)
    elif len(style) == 4:
        four_style_transfer(content, 'core/models/paintA_models/', result, style)


if __name__ == '__main__':
    #
    # vgg16_style_transfer('content/stu.jpg', "models/vgg16_scream/fast_style_transfer.ckpt-done",
    #                      'result/2.jpg')
    # vgg19_style_transfer('../static/img/content/one.jpg', 'models/vgg19_paintA_cubist', 'result/vgg19.jpg')
    one_style_transfer('content/stu.jpg', './models/paintA_models/', './result/', [10])
    # two_style_transfer('content/stu.jpg', './models/paintA_models/', 'result/', [5])
    # four_style_transfer('content/stu.jpg', './models/paintA_models/', 'result/', [0, 1, 2, 3])

    pass
