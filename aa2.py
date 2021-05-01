"""
    @Time    : 2021/2/15 11:14 
    @Author  : fate
    @Site    : https://www.cnblogs.com/xiaochouk/p/8685909.html
    @File    : aa2.py
    @Software: PyCharm
"""
import os

import numpy as np
import tensorflow as tf

from generateds import get_content_tfrecord

content_batch = get_content_tfrecord(4, os.path.join('data', 'painting_trainA.tfrecords'), 256)

# 初始化全局变量
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    batch_content = sess.run(content_batch)
    batch_content = np.reshape(batch_content, [4, 256, 256, 3])

    # 关闭多线程
    coord.request_stop()
    coord.join(threads)

