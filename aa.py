"""
    @Time    : 2021/2/14 20:12 
    @Author  : fate
    @Site    : 
    @File    : aa.py
    @Software: PyCharm
"""
import tensorflow as tf
import numpy as np

aa = tf.random_uniform(shape=[2, 2], dtype=tf.int32, maxval=100)
print(aa)

bb = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)
print(bb)

cc = aa + bb
print(cc)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     aa1 = sess.run([aa])
#     print(aa1)
#     aa1 = sess.run([aa])
#     print(aa1)
#     bb1 = sess.run([bb])
#     print(bb1)
#     在执行cc = aa + bb时,aa重新创建的,与上面的aa不一样
#     cc1 = sess.run([cc])
#     print(cc1)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # []:用于共享同一随机值
    print(sess.run([aa, bb, cc]))

