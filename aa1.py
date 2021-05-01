"""
    @Time    : 2021/2/14 20:26 
    @Author  : fate
    @Site    : 
    @File    : aa1.py
    @Software: PyCharm
"""
import tensorflow as tf

state = tf.Variable(0.0, dtype=tf.float32)
print(state)
one = tf.constant(1.0, dtype=tf.float32)
print(one)
# state + one
new_val = tf.add(state, one)
print(new_val)
# 返回tensor， 值为new_val #update 1 # state 1
# 将new_val的值赋予state
update = tf.assign(state, new_val)
print(update)
# 没有fetch，便没有执行 # update2 10000 # state 10000
update2 = tf.assign(state, 10000)
print(update2)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    print(sess.run(new_val))
    print(sess.run(update))
    print(sess.run(state))
    print(sess.run(new_val))
    # print(sess.run(update2))
    print(sess.run([state, new_val]))
    print(sess.run(state))
    print(sess.run(new_val))
    print(sess.run(update))
    print(sess.run(update2))
