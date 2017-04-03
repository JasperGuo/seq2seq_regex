# coding=utf8
__author__ = '家琪'


import tensorflow as tf


a = tf.Variable(initial_value=tf.truncated_normal([3, 5, 4]))
b = tf.reverse_sequence(a, seq_axis=1, batch_axis=0, seq_lengths=[4, 4, 4])
c = tf.reverse_sequence(b, seq_axis=1, batch_axis=0, seq_lengths=[1, 2, 3])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(a.eval())
    print("======================")
    print(b.eval())
    print("======================")
    print(c.eval())