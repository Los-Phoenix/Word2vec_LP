#coding:utf-8
#这里是tf的玩具场

import tensorflow as tf
import numpy as np
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print sess.run(hello)

a = tf.constant(10)
b = tf.constant(32)
print sess.run(a + b)

sess.close()

