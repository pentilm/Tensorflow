#############################################################
# recover piecewise constant function
# Author: Zhiwei Fang
# Copyright reserved
#############################################################
import tensorflow as tf
import numpy as np


xs = np.linspace(0,5,1000).reshape(-1,1)
ys = np.piecewise(xs,
                  [(xs>0)&(xs<=1),(xs>1) & (xs<=2),(xs>2)&(xs<=3),(xs>3)&(xs<=4),(xs>4)&(xs<=5)],
                  [0.8147, 0.9058, 0.1270, 0.9134, 0.6324])
c = tf.Variable([0.,0.,0.,0.,0.],dtype=tf.float32)
x_tf = tf.placeholder(tf.float32, shape=[None,1])
y_tf = tf.placeholder(tf.float32, shape=[None,1])
r = 100.
p = [0.,1.,2.,3.,4.,5.]
f = c[0]*tf.sigmoid(r*(x_tf-p[0])) - c[-1]*tf.sigmoid(r*(x_tf-p[-1]))
for i in [1,2,3,4]:
    f += (c[i]-c[i-1])*tf.sigmoid(r*(x_tf-p[i]))

loss = tf.reduce_sum(tf.square(f-y_tf))
train_step = tf.train.AdamOptimizer().minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(20000):
    if i % 1000 ==0:
        print(i)
    sess.run(train_step, feed_dict={x_tf:xs,y_tf:ys})
print(sess.run(loss,feed_dict={x_tf:xs,y_tf:ys}))
print(sess.run(c))
