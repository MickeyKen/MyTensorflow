import tensorflow as tf

a = tf.Variable(1, name='a')
b = tf.assign(a, a+1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print ('First time b = ', sess.run(b))
    print ('Second time b = ', sess.run(b))
