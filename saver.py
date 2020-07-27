import tensorflow as tf

a = tf.Variable(1, name='a')
b = tf.assign(a, a+1)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print ('First time b = ', sess.run(b))
    print ('Second time b = ', sess.run(b))
    saver.save(sess, 'model/model.ckpt')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_path='model/model.ckpt')
    print ('First time b = ', sess.run(b))
    print ('Second time b = ', sess.run(b))
