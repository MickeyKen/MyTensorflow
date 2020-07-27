import tensorflow as tf

x = tf.Variable(0., name='x')
func = (x - 1) ** 2

optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=0.1
)

train_step = optimizer.minimize(func)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for  i in range(20):
        sess.run(train_step)
    print ('x = ', sess.run(x))
