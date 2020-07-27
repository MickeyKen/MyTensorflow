import tensorflow as tf

LOG_DIR = './logs'

a=tf.constant(1, name='a')
b=tf.constant(1, name='b')

c = a+b


graph = tf.compat.v1.get_default_graph()
with tf.compat.v1.summary.FileWriter(LOG_DIR) as writer:
    writer.add_graph(graph)
