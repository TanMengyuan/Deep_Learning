import tensorflow as tf

a = tf.Variable([10, 222, 442], dtype=tf.float32)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print(tf.log(a))