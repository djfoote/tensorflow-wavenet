import tensorflow as tf


sess = tf.InteractiveSession()
first_var = tf.Variable([1.0], trainable=True)
second_var = tf.Variable([2.0], trainable=True)
c = tf.reshape(tf.concat(0, [first_var, second_var]), (1,2))
trainable = tf.trainable_variables()
print(trainable, len(trainable))
trainable.remove(trainable[0])
loss = tf.nn.l2_loss(c)
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss, var_list = [first_var, second_var])

tf.initialize_all_variables().run()
# first_var.trainable = True
sess.run([train_step])



print(sess.run([c]))