import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_size = data.train.num_examples
validation_size = data.validation.num_examples
test_size = data.test.num_examples

n_input = 28*28
n_hidden_1 = 512
n_hidden_2 = 256
n_hidden_3 = 128
n_output = 10

learning_rate = 1e-4
n_iterations = 1000
batch_size = 128

# eliminates some units at random
# used only in hidden layer
dropout = 0.5 #prevents overfitting

# x = tf.placeholder(shape=(None, n_input), dtype=tf.float32, name="input")
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])

# controls dropout rate
keep_prob = tf.placeholder(dtype=tf.float32)



# *********************************************************************************
weights = {
    # 'w1': tf.Variable(tf.random_uniform([n_input, n_hidden_1], minval=-0.05, maxval=0.05), dtype=tf.float32)
    # tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev = 0.03)
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1, name = 'w1')),
    'w2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1), name ='w2'),
    'w3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1), name ='w3'),
    'out': tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1))
}
# truncated: values follow normal distribution, except that those values whose magnitude is more than 2 std are dropped and repicked
biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden_1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden_2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden_3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), biases['b1']), name='layer1')
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['w2']), biases['b2']), name='layer2')
layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, weights['w3']), biases['b3']))
layer_drop = tf.nn.dropout(layer3, keep_prob)
output_layer = tf.matmul(layer3, weights['out'])+biases['out']




# ***********************************************************************

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_layer))
# ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits)
# cross_entropy = -tf.reduce_mean(tf.reduce_sum(y*tf.log(y)+ (1-y)*tf.log(1-y), axis=1))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_predictions = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name = 'accuracy')

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(n_iterations):
    batch_x, batch_y = data.train.next_batch(batch_size)
    feed_dict = {x: batch_x,
                 y: batch_y}
    # with tf.Session() as sess
    sess.run(optimizer, feed_dict={x: batch_x,
                                     y: batch_y,
                                     keep_prob: dropout})
    # optimizer.run(feed_dict=feed_dict)
    if i%100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy], feed_dict= feed_dict)

        print('Iteration', str(i), '\t Loss=', str(minibatch_loss), '\t Accuracy', str(minibatch_accuracy))
        # train_accuracy = accuracy.eval(feed_dict=feed_dict)

test_accuracy = sess.run(accuracy, feed_dict={x: data.test.images,
                                                      y: data.test.labels, keep_prob:1.})
est_accuracy = accuracy.eval(feed_dict={x: data.test.images,
                                        y: data.test.labels, keep_prob:1.})

print('Accuracy on test set: ', test_accuracy)









