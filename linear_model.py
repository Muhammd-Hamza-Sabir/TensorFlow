import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("./data", one_hot=True)

# Data dimension
img_size = 28
num_channel = 1
img_flat = img_size*img_size*num_channel
num_classes = 10
batch_size = 100

# Input placeholder
x = tf.placeholder(tf.float32, shape=[None, img_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])

# Input parameters
weights = tf.Variable(tf.zeros([img_flat, num_classes]))
bias = tf.Variable(tf.zeros([num_classes]))

# logits
logits = tf.add(tf.matmul(x, weights), bias)
y_pred = tf.nn.softmax(logits)

# Loss & Cost
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits)
cost = tf.reduce_mean(cross_entropy)

# optimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

# Accuracy
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Running session
session = tf.Session()
session.run(tf.global_variables_initializer())


def optimize(num_epochs):
    for num_epoch in range(num_epochs):
        for num_step in range(len(data.train.images)//batch_size):
            x_batch, y_true_batch = data.train.next_batch(batch_size=batch_size, shuffle=True)
            feed_dict_train = {x: x_batch, y_true: y_true_batch}
            session.run(optimizer, feed_dict=feed_dict_train)


def weight_plot():
    w = session.run(weights)
    fig, axes = plt.subplots(5, 2)
    fig.subplots_adjust(wspace=0.2, hspace=0.3)

    for index, y in enumerate(axes.flat):
        image = w[:, index].reshape(img_size, img_size)
        y.set_xlabel("Weights: {}".format(index))
        y.imshow(image)

        y.set_xticks([])
        y.set_yticks([])
    plt.show()


feed_dict_test = {x: data.test.images, y_true: data.test.labels}


def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)

    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))


optimize(500)
print_accuracy()
weight_plot()

session.close()
