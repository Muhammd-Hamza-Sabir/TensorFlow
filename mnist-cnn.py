import tensorflow as tf


def cnn_model(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=5, padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=5, padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {"Class": tf.argmax(logits, axis=1),
                   "Probabilities": tf.nn.softmax(logits=logits, name="softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {"Accuracy": tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=predictions["Class"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    from tensorflow.examples.tutorials.mnist import input_data
    data = input_data.read_data_sets("./data", one_hot=True)
    x_train = data.train.images
    y_train = data.train.labels
    x_eval = data.validation.images
    y_eval = data.validation.labels

    classifier = tf.estimator.Estimator(model_fn=cnn_model, model_dir='./model')

    tensor_to_hook = {"Probabilities": "softmax_tensor"}
    logging_tesnor = tf.train.LoggingTensorHook(tensors=tensor_to_hook, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_train}, y=y_train,
                                                        batch_size=100, num_epochs=30, shuffle=True)
    classifier.train(input_fn=train_input_fn, steps=len(x_train)//100, hooks=[logging_tesnor])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_eval}, y=y_eval, num_epochs=1, shuffle=False)
    eval_acc = classifier.evaluate(input_fn=eval_input_fn)
    return eval_acc


result = main()
print("Accuracy: ", result)
