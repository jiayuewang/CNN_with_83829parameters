import os
import pickle
import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime

"""
Configuration Part.
"""

# Path to the textfiles for the trainings and validation set
train_file = './train.txt'
valid_file = './valid.txt'

# Learning params
learning_rate = 0.001
batch_size = 128

# Network params
num_epochs = 2
train_steps = 2000
dropout_rate = 0.5
num_classes = 1000
error_tolerance = 0.44
decrease_rate = 0.01
sparsity = dict()
weights = dict()

# How often we want to write the tf.summary data to disk
display_step = 200

# Path for tf.summary.FileWriter and to store model checkpoints
# filewriter_path = "./log"
# checkpoint_path = "./model"

"""
Main Part of the Script.
"""

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):# cpu
    tr_data = ImageDataGenerator(train_file,#创建generator
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(valid_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    train_iterator = tf.data.Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    train_next_batch = train_iterator.get_next()

    valid_iterator = tf.data.Iterator.from_structure(val_data.data.output_types,
                                       val_data.data.output_shapes)
    valid_next_batch = valid_iterator.get_next()


# Ops for initializing the two different iterators
training_init_op = train_iterator.make_initializer(tr_data.data)
validation_init_op = valid_iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes)

# Load the pretrained weights into the non-trainable layer
load_op = model.load_initial_weights_ops()

# Link variable to model output
logits = model.fc8

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

var_list = [v for v in tf.trainable_variables()]
for var in var_list:
    if 'weights' in var.name.split('/')[1]:
        sparsity[var.name.split('/')[0]] = 0.80
        weights[var.name.split('/')[0]] = var
# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    # gradients = tf.gradients(loss, var_list)
    # gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    # train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
# for gradient, var in gradients:
#     tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
# for var in var_list:
    # tf.summary.histogram(var.name, var)

# Add the loss to summary
# tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.name_scope("error_rate"):
    error_rate = 1. - accuracy

# Add the accuracy and error rate to the summary
# tf.summary.scalar('accuracy', accuracy)
# tf.summary.scalar('error_rate', error_rate)

# Merge all summaries together
# merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
# train_writer = tf.summary.FileWriter(filewriter_path + '/train')
# val_writer = tf.summary.FileWriter(filewriter_path + '/valid')

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = tr_data.data_size // batch_size
val_batches_per_epoch = val_data.data_size // batch_size

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    sess.run(load_op)

    f = open('./log/log_determine_sparsity.txt', 'w')

    def log_print(s):
        print(s)
        f.write(s + '\n')

    def get_threshold(sess, weight, sparsity):
        if sparsity > 1. or sparsity < 0.:
            raise ValueError('Sparsity out of range')
        if sparsity == 1.:
            return np.inf
        if sparsity == 0.:
            return -np.inf
        weight_arr = sess.run(weight)
        flat = np.reshape(weight_arr, [-1])
        flat_list = sorted(map(abs,flat))
        return flat_list[int(len(flat_list) * sparsity)]

    def prune(sess, weight, threshold):
        weight_arr = sess.run(weight)
        under_threshold = abs(weight_arr) < threshold
        weight_arr[under_threshold] = 0
        return weight_arr

    def get_error():
        sess.run(validation_init_op)
        test_acc = 0.
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(valid_next_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.})
            test_acc += acc
        test_acc /= val_batches_per_epoch
        return 1. - test_acc
    
    sess.run(training_init_op)

    for layer_name in sorted(list(sparsity.keys()), reverse=True):
        log_print('Start pruning Layer: %s' % layer_name)
        threshold = get_threshold(sess, weights[layer_name], sparsity[layer_name])
        sparse_weight = prune(sess, weights[layer_name], threshold)
        weights[layer_name].load(sparse_weight)

        t = 0
        error_before_retrain = get_error()
        log_print('Iter %d, error rate %g' % (t, error_before_retrain))

        while error_before_retrain > error_tolerance:
            for _ in range(1, train_steps + 1):
                try:
                    img_batch, label_batch = sess.run(train_next_batch)
                    sess.run(train_op, feed_dict={x: img_batch, y: label_batch, keep_prob: dropout_rate})
                except:
                    sess.run(training_init_op)
                    img_batch, label_batch = sess.run(train_next_batch)
                    sess.run(train_op, feed_dict={x: img_batch, y: label_batch, keep_prob: dropout_rate})

                if _ % display_step == 0:
                    train_acc = sess.run(accuracy, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.0})
                    log_print("Re-train step %d, error rate %g" % (_, 1. - train_acc))
            
            threshold = get_threshold(sess, weights[layer_name], sparsity[layer_name])
            sparse_weight = prune(sess, weights[layer_name], threshold)
            weights[layer_name].load(sparse_weight)

            t += 1
            error_after_retrain = get_error()

            log_print('Iter %d, error rate %g' % (t, error_after_retrain))

            if error_after_retrain <= error_before_retrain:
                log_print('Current error rate %g is less than previous error rate %g, dscrease sparsity by 1%%' % (error_after_retrain, error_before_retrain))
                if sparsity[layer_name] > 0:
                    sparsity[layer_name] -= decrease_rate
                    if sparsity[layer_name] == 0:
                        break

            error_before_retrain = error_after_retrain

        log_print('Achieved sparsity in Layer %s is %g' % (layer_name, sparsity[layer_name]))

    with open('achieved_sparsity.pickle', 'wb') as f_s:
        pickle.dump(sparsity, f_s)

    log_print('Done')

    for layer_name in sparsity.keys():
        log_print('Achieved sparsity in Layer %s is %g' % (layer_name, sparsity[layer_name]))
    f.close()
