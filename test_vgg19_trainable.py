"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf

import vgg19_trainable as vgg19
import utils

batch1 = img1.reshape((1, 64, 64, 3))

with tf.device('/cpu:0'):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [1, 64, 64, 3])
    true_images=tf.placeholder(tf.float32, [1, 64, 64, 3])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19()
    vgg.build(images, train_mode)

    sess.run(tf.global_variables_initializer())

    # simple 1-step training
    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    sess.run(train, feed_dict={images: batch1, true_out: [img1_true_result], train_mode: True})

    # test classification again, should have a higher probability about tiger
    prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    utils.print_prob(prob[0], './synset.txt')

    # test save
    vgg.save_npy(sess, './test-save.npy')
