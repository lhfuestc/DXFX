# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 14:18:51 2018

@author: zhu
"""

import tensorflow as tf
import matplotlib.pyplot as plt

def read_and_decode_single_example(filenames):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
    
    reader = tf.TFRecordReader()
    
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'data': tf.FixedLenFeature([], tf.float32),
            'label': tf.FixedLenFeature([], tf.string)
        })
    
    # now return the converted data
    label = features['data']
    image = tf.decode_raw(features['label'],tf.float32)
    image = tf.reshape(image, [256, 256, 1])
    
    return label, image
    
label, image = read_and_decode_single_example(['TFcodeX_1.tfrecord'])
images_batch, labels_batch = tf.train.batch([image, label], batch_size=8, capacity=32)
global_step = tf.Variable(0, trainable=False)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    la_b, im_b = sess.run([labels_batch, images_batch])
    #print(im_b.get_shape()[-1].value )
    print(im_b.shape)
    for j in range(3):
        la_b, im_b = sess.run([labels_batch, images_batch])
        
        #
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)