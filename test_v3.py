# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 14:50:26 2018

@author: LHF
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import tensorflow as tf
import  numpy as np
pb_file_path = 'model_8.pb'
img_H = 256 
img_W = 256
channel = 1
batch_size = 1
test_data_dir = 'test'
import os
def Generate_Data(data_dir):
    images= []
    labels = []
    file_names_list = os.listdir(data_dir)
    num = len(file_names_list)
    for i in range(num):
        path = os.path.join(data_dir, file_names_list[i])
        TFR = tf.python_io.tf_record_iterator(path)
        for data in TFR:
            example = tf.train.Example()
            example.ParseFromString(data)
            data = example.features.feature['data'].float_list.value
            label = example.features.feature['label'].int64_list.value[0]
            data = np.reshape(data, (256, 256, 1))
            label = label - 1
            images.append(data)
            labels.append(label)
    return images, labels


def Validation(pred_labels, labels):
    correct_prediction = np.equal(pred_labels, labels)
    accuracy = np.mean(correct_prediction.astype(np.float32))
    correct_times_in_batch = np.sum(correct_prediction.astype(np.int32))
    return accuracy, correct_times_in_batch


def get_test_batch(test_images, test_labels):
    num_test_examples = len(test_images)
    test_img_batch = np.zeros([num_test_examples, img_H, img_W, channel], dtype = np.float32)
    test_lab_batch = np.zeros([num_test_examples], dtype = np.int64)
    for i in range(num_test_examples):
        test_img_batch[i, :, :, :] = test_images[i]
        test_lab_batch[i] = test_labels[i]
    return test_img_batch, test_lab_batch


def Image_Enhancement(image_list):
    num_image = len(image_list)
    img_E = []
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path_E, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            inputs = sess.graph.get_tensor_by_name("model_input:0")
            out_label = sess.graph.get_tensor_by_name("model_outout:0")
            for i in range(num_image):
                single_image = image_list[i]
                image_E = sess.run(out_label, feed_dict={inputs:single_image})
                img_E.append(image_E)
    return img_E


def model_test(TFR):
    images_list, labels_list = Generate_Data(TFR)
    images_list = Image_Enhancement(images_list)
    test_img_batch, test_lab_batch = get_test_batch(images_list, labels_list)
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            inputs = sess.graph.get_tensor_by_name("input:0")
            out_label = sess.graph.get_tensor_by_name("output:0")
            pred_labels = sess.run(out_label, feed_dict={inputs:test_img_batch})
            pred_labels = pred_labels + 1
            test_lab_batch = test_lab_batch + 1
    return pred_labels, test_lab_batch


def main(argv = None):
    pred_labels, true_labels = model_test(test_data_dir)
    ACC, CTIB = Validation(pred_labels, true_labels)
    print(ACC, CTIB)


if __name__ == '__main__':
    tf.app.run()
    
    
    
    
 