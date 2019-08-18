# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 14:50:26 2018

@author: 64360
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import tensorflow as tf
import  numpy as np
pb_file_path = 'model_4.pb'
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
            data = np.reshape(data, (1, 256, 256, 1)).astype(np.float32)
            label = label - 1
            images.append(data)
            labels.append(label)
    return images, labels


def Validation(pred_labels, labels):
    #prediction_labels = np.argmax(pred_labels, axis=1)
    #num = len(labels)
    #labels = np.reshape(labels, (num, 1))
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


def model_test(TFR):
    images_list, labels_list = Generate_Data(TFR)
    #test_img_batch, test_lab_batch = get_test_batch(images_list, labels_list)
    labels_list = np.array(labels_list, dtype = np.int64)
    labels_list = labels_list + 1
    num = len(images_list)
    pred_labels_list = []
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
            for i in range(num):
                test_img_batch = images_list[i]
                pred_label = sess.run(out_label, feed_dict={inputs:test_img_batch})
                pred_label = pred_label + 1
                pred_labels_list.append(pred_label[0])
    return pred_labels_list, labels_list


def main(argv = None):
    pred_labels, true_labels = model_test(test_data_dir)
    return pred_labels, true_labels

pred_labels, true_labels  = main()

ACC, CTIB = Validation(pred_labels, true_labels)
print(ACC, CTIB)

if __name__ == '__main__':
    tf.app.run()
    
    
    
    
 