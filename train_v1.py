# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 18:58:37 2018
@author: LHF
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from datetime import datetime
from tensorflow.python.framework import graph_util
import tensorflow as tf
import numpy as np
import model
import time
import loss
import optimize
import sys
import os
import termcolor
import warnings
import random
sys.path.append("..")
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
img_H = 256
img_W = 256
channel = 1
Batch_Size = 16
EPOCH = 10000
Data_Dir = 'dataset/'
train_log_dir = 'train_logs/'
exist_model_dir = 'train_logs/'
train_from_exist = False
pb_file_path = 'model_8.pb'
learning_rate = 1e-4
max_steps = 1000000

def augmentation(input_patch):
    rot = np.random.randint(low = 2, high = 9)
    if rot == 2:    # up-down flip
        Inp_Patch = input_patch[::-1, :, :]
    elif rot == 3:  # left-right flip
        Inp_Patch = input_patch[:, ::-1, :]
    elif rot == 4:  # up-down + left-right flip
        Inp_Patch = input_patch[::-1, :, :]
    elif rot == 5:  # transpose
        Inp_Patch = np.transpose(input_patch, [1, 0, 2])
    elif rot == 6:  # transpose + up-down flip
        Inp_Patch = np.transpose(input_patch, [1, 0, 2])[::-1, :, :]
    elif rot == 7:  # transpose + left-right flip
        Inp_Patch = np.transpose(input_patch, [1, 0, 2])[:, ::-1, :]
    else:           # transpose + up-down + left-right flip
        Inp_Patch = np.transpose(input_patch, [1, 0, 2])[::-1, ::-1, :]
    return Inp_Patch


def Validation(pred_labels, labels):
    #prediction_labels = np.argmax(pred_labels, axis=1)
    #num = len(labels)
    #labels = np.reshape(labels, (num, 1))
    correct_prediction = np.equal(pred_labels, labels)
    accuracy = np.mean(correct_prediction.astype(np.float32))
    correct_times_in_batch = np.sum(correct_prediction.astype(np.int32))
    return accuracy, correct_times_in_batch


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
            data = (np.reshape(data, (256, 256, 1)) + 1) * 128
            label = label - 1
            images.append(data)
            labels.append(label)
    
    train_images = images[:-200]
    train_labels = labels[:-200]
    valid_images = images[-200:]
    valid_labels = labels[-200:]
    return train_images, train_labels, valid_images, valid_labels


def get_train_batch(train_images, train_labels, img_index, batch_size):
    train_img_batch = np.zeros([batch_size, img_H, img_W, channel], dtype = np.float32)
    train_lab_batch = np.zeros([batch_size], dtype = np.int64)
    for i in range(batch_size):
        train_img = augmentation(train_images[img_index[i]])
        train_img_batch[i, :, :, :] = train_img
        train_lab_batch[i] = train_labels[img_index[i]]
    return train_img_batch, train_lab_batch
        
        
def get_valid_batch(valid_images, valid_labels):
    num_valid_examples = len(valid_images)
    valid_img_batch = np.zeros([num_valid_examples, img_H, img_W, channel], dtype = np.float32)
    valid_lab_batch = np.zeros([num_valid_examples], dtype = np.int64)
    for i in range(num_valid_examples):
        valid_img_batch[i, :, :, :] = valid_images[i]
        valid_lab_batch[i] = valid_labels[i]
    return valid_img_batch, valid_lab_batch

    
def restore_model(sess, saver, exist_model_dir, global_step):
    log_info = "Restoring Model From %s..." % exist_model_dir
    print(termcolor.colored(log_info, 'green', attrs = ['bold']))
    ckpt = tf.train.get_checkpoint_state(exist_model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        init_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        sess.run(tf.assign(global_step, init_step))
    else:
        print('No Checkpoint File Found!')
        return
    
    return init_step 


def train_network():
    global_step = tf.train.get_or_create_global_step()
    Img_Batch = tf.placeholder(dtype = tf.float32, shape = [None, img_H, img_W, channel] , name = 'input')
    Lab_Batch = tf.placeholder(dtype = tf.int64, shape = [None])
    Drop = tf.placeholder(dtype = tf.float32)
    print(termcolor.colored("Building Computation Graph...", 'green', attrs = ['bold']))
    
    #pred_model = model.Res_Net(Img_Batch, Drop)
    pred_model = model.My_Model_1(Img_Batch, Drop)
    #pred_model = model_v1.My_Model(Img_Batch, Drop)
    #pred_model = model_v1.My_Model(Img_Batch, Drop)
    #pred_model = model.my_VGG(Img_Batch, Drop)
    finaloutput = tf.nn.softmax(pred_model, name = 'softmax')
    prediction_labels = tf.argmax(finaloutput, axis=1, name = 'output')
    train_loss = loss.loss_cross(pred_model, Lab_Batch)
    
    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops):
    train_op = optimize.optimize(train_loss, learning_rate, global_step)
    
    saver = tf.train.Saver(var_list = tf.global_variables(), max_to_keep = 3)
    
    summ_op = tf.summary.merge_all()
    
    config = tf.ConfigProto()
    config.log_device_placement = True
    config.allow_soft_placement = True
    sess = tf.Session(config = config)
    
    print(termcolor.colored("Defining Summary Writer...", 'green', attrs = ['bold']))
    summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    
    step = 0
    if train_from_exist:
        step = restore_model(sess, saver, exist_model_dir, global_step)
    else:
        print(termcolor.colored("Initializing Variables...", 'green', attrs = ['bold']))
        sess.run(tf.global_variables_initializer())
        
    min_loss = float('Inf')
    max_accuracy = float('-Inf')
    max_ctib = float('-Inf')
    

    train_images, train_labels, valid_images, valid_labels = Generate_Data(Data_Dir)
    print(termcolor.colored("Starting To Train...", 'green', attrs = ['bold']))
    num_images = len(train_images)
    index_img = np.arange(num_images)
    index = np.arange(int(num_images / Batch_Size))
    index = index*Batch_Size
    for i in range(EPOCH):
        random.shuffle(index_img)
        for k in index:
            step += 1
            
            start_time = time.time()
            train_img_batch, train_lab_batch = get_train_batch(train_images, train_labels, index_img[k:k+Batch_Size], Batch_Size)
            duration1 = time.time() - start_time
            
            feed_dict = {Img_Batch:train_img_batch, Lab_Batch:train_lab_batch, Drop:0.5}
            _, model_loss, pred_train_labels = sess.run([train_op, train_loss, prediction_labels], feed_dict = feed_dict)
            duration2 = time.time() - start_time
            accuracy, correct_times_in_batch = Validation(pred_train_labels, train_lab_batch)
            if (step + 1) % 100 == 0:
                examples_per_second = Batch_Size/(duration2 - duration1)
                seconds_per_batch = float(duration2 - duration1)
                seconds_get_per_batch = float(duration1)
                if model_loss < min_loss: min_loss = model_loss
                if accuracy > max_accuracy: max_accuracy = accuracy
                if correct_times_in_batch > max_ctib: max_ctib = correct_times_in_batch
            
                summary = tf.Summary()
                summary.value.add(tag = 'Accuracy', simple_value = accuracy)
                summary.value.add(tag = 'CTIB', simple_value = correct_times_in_batch)
                summary_writer.add_summary(summary, step)
                
                print(termcolor.colored('%s ---- step #%d' % (datetime.now(), step + 1), 'green', attrs = ['bold']))
                print('  LOSS = %.6f\t MIN_LOSS = %.6f' % (model_loss, min_loss))
                print('  ACCY = %.6f\t MAX_ACCY= %.6f' % (accuracy, max_accuracy))
                print('  CTIB = %.6f\t MAX_CTIB = %.6f' % (correct_times_in_batch, max_ctib))
                print('  ' + termcolor.colored('%.6f seconds/geting one batch' % (seconds_get_per_batch), 'blue', attrs = ['bold']))
                print('  ' + termcolor.colored('%.6f images/sec\t%.6f seconds/batch' % (examples_per_second, seconds_per_batch), 'blue', attrs = ['bold']))
                with open("records/train_records.txt", "a") as file:
                   
                    format_str = "%d\t%.6f\t%.6f\t%.6f\t%.6f\n"
                    file.write(str(format_str) % (step + 1, accuracy, correct_times_in_batch, examples_per_second, seconds_per_batch))
                    
            if ((step + 1) % 200 == 0) or ((step + 1) == max_steps):
               summary_str = sess.run(summ_op, feed_dict = feed_dict)
               summary_writer.add_summary(summary_str, step + 1)
                    
            if (step == 0) or ((step + 1) % 1000 == 0) or ((step + 1) == max_steps):
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                print("saving checkpoint into %s-%d" % (checkpoint_path, step + 1))
                saver.save(sess, checkpoint_path, global_step = step + 1)
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
                with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
            if (step == 0) or ((step + 1) % 100 == 0) or ((step + 1) == max_steps):
                
                valid_image_batch, valid_label_batch = get_valid_batch(valid_images, valid_labels)
        
                pred_valid_labels = sess.run(prediction_labels, feed_dict = {Img_Batch:valid_image_batch, Drop:1})
                
                accuracy, correct_times_in_batch = Validation(pred_valid_labels, valid_label_batch)
        
                summary = tf.Summary()
                summary.value.add(tag = 'Accuracy', simple_value = accuracy)
                summary.value.add(tag = 'CTIB', simple_value = correct_times_in_batch)
                summary_writer.add_summary(summary, step)
           
                with open("records/valid_records.txt", "a") as file:
                    format_str = "%d\t%.6f\t%.6f\n"              
                    file.write(str(format_str) % (step + 1, accuracy, correct_times_in_batch))
    #测试结果要加1
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
    with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())
    summary_writer.close()
    sess.close()

    
def main(argv = None):  
    if not train_from_exist:
        if tf.gfile.Exists(train_log_dir):
            tf.gfile.DeleteRecursively(train_log_dir)
        tf.gfile.MakeDirs(train_log_dir)
    else:
        if not tf.gfile.Exists(exist_model_dir):
            raise ValueError("Train from existed model, but the target dir does not exist.")
        
        if not tf.gfile.Exists(train_log_dir):
            tf.gfile.MakeDirs(train_log_dir)
    train_network()


if __name__ == '__main__':
    tf.app.run()
                                                                                                                                                   
