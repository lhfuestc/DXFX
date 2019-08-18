# -*- coding:utf-8 -*-
#from __future__ import absolute_import
#from __future__ import print_function
#from __future__ import division
import sys
import tensorflow as tf 
from scipy import misc
import numpy as np
import subprocess
#import os
import concurrent.futures
#from tensorflow.python.framework import graph_util
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
#from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input#, decode_predictions
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
# from keras.utils import plot_model
pb_file_path_E = 'ppc.pb'
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


def Image_Enhancement(image, idn, labeln):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path_E, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            inputs = sess.graph.get_tensor_by_name("model_input:0")
            out_label = sess.graph.get_tensor_by_name("model_output:0")
           
            single_image = np.reshape(image, (1, 256, 256, 1))
            image_E = sess.run(out_label, feed_dict={inputs:single_image})
            misc.imsave('./image/' + str(idn) + '_' + str(labeln) + 'bm3d.png', image_E[0, :, :, 0])
            


def count(iter):
    try:
        return len(iter)
    except TypeError:
        return sum(1 for _ in iter)


def unpackImage(data):
    example = tf.train.Example()
    example.ParseFromString(data)
    label = example.features.feature['label'].int64_list.value[0]
    data = example.features.feature['data'].float_list.value
    id = example.features.feature['id'].int64_list.value[0]
    data = (np.reshape(data, (256, 256)) + 1) * 0.5
    print('./image/' + str(id) + '_' + str(label) + '.png')
    
    #res = cv2.resize(data, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    misc.imsave('./image/' + str(id) + '_' + str(label) + '.png', data)
    Image_Enhancement(data, id, label)
    #subprocess.call(["./bm3d", './image/' + str(id) + '_' + str(label) + '.png', '23', './image/' + str(id) + '_' +str(label) + 'bm3d.png'])
    # x = np.dstack((res,res,res))
    x = misc.imread('./image/' + str(id) + '_' + str(label) + 'bm3d.png');
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x, label
        
#用预训练的inception_v3对图片进行预测

def prepareImage():
    model = InceptionV3(include_top=False, weights='imagenet')
    for i in range(1, 11):
        #features = []
        #labels = []
        it = tf.python_io.tf_record_iterator('./dataset/TFcodeX_'+ str(i) +'.tfrecord')
        with concurrent.futures.ProcessPoolExecutor() as executor:
            res = zip(*executor.map(unpackImage, it)) 
            res[0] = map(model.predict, res[0])
            np.save(open('./predict/TFcodeX_'+ str(i) +'.npy', 'wb+'), res[0])
            np.save(open('./predict/Labels_'+ str(i) +'.npy', 'wb+'), res[1])


batch_size = 20                                               # 原来20
#对预测的图片进行分类
  
def train(train_data, train_labels, validation_data, validation_labels):
    epochs = 130
    top_model_weights_path = 'bottleneck_fc_model.h5'
    train_labels = train_labels - 1
    validation_labels = validation_labels - 1
    train_labels = to_categorical(train_labels, 5)
    validation_labels = to_categorical(validation_labels, 5)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[tensorboard],
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
    score = model.evaluate(validation_data, validation_labels, verbose=0)
    print('Test loss: %s' %(score[0]))
    print( 'Test accuracy: %s' %(score[1]))


#交叉验证，修改folds

def cross_validation():
    folds = 2
    train_data = []
    train_labels = []
    validation_data = []
    validation_labels = []
    for i in range(1, 11):
        if i != folds:
            train_data.extend(np.load(open('./predict/TFcodeX_'+ str(i) +'.npy', 'rb')))
            train_labels.extend(np.load(open('./predict/Labels_'+ str(i) +'.npy', 'rb')))
        else:
            validation_data.extend(np.load(open('./predict/TFcodeX_'+ str(i) +'.npy', 'rb')))
            validation_labels.extend(np.load(open('./predict/Labels_'+ str(i) +'.npy', 'rb')))
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    validation_data = np.array(validation_data)
    validation_labels = np.array(validation_labels)
    print(np.shape(train_data))
    train(train_data, train_labels, validation_data, validation_labels)


#导出keras网络，通过https://github.com/amir-abdi/keras_to_tensorflow 工具转成tensorflow格式

def save_network():
    input_tensor = Input(shape=(256,256,3))
    top_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_tensor)
    model = Sequential()
    model.add(Flatten(input_shape=top_model.output_shape[1:]))
    #model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    model.load_weights('bottleneck_fc_model.h5')
    new_model = Model(inputs= top_model.input, outputs= model(top_model.output))
    new_model.save('model.h5')
    # plot_model(new_model, to_file='model.png')


#　预测测试集
def predict(testset):
    model = load_model('model.h5')
    labels = []
    it = tf.python_io.tf_record_iterator(testset)
    for data in it:
        example = tf.train.Example()
        example.ParseFromString(data)
        label = example.features.feature['label'].int64_list.value[0]
        data = example.features.feature['data'].float_list.value
        id = example.features.feature['id'].int64_list.value[0]
        data = (np.reshape(data, (256, 256)) + 1) * 128
        res = cv2.resize(data, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./image/' + str(id) + '_' + str(label) + '.png', res)
        subprocess.call(["./bm3d", './image/' + str(id) + '_' + str(label) + '.png', '23', './image/' + str(id) + '_' +       str(label) + 'bm3d.png'])
        # x = np.dstack((res,res,res))
        x = cv2.imread('./image/' + str(id) + '_' + str(label) + 'bm3d.png');
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = model.predict(x)
        print('./image/' + str(id) + '_' + str(label) + '.png')
        print(x.argmax(axis=-1)+1)
        labels.append(x.argmax(axis=-1)+1)
    return labels


def main():
    try:
        prepareImage()
        cross_validation()
        save_network()
        #labels = predict('./dataset/TFcodeX_5.tfrecord')
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
   
# 准确率： 在dataset: test loss : 1.39    Test accuracy:0.83
# 		   在data:    test loss : 1.068   Test accuracy:0.911
