from __future__ import absolute_import, division, print_function
"""
this file contains helper code for setting keras parameters
"""

import numpy as np
from tensorflow import keras as keras
from keras import backend as K
import os

import tensorflow as tf
#import keras.backend.tensorflow_backend as KTF
import math

def get_model_memory_usage(batch_size, model):

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

def set_keras_parms(threads=0, gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if gpu_fraction < 0.1:
        gpu_fraction = 0.1
    #print(f"setting keras params based on model size: {gpu_fraction}")
    num_threads = threads #os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    #tf.device('/device:GPU:0'):
    if num_threads:
        session = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.set_session(session)

def set_keras_growth(gpunumber=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpunumber)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.set_random_seed(1)
    sess = tf.Session(config=config)

    K.set_session(sess)


def manhattan_distance(A,B):
   return K.sum(K.abs( A-B),axis=1,keepdims=True)

# if network output is softmax, the total output of the net is 2 x sum(softmax), i.e. 1, see page 4 of chopra2005learning
upper_b = 2
def my_func(x):
  # x will be a numpy array with the contents of the placeholder below
  print(x.shape[1])
# input = tf.placeholder(tf.float32)


# from https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


# y_pred is a vector of vector of concated outputs from G(x), these vectors
# needs to be unconcated and calculate the L1 norm between each of them
# Then this needs to be combined using eq 9 of chopra2005 
def chopraloss(y_pred,y_true):
    #g_shape = tf.shape(y_pred)[1]
    #g_length = tf.constant(tf.round(tf.divide(g_shape,tf.constant(2))))
    tf.Print(y_pred,[tf.shape(y_pred)],"shape = ")
    g_one, g_two = tf.split(y_pred,2,axis=0)
    #g_one = y_pred[:,0:g_length].eval()
    # g_two = y_pred[:,g_length+1:g_length*2].eval()
    #E_w = || G_w(X_1) - G_w(X_2) ||
    ml = tf.norm(tf.subtract(g_one,g_two),ord=1)
    #ml = manhattan_distance(y_pred,y_true)
    y = y_true
    l_g = 1
    l_l = 1
    q = upper_b
    thisshape = tf.shape(y_pred)#.shape
    ones = tf.ones(thisshape,tf.float32)
    negy = (ones-y)
    return (y*(2.0/q)*(ml**2))+(negy*2*q*tf.exp(-(2.77/q)*ml))

# y_pred is a vector of vector of concated outputs from G(x), these vectors
# needs to be unconcated and calculate the L1 norm between each of them
# Then this needs to be combined using eq 9 of chopra2005 
def chopraloss3(y_pred,y_true):
    #g_shape = tf.shape(y_pred)[1]
    #g_length = tf.constant(tf.round(tf.divide(g_shape,tf.constant(2))))
    tf.Print(y_pred,[tf.shape(y_pred)],"shape = ")
    g_one, g_two = tf.split(y_pred,2,axis=0)
    #g_one = y_pred[:,0:g_length].eval()
    # g_two = y_pred[:,g_length+1:g_length*2].eval()
    #E_w = || G_w(X_1) - G_w(X_2) ||
    ml = K.abs(g_one-g_two)
    #ml = manhattan_distance(y_pred,y_true)
    y = K.round(ml)
    l_g = 1
    l_l = 1
    q = upper_b
    thisshape = K.get_variable_shape(y_pred)#.shape
    ones = K.ones_like(g_one)
    #negy = (ones-y)
    part_one = ((ones-y)*(2.0/q)*K.square(ml))
    part_two = (y*2*q*K.exp(-(2.77/q)*ml))
    return part_one + part_two

def chopraloss2(y_pred,y_true):
    l_g = 1
    l_l = 1
    q = upper_b
    total = 0
    for i in zip(y_pred, y_true):
        i_ypred = i[0]
        i_ytrue = i[1]
        d = manhattan_distance(i_ytrue,i_ypred)
        r = d/double(len(i_ypred))
        y = 1.0
        if r < 0.5:
            y = 0
        else:
            y = 1
        total += ((1-y)*(2.0/q)*(e_w**2))+(y*q*tf.exp(-(2.77/q)*e_w))

    return total/double(y_pred.shape[0])

def keras_sqrt_diff(tensors):
    t1 = tensors[0]
    t2 = tensors[1]
    #for i in range(1, len(t1)):
    #    s += X[i]
    #s = K.sqrt(K.square(s) + 1e-7)
    return K.abs(t1 - t2)


