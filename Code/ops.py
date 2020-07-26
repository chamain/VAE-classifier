# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 14:39:45 2020
Operations for AE classifier
Most of the methods are copied from https://github.com/fab-jul/imgcomp-cvpr
@author: Lahiru D. Chamain
"""


import tensorflow as tf
import tensorflow.contrib as tf_contrib
from utils import denormalize,clip_to_image_range

weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)

def _weight_variable(name, shape):
    return tf.get_variable(name, shape, tf.float32, tf.contrib.layers.xavier_initializer())

def _bias_variable(name, shape):
    return tf.get_variable(name, shape,tf.float32, tf.zeros_initializer())

##################################################################################
# Layers
##################################################################################

def conv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding)
    return x 
def conv2d_transpose(x, channels, kernel=3, stride=2, scope='from_bn',activation_fn='relu'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d_transpose(
                                    inputs=x, filters=channels, kernel_size =kernel , strides=(stride, stride), padding='same',
                                    data_format='channels_last', activation=activation_fn, use_bias=True,
                                    kernel_initializer=None, bias_initializer=tf.zeros_initializer(),
                                    kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                    kernel_constraint=None, bias_constraint=None, trainable=True, name=None,
                                    reuse=None
                                )


    return x


def conv3d(x, channels, kernel=[3,3,3], stride=[1,1,1], padding='SAME', use_bias=True, scope='conv3d_0'):
    with tf.variable_scope(scope):
        x = tf.layers.Conv3D( filters =channels ,
                                   kernel_size = kernel,
                                   strides=stride,
                                   padding=padding,
                                   data_format=None, ## assumes channel last
                                   dilation_rate=(1, 1, 1),
                                   activation=None,
                                   use_bias=use_bias,
                                   kernel_initializer=weight_init,
                                   bias_initializer='zeros',
                                   kernel_regularizer=weight_regularizer,
                                   bias_regularizer=None,
                                   activity_regularizer=None,
                                   kernel_constraint=None,
                                   bias_constraint=None)

        return x

def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='resblock') :
    with tf.variable_scope(scope) :

        x = batch_norm(x_init, is_training, scope='batch_norm_0')
        x = relu(x)


        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            x_init = conv(x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')

        x = batch_norm(x, is_training, scope='batch_norm_1')
        x = relu(x)
        x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')



        return x + x_init

def bottle_resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='bottle_resblock') :
    with tf.variable_scope(scope) :
        x = batch_norm(x_init, is_training, scope='batch_norm_1x1_front')
        shortcut = relu(x)

        x = conv(shortcut, channels, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_front')
        x = batch_norm(x, is_training, scope='batch_norm_3x3')
        x = relu(x)

        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels*4, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels * 4, kernel=1, stride=1, use_bias=use_bias, scope='conv_init')

        x = batch_norm(x, is_training, scope='batch_norm_1x1_back')
        x = relu(x)
        x = conv(x, channels*4, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_back')

        return x + shortcut



def get_residual_layer(res_n) :
    x = []

    if res_n == 18 :
        x = [2, 2, 2, 2]

    if res_n == 34 :
        x = [3, 4, 6, 3]

    if res_n == 50 :
        x = [3, 4, 6, 3]

    if res_n == 101 :
        x = [3, 4, 23, 3]

    if res_n == 152 :
        x = [3, 8, 36, 3]

    return x



##################################################################################
# Sampling
##################################################################################

def flatten(x) :
    return tf.layers.flatten(x)

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

def max_pooling(x,kernel,stride,name='name'):
    return tf.nn.max_pool(x,[kernel,kernel],[stride,stride],'SAME', name='name')


def avg_pooling(x) :
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')

##################################################################################
# Activation function
##################################################################################


def relu(x):
    return tf.nn.relu(x)

def sigmoid(x):
    return tf.nn.sigmoid(x)


##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

##################################################################################
# Loss function
##################################################################################

def classification_loss(logit, label) :
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit))
    prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return loss, accuracy

def entropyLoss(bc,H_target):
        
        H_soft = tf.reduce_mean(bc, name='H_softl')

        H_target = tf.constant(H_target, tf.float32, name='H_target')
        
        return tf.maximum(H_soft - H_target, 0)

def reconMSE(x,x_hat,normalized):
    if(normalized):
        denormalizedX = clip_to_image_range(denormalize(x))
    return tf.losses.mean_squared_error(denormalizedX/255.0,x_hat/255.0)
