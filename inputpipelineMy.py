# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:21:08 2020

@author: Lahiru D. Chamain
"""
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow.compat.v1 as tf
import imagenet_input

class InputPipeline(object):
    def __init__(self,
                 isTrain,
                 datadir,
                 numImages,
                 batch_size=64,
                 augment = None,
                 cache=False,
                 image_size=160,
                 num_parallel_calls=4,
                 dtype_out=tf.float32):
        self.numImages = numImages
        self.isTrain = isTrain
        self.datadir = datadir
        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls
        self.dtype_out = dtype_out
        self.cache = cache
        self.image_size=image_size
        #with tf.device('/gpu:0'):
        self.params =	{'batch_size': self.batch_size}
        self.dataset = imagenet_input.ImageNetInput(  # pylint: disable=g-complex-comprehension
                is_training=self.isTrain,
                data_dir=self.datadir,
                transpose_input=False,
                cache=self.cache,
                image_size=self.image_size,
                num_parallel_calls=self.num_parallel_calls,
                include_background_label=False,
                use_bfloat16=False,
                augment_name=augment,
                randaug_num_layers=None,
                randaug_magnitude=None)
    
    
        self.dataIter = self.dataset.input_fn(self.params)
        #self.dataIter = self.dataIter.make_one_shot_iterator()
        self.dataIter = tf.data.make_one_shot_iterator(self.dataIter)

    def get_batch(self):
        #with tf.device('/cpu:0'):
        imgs,labels = self.dataIter.get_next()
        images_batch = tf.cast(imgs, tf.float32, name='cast')
        labels_batch = tf.cast(labels, tf.int64, name='cast1')
        return images_batch,tf.cast(tf.one_hot(labels_batch, depth=10),tf.float32)
    
    def getNumImages(self):
        return self.numImages
