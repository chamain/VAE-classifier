# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 16:12:40 2020

@author: Lahiru D. Chamain
"""


from __future__ import absolute_import
import os
import tensorflow as tf
import multiprocessing

def caffe_center_crop(image_encoded,image_size,training,Q,resize_size=256):
    """
    Emulates the center crop function used in caffe
    :param image_encoded:
        Jpeg string
    :param image_size:
        Output width and height
    :param training:
        Whether or not the model is training
    :param resize_size:
        Size to which to resize the decoded image before center croping. Default size is 256
        to match the size used in this script:
        https://github.com/BVLC/caffe/blob/master/examples/imagenet/create_imagenet.sh
    :return:
        Image of size [image_size,image_size,3]
    """
    # decode resize and shape jpeg image
    image = tf.image.decode_jpeg(image_encoded,channels=3)
    
            
    image = tf.image.resize_images(image, [resize_size, resize_size])
    image = tf.reshape(image, [resize_size, resize_size,3])

        
    # when training do random crop and random flip during eval do center crop
    if training:
        image = tf.random_crop(image,[image_size,image_size,3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.convert_image_dtype(image,tf.float32)
        image = tf.clip_by_value(image, 0.0, 255.0)
    else:
        crop_min = tf.cast(tf.abs(resize_size / 2 - (image_size / 2)),tf.int32) 
        crop_max = crop_min+ image_size
        image = image[crop_min:crop_max,crop_min:crop_max,:]
    return image

def _parse_function(example_proto, image_size, num_classes,training,Quality,mean_value=(123.68,116.78,103.94),method="crop"):
    """
    Parses tf-records created with build_imagenet_data.py
    :param example_proto:
        Single example from tf record
    :param image_size:
        Output image size
    :param num_classes:
        Number of classes in dataset
    :param training:
        Whether or not the model is training
    :param mean_value:
        Imagenet mean to subtract from the output iamge
    :param method:
        How to generate the input image
    :return:
        Features dict containing image, and labels dict containing class index and one hot vector
    """

    # Schema of fields to parse
    schema = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
    }


    image_size = tf.cast(image_size,tf.int32)
    mean_value = tf.cast(tf.stack(mean_value),tf.float32)

    # Parse example using schema
    parsed_features = tf.parse_single_example(example_proto, schema)
    jpeg_image = parsed_features["image/encoded"]
    # generate correctly sized image using one of 2 methods
    if method == "crop":
        image = caffe_center_crop(jpeg_image,image_size,training,Q=Quality)
    elif method == "resize":
        image = tf.image.decode_jpeg(jpeg_image)
        image = tf.image.resize_images(image, [image_size, image_size])
    else:
        raise("unknown image process method")
    # subtract mean
    means = tf.broadcast_to(mean_value, tf.shape(image))
    image = (tf.cast(image,tf.float32) - means)/64.15

    #image = (tf.cast(image,tf.float32) - 120.707)/64.15

    # subtract 1 from class index as background class 0 is not used
    label_idx = tf.cast(parsed_features['image/class/label'], dtype=tf.int32)-1

    # create one hot vector
    label_vec = tf.one_hot(label_idx, num_classes)
    
    return tf.reshape(image,[image_size,image_size,3]), label_idx


class ReadTFRecords(object):
    def __init__(self, image_size, batch_size, num_classes,glob_pattern,training,Q=100):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.glob_pattern = glob_pattern
        self.training = training
        self.Q=Q;

    def input_fn(self):
        """
        Read tf records matching a glob pattern
        :param glob_pattern:
            glob pattern eg. "/usr/local/share/Datasets/Imagenet/train-*.tfrecords"
        :param training:
            Whether or not to shuffle the data for training and evaluation
        :return:
            Iterator generating one example of batch size for each training step
        """
        threads = multiprocessing.cpu_count()
        with tf.name_scope("tf_record_reader"):
            # generate file list
            files = tf.data.Dataset.list_files(self.glob_pattern, shuffle=self.training)
           
            #print('files list:',files)

            # parallel fetch tfrecords dataset using the file list in parallel
            dataset = files.apply(tf.contrib.data.parallel_interleave(
                lambda filename: tf.data.TFRecordDataset(filename), cycle_length=threads))
            
            # shuffle and repeat examples for better randomness and allow training beyond one epoch
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(32*self.batch_size))

            # map the parse  function to each example individually in threads*2 parallel calls
            dataset = dataset.map(map_func=lambda example: _parse_function(example, self.image_size, self.num_classes,training=self.training,Quality =self.Q ),
                                  num_parallel_calls=threads)

            # batch the examples
            dataset = dataset.batch(batch_size=self.batch_size)

            #prefetch batch
            dataset = dataset.prefetch(buffer_size=32)

            return dataset.make_one_shot_iterator().get_next()