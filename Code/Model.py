# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 14:12:50 2020
Models for AE-Classifier
@author: Lahiru D. Chamain
"""

import tensorflow as tf
from ops import resblock,bottle_resblock,get_residual_layer,conv,batch_norm,max_pooling,global_avg_pooling,fully_conneted,relu
from ops import entropyLoss
from utils import quantize,transpose_NHWC_to_NCHW,_Network3D, bitcost_to_bpp
from collections import namedtuple

# z is the bottleneck before quantization
EncoderOutput = namedtuple('EncoderOutput', ['qbar', 'qhard', 'symbols', 'z'])

# returned by _Network._quantize
_QuantizerOutput = namedtuple('_QuantizerOutput', ['qbar', 'qsoft', 'qhard', 'symbols'])


class AE(object):
        def __init__(self,codec = 'v3',ncenters=2,H_target=0.4,res_n=18,label_dim=1000,is_training=True,reuse = False):
            self.is_training = is_training
            self.reuse = reuse
            self.res_n = res_n
            self.label_dim=label_dim
            self.ncenters = ncenters
            self._centers = self.createCenters(self.ncenters,-2,2)
            self.pc = _Network3D(kernel_size=3, num_centers=self.ncenters) ##kernelsize,numcenters
            self.H_target = H_target
            self.codec = codec
            
        def createCenters(self,num_cernters,rmin,rmax):
            centerInitializer = tf.random_uniform_initializer(minval=rmin, maxval=rmax, seed=666)
            centerVariable = tf.get_variable('centers', shape=(num_cernters,), dtype=tf.float32,
                                            initializer=centerInitializer)
            return centerVariable
        
        def quantizer(self, x, data_format = 'NHWC'):
                with tf.variable_scope("quantizer", reuse=self.reuse):
                    qsoft, qhard, symbols = quantize(x, self._centers, sigma=1,data_format = data_format)
                    with tf.name_scope('qbar'):
                        qbar = qsoft + tf.stop_gradient(qhard - qsoft)
                    return _QuantizerOutput(qbar, qsoft, qhard, symbols)
                    return x
        ##################################################################################
        # Encoders
        ##################################################################################
        
        def encoderV3(self, x):
            with tf.variable_scope("encoder", reuse=self.reuse):
    
                if self.res_n < 50 :
                    residual_block = resblock
                else :
                    residual_block = bottle_resblock
    
                residual_list = get_residual_layer(self.res_n)
    
                ch =32
                x = conv(x, channels=ch, kernel=7, stride=2, scope='conv1') #80x64
                x = max_pooling(x,kernel=3,stride=2,name='maxpool1')
                
                for i in range(residual_list[0]) :
                    x = residual_block(x, channels=ch, is_training=self.is_training, downsample=False, scope='resblock0_' + str(i)) #40x32
                x = residual_block(x, channels=ch, is_training=self.is_training, downsample=True, scope='resblock1_0')
    
                for i in range(1, residual_list[1]) :
                    x = residual_block(x, channels=ch, is_training=self.is_training, downsample=False, scope='resblock1_' + str(i)) #20x32
    
                x = residual_block(x, channels=ch, is_training=self.is_training, downsample=False, scope='resblock2_0_0') #20x32
                x = batch_norm(x, self.is_training, scope='batch_norm')
                qout = self.quantizer(x,data_format = 'NCHW')
                return EncoderOutput(qout.qbar, qout.qhard, qout.symbols, x)
        
        def encoderV4(self, x):
            with tf.variable_scope("encoder", reuse=self.reuse):
    
                if self.res_n < 50 :
                    residual_block = resblock
                else :
                    residual_block = bottle_resblock
    
                residual_list = get_residual_layer(self.res_n)
    
                ch =128
                x = conv(x, channels=ch, kernel=7, stride=2, scope='conv1') 
                x = max_pooling(x,kernel=3,stride=2,name='maxpool1')
                
                for i in range(residual_list[0]) :
                    x = residual_block(x, channels=ch, is_training=self.is_training, downsample=False, scope='resblock0_' + str(i)) #40x32
                x = residual_block(x, channels=ch, is_training=self.is_training, downsample=True, scope='resblock1_0')
    
                for i in range(1, residual_list[1]) :
                    x = residual_block(x, channels=ch, is_training=self.is_training, downsample=False, scope='resblock1_' + str(i)) #20x32
    
                x = residual_block(x, channels=ch, is_training=self.is_training, downsample=False, scope='resblock2_0_0')
                x = conv(x, channels=ch//4, kernel=3, stride=1, scope='conv2')
                x = batch_norm(x, self.is_training, scope='batch_norm')
                qout = self.quantizer(x,data_format = 'NCHW')
                
                return EncoderOutput(qout.qbar, qout.qhard, qout.symbols, x)
            
        ##################################################################################
        # Classifiers
        ##################################################################################
        def classifierV3(self, x):
            with tf.variable_scope("classifier", reuse=self.reuse):
                    if self.res_n < 50 :
                        residual_block = resblock
                    else :
                        residual_block = bottle_resblock
                    residual_list = get_residual_layer(self.res_n)
                    
                    ch = 64
                    x = conv(x, channels=ch*4, kernel=3, stride=1, scope='conv3')
                    
                    x = residual_block(x, channels=ch*4, is_training=self.is_training, downsample=False, scope='resblock2_0_1')
    
                    for i in range(1, residual_list[2]) :
                        x = residual_block(x, channels=ch*4, is_training=self.is_training, downsample=False, scope='resblock2_' + str(i)) #10x256
                        
                    x = residual_block(x, channels=ch*8, is_training=self.is_training, downsample=True, scope='resblock_3_0')
        
                    for i in range(1, residual_list[3]) :
                        x = residual_block(x, channels=ch*8, is_training=self.is_training, downsample=False, scope='resblock_3_' + str(i)) #5x512
                    
                    x = batch_norm(x, self.is_training, scope='batch_norm')
                    x = relu(x)
        
                    x = global_avg_pooling(x)
                    x = fully_conneted(x, units=self.label_dim, scope='logit')
                    return x
                 
        ##################################################################################
        # Models
        ##################################################################################
        def networkV3(self, x):
            with tf.variable_scope("network", reuse=self.reuse):
                
                z = self.encoderV3(x)
                y = self.classifierV3(z.qbar)
                pc_in = z.qbar
                pc_in = transpose_NHWC_to_NCHW(pc_in)
                bc_train = self.pc.bitcost(pc_in, transpose_NHWC_to_NCHW(z.symbols), is_training=self.is_training, pad_value=self.pc.auto_pad_value(self._centers))
                bpp_train = bitcost_to_bpp(bc_train, transpose_NHWC_to_NCHW(x))
                H_loss = entropyLoss(bpp_train,self.H_target)
                
                return  y,bpp_train,H_loss
            
        def networkV4(self, x):
            with tf.variable_scope("network", reuse=self.reuse):
                
                z = self.encoderV4(x)
                y = self.classifierV3(z.qbar)
                pc_in = z.qbar
                pc_in = transpose_NHWC_to_NCHW(pc_in)
                bc_train = self.pc.bitcost(pc_in, transpose_NHWC_to_NCHW(z.symbols), is_training=self.is_training, pad_value=self.pc.auto_pad_value(self._centers))
                bpp_train = bitcost_to_bpp(bc_train, transpose_NHWC_to_NCHW(x))
                H_loss = entropyLoss(bpp_train,self.H_target)
                
                return  y,bpp_train,H_loss
            
        def network(self,x):
            if(self.codec == 'v3'):
                return self.networkV3(x)
            elif(self.codec == 'v4'):
                return self.networkV4(x)
            else:
                print('Unknown configuration')
                return None