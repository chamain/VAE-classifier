# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 15:03:07 2020
trainig for AE-Classifier
@author: Lahiru D. Chamain
"""

import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from Model import AE
import time
import dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description='AE-Classifier training and evaluating a model')
    parser.add_argument('--config' ,help='AE-Classifier configuration: supports v3 and v4', default='v3')
    parser.add_argument('--modeldir',help='save loacation for model', default='./model')
    parser.add_argument('--train_glob',help = 'glob for training tf-records')
    parser.add_argument('--test_glob',help = 'glob for validation tf-records')
    parser.add_argument('--train_imgs',type = int, help = 'number of training images',default=1281167)
    parser.add_argument('--test_imgs',type = int, help = 'number of test/val images',default=50000)
    parser.add_argument('--nclasses',type = int, help = 'number of classes',default=1000)
    parser.add_argument('--batch_size',type = int, help = 'number of test/val images',default=96)
    parser.add_argument('--img_size',type = int, help = 'height/width of image crops for training',default=160)
    parser.add_argument('--epochs',type = int, help = 'number of epochs for training',default=90)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    assert args.train_glob and args.test_glob:
        raise ValueError('Please specify trainig and test globs')
    
    AEConfig = str(args.config)
    modeldir = str(args.modeldir)
    train_glob = str(args,train_glob)
    test_glob = str(args,train_glob)
    
    nTrImgs = args.train_imgs
    nTsImgs = args.test_imgs
    batch_size = args.batch_size
    img_size = args.img_size
    c_dim = 3
    label_dim = args.nclasses
    nTrbatch = int(nTrImgs/batch_size)
    nTsbatch = int(nTsImgs/batch_size)
    nepochs = args.epochs
    
    ##steps
    tot_steps = int(nepochs*nTrImgs/batch_size)
    currentCheckpoint=0
    remainingSteps = tot_steps-currentCheckpoint
    step_bound= [int(tot_steps*0.5),int(tot_steps*0.75)]
    
    ## learning rates
    lrPEScale = 0.0005
    lrvalues = [0.1,0.01,0.001]
    weight_decay = 2e-4
    stepsPerEpoch = int(nTrImgs/batch_size)
    
    #control parameter
    beta=2.0

    if not (os.path.exists(modeldir)):
        os.makedirs(modeldir)
        
    def learning_rate_fn(global_step):
        lr = tf.train.piecewise_constant(global_step,step_bound, lrvalues)
        return lr
        
    def model_fn(features, labels, mode):
            
            model_obj = AE(codec = 'v3',ncenters=2,H_target=0.4)
            model = model_obj.network
            
            global_step=tf.train.get_global_step()
            
            images = tf.reshape(features, [-1,img_size,img_size,c_dim])
            
            
            logits,bpp,H_loss = model(images)
            predicted_logit = tf.argmax(input=logits, axis=1, output_type=tf.int32)
            probabilities = tf.nn.softmax(logits)
            
            
            
            #PREDICT
            predictions = {
              "predicted_logit": predicted_logit,
              "probabilities": probabilities
            }
            if mode == tf.estimator.ModeKeys.PREDICT:
              return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
          
            with tf.name_scope('AEloss'):
                cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                    labels=labels, logits=logits, scope='Closs')
                
                l2_loss = weight_decay * tf.add_n([
                  tf.nn.l2_loss(tf.cast(v, tf.float32))
                  for v in tf.compat.v1.trainable_variables()])
            
                tf.summary.scalar('l2_loss', l2_loss)
                loss = cross_entropy + l2_loss
                tf.summary.scalar('CRloss', loss)
                
                entropyLoss = beta*H_loss
                tf.summary.scalar('bpp', bpp)
            
            with tf.name_scope('accuracy'):
                accuracy1 = tf.metrics.accuracy(
                    labels=labels, predictions=predicted_logit, name='acc1')
                accuracy5 =  tf.metrics.mean(
                    tf.nn.in_top_k(predictions=logits, targets=tf.squeeze(labels), k=5, name='acc5'))
               
                tf.summary.scalar('acc1', accuracy1[1])
                tf.summary.scalar('acc5', accuracy5[1])
                
                
            learning_rate = tf.train.piecewise_constant(global_step,step_bound, lrvalues)
           #EVAL    
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss+entropyLoss,
                    eval_metric_ops={'testacc1': accuracy1,'testacc5': accuracy5,'testbpp':tf.metrics.mean(bpp)},
                    evaluation_hooks=None)
               
        
            
            # Create optimizers
            optimizercls = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(loss)
            optimizerbpp = tf.train.AdamOptimizer(learning_rate*lrPEScale).minimize(entropyLoss,global_step=global_step)
            train_op = tf.group(optimizercls,optimizerbpp)
            
            # Create a hook to print acc, loss & global step every 100 iter.   
            train_hook_list= []
            train_tensors_log = {'acc1': accuracy1[1],
                                 'acc5': accuracy5[1],
                                 'closs': cross_entropy,
                                 'crloss': loss,
                                 'bpploss': bpp,
                                 'tloss': loss+entropyLoss,
                                 'step': global_step,
                                 'remaining steps':tot_steps-global_step}
            
            train_hook_list.append(tf.train.LoggingTensorHook(
                tensors=train_tensors_log, every_n_iter=stepsPerEpoch))
            
            if mode == tf.estimator.ModeKeys.TRAIN:
              return tf.estimator.EstimatorSpec(
                  mode=mode,
                  loss=loss+entropyLoss,
                  train_op=train_op,
                  training_hooks=train_hook_list)
    
    def classifier(_):
            
            ## INPUT           
            datasetTrain = dataset.ReadTFRecords(img_size, batch_size, label_dim,
                                                 train_glob,
                                                 True)
            train_input_fn = datasetTrain.input_fn
            datasetTest = dataset.ReadTFRecords(img_size, batch_size, label_dim,
                                                 test_glob,
                                                 False)
            test_input_fn = datasetTest.input_fn
            
            run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=stepsPerEpoch,
                                                log_step_count_steps=500)
            image_classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,config=run_config)

            image_classifier.train(input_fn=train_input_fn,steps = remainingSteps)
            metrics = image_classifier.evaluate(input_fn=test_input_fn,steps = nTsbatch )
            print('Test:======',metrics)
    tf.app.run(classifier)
    
if __name__ == '__main__':
    main()
        
    
   