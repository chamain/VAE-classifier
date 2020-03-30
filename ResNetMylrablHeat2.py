import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import time

from ops import resblock,bottle_resblock,get_residual_layer,conv,batch_norm,global_avg_pooling,fully_conneted,relu,classification_loss,entropyLossHeat2
from utils import quantize,transpose_NHWC_to_NCHW,_Network3D,bitcost_to_bpp,transpose_NCHW_to_NHWC
from tqdm import tqdm
import inputpipelineMy as dataset
from collections import namedtuple
import numpy as np

# z is the bottleneck before quantization
EncoderOutput = namedtuple('EncoderOutput', ['qbar', 'qhard', 'symbols', 'z', 'heatmap'])

# returned by _Network._quantize
_QuantizerOutput = namedtuple('_QuantizerOutput', ['qbar', 'qsoft', 'qhard', 'symbols'])

class ResNet(object):
    def __init__(self, sess, args):
        self.model_name = 'ResNet'
        self.sess = sess
        self.dataset_name = args.dataset
        self.__log = open('logHeatn6beta4.txt','w')
        if self.dataset_name == 'imgnetwoof' :
            self.dataDir = '/home/lchamain/chamain/imagenet/imagenet/data/tfrecordsimwoof/train/' ## change this to point tfrecords location
            self.nTrImgs = 9025
            self.nTsImgs = 3929
            self.test_x, self.test_y = None,None
            self.nParallelTreads = 32
            self.batch_size = args.batch_size
            self.img_size = 160
            self.c_dim = 3
            self.label_dim = 10
            self.H_target = 0.2
            
            self.ip_train = dataset.InputPipeline(
                True,
                self.dataDir,
                self.nTrImgs,
                batch_size=self.batch_size,
                augment = None,#'autoaugment',
                cache = False,
                image_size=self.img_size,
                num_parallel_calls = self.nParallelTreads)
    
# =============================================================================
#             self.ip_test = dataset.InputPipeline(
#                     False,
#                     self.dataDir,
#                     self.nTsImgs,
#                     batch_size=self.batch_size,#self.nTsImgs,
#                     image_size=self.img_size,
#                     num_parallel_calls = self.nParallelTreads)
# =============================================================================
            self.ip_test = dataset.InputPipeline(
                    True,
                    self.dataDir,
                    self.nTsImgs,
                    batch_size=self.batch_size,#self.nTsImgs,
                    image_size=self.img_size,
                    num_parallel_calls = self.nParallelTreads)


        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir

        self.res_n = args.res_n

        self.epoch = args.epoch
        #self.batch_size = args.batch_size
        #self.iteration = self.nTrImgs // self.batch_size
        self.iteration = 1 ########################################### for testing consider one batch at a time
        self.tsiteration = self.nTsImgs // self.batch_size

        self.init_lr = args.lr
        
        ### quantizer variables
        self.ncenters = 6
        self._centers = self.createCenters(self.ncenters,-2,2)
        self.pc = _Network3D(kernel_size=3, num_centers=self.ncenters) ##kernelsize,numcenters


    
    def createCenters(self,num_cernters,rmin,rmax):
        centerInitializer = tf.random_uniform_initializer(minval=rmin, maxval=rmax, seed=666)
        centerVariable = tf.get_variable('centers', shape=(num_cernters,), dtype=tf.float32,
                                        initializer=centerInitializer)
        return centerVariable
    ##################################################################################
    # encoder
    ##################################################################################
    
    def quantizer(self, x, is_training=True, reuse=False, data_format = 'NHWC'):
            with tf.variable_scope("quantizer", reuse=reuse):
                qsoft, qhard, symbols = quantize(x, self._centers, sigma=1,data_format = data_format)
                with tf.name_scope('qbar'):
                    qbar = qsoft + tf.stop_gradient(qhard - qsoft)
                return _QuantizerOutput(qbar, qsoft, qhard, symbols)
                return x
    def encoder(self, x, is_training=True, reuse=False):
        with tf.variable_scope("encoder", reuse=reuse):

            if self.res_n < 50 :
                residual_block = resblock
            else :
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 16 # paper is 64
            print('******************encoder*********** input:',x.shape)
            x = conv(x, channels=ch*2, kernel=5, stride=2, scope='conv')
            print('******************encoder*********** conv:',x.shape) #80X16
            for i in range(residual_list[0]) :
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock0_' + str(i))
            print('******************model*********** resblock0_:',x.shape)
            ########################################################################################################

            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0') #40X32

            for i in range(1, residual_list[1]) :
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))
            print('******************encoder*********** resblock1_:',x.shape)
            ########################################################################################################

            C = 2*ch +1 #for heatmap
            x = residual_block(x, channels=C, is_training=is_training, downsample=True, scope='resblock2_0_0') #20 X32
            print('******************encoder break*********** resblock2_:',x.shape)
            x = batch_norm(x, is_training, scope='batch_norm')
            ## convert to NCHW
            x = transpose_NHWC_to_NCHW(x)
            heatmap = _get_heatmap3D(bottleneck=x,reuse=reuse)
            
            x = _mask_with_heatmap(x, heatmap,False)
            x = transpose_NCHW_to_NHWC(x)
            qout = self.quantizer(x,is_training=is_training, reuse=reuse, data_format = 'NCHW')
            return EncoderOutput(qout.qbar, qout.qhard, qout.symbols, x, heatmap)
        
                
    def classifier(self, x, is_training=True, reuse=False):
        with tf.variable_scope("classifier", reuse=reuse):
                if self.res_n < 50 :
                    residual_block = resblock
                else :
                    residual_block = bottle_resblock
    
                residual_list = get_residual_layer(self.res_n)
    
                ch = 32 # paper is 64
                #print('******************model break*********** resblock2_:',x.shape)
                #x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_0')
                x = conv(x, channels=ch*4, kernel=3, stride=1, scope='conv2')
                x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_0_0')
                
                for i in range(1, residual_list[2]) :
                    x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))
                print('******************classifier*********** resblock2_:',x.shape)
                ########################################################################################################
                
                x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')
            
                for i in range(1, residual_list[3]) :
                    x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))
                print('******************classifier*********** resblock3_:',x.shape)
                ########################################################################################################
    
    
                x = batch_norm(x, is_training, scope='batch_norm')
                x = relu(x)
    
                x = global_avg_pooling(x)
                print('******************classifier*********** pool:',x.shape)
                x = fully_conneted(x, units=self.label_dim, scope='logit')
                print('******************classifer*********** fc:',x.shape)
                return x
             
    ##################################################################################
    # Model
    ##################################################################################
    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):
            
            z = self.encoder(x, is_training=is_training, reuse=reuse)
            print('******************qauntizer***********:',z.qbar.shape)
            y = self.classifier(z.qbar,is_training=is_training, reuse=reuse)
            # stop_gradient is beneficial for training. it prevents multiple gradients flowing into the heatmap.
            #pc_in = tf.stop_gradient(z.qbar)
            pc_in = z.qbar
            ## NHWC to NCHW
            #print('******************input to pc***********:',pc_in.shape)
            pc_in = transpose_NHWC_to_NCHW(pc_in)
            print('******************input to pc transposed***********:',pc_in.shape)
            bc_train = self.pc.bitcost(pc_in,transpose_NHWC_to_NCHW(z.symbols), is_training=is_training, pad_value=self.pc.auto_pad_value(self._centers))
            bpp_train = bitcost_to_bpp(bc_train, transpose_NHWC_to_NCHW(x))
            
            ##
            return y,bc_train,bpp_train,z.heatmap
    def build_model(self):
        """ Graph Input """
        self.train_inptus = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='train_inputs')
        self.train_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name='train_labels')

        self.test_inptus = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='test_inputs')
        self.test_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name='test_labels')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Model """
        self.train_logits, self.train_bc,self.train_bpp,self.train_heat = self.network(self.train_inptus)
        
        print('////////////////////////////////////////////////////////')
        print('////////////////////////////////////////////////////////')
        print('list of trainable variables:')
        print(tf.trainable_variables())
        
        
            
        self.test_logits,self.test_bc,self.test_bpp,self.test_heat = self.network(self.test_inptus, is_training=False, reuse=True)
        

        self.train_closs, self.train_accuracy = classification_loss(logit=self.train_logits, label=self.train_labels)
        self.test_closs, self.test_accuracy = classification_loss(logit=self.test_logits, label=self.test_labels)
        
        self.beta = 4.0
        reg_loss = tf.losses.get_regularization_loss()
        self.train_loss = self.train_closs + reg_loss
        self.train_Hreal, self.train_Hmask,self.train_H = entropyLossHeat2(self.train_bc,self.H_target,self.train_heat)
        self.train_entropy_loss = self.beta*self.train_H
        self.test_loss = self.test_closs + reg_loss
        self.test_Hreal, self.test_Hmask, self.test_H = entropyLossHeat2(self.test_bc,self.H_target,self.test_heat)


        """ Training """
        self.optimcl = tf.train.MomentumOptimizer(self.lr, momentum=0.9).minimize(self.train_loss)
        self.optimEn = tf.train.AdamOptimizer(self.lr*0.0001).minimize(self.train_entropy_loss)
        self.optim = tf.group(self.optimcl,self.optimEn)

        """" Summary """
        self.summary_train_loss = tf.summary.scalar("train_loss", self.train_loss)
        self.summary_train_accuracy = tf.summary.scalar("train_accuracy", self.train_accuracy)
        self.summary_train_closs = tf.summary.scalar("train_closs", self.train_closs)
        self.summary_train_bpp = tf.summary.scalar("train_bpp", self.train_bpp)

        self.summary_test_loss = tf.summary.scalar("test_loss", self.test_loss)
        self.summary_test_accuracy = tf.summary.scalar("test_accuracy", self.test_accuracy)
        self.summary_test_closs = tf.summary.scalar("test_closs", self.test_closs)
        self.summary_test_bpp = tf.summary.scalar("test_bpp", self.test_bpp)


        self.train_summary = tf.summary.merge([self.summary_train_loss, self.summary_train_accuracy])
        self.test_summary = tf.summary.merge([self.summary_test_loss, self.summary_test_accuracy])

    ##################################################################################
    # Train
    ##################################################################################
    
    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        ###no restoring
        
        epoch_lr = self.init_lr
        start_epoch = 0
        start_batch_id = 0
        counter = 1
        print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        batch = self.ip_train.get_batch()
        for epoch in range(start_epoch, self.epoch):
            avgAcc = 0
            avgloss = 0
            avgbpp = 0
            avgcloss = 0
            avgRbpp = 0
            avgMbpp = 0
            if epoch == int(self.epoch * 0.5) or epoch == int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1

            # get batch data
            
            for idx in range(start_batch_id, self.iteration):
                batch1 = self.sess.run(batch)
                #print('.......................trbatch shape',batch1[0].shape,batch1[1].shape)
                #batch_x = self.train_x[idx*self.batch_size:(idx+1)*self.batch_size]
                #batch_y = self.train_y[idx*self.batch_size:(idx+1)*self.batch_size]

                #batch_x = data_augmentation(batch_x, self.img_size, self.dataset_name)
                batch_x = (batch1[0] - 120.707)/64.15
                batch_y = batch1[1]
                train_feed_dict = {
                    self.train_inptus : batch_x,
                    self.train_labels : batch_y,
                    self.lr : epoch_lr
                }
                #print('label tr average:',np.mean(batch_x))
                
                # update network
                _,train_closs,train_accuracy,train_bpp,train_Hreal,train_Hmask,centers= self.sess.run(
                    [self.optim,self.train_closs, self.train_accuracy,self.train_bpp, self.train_Hreal,self.train_Hmask,self._centers], feed_dict=train_feed_dict)
                #self.writer.add_summary(summary_str, counter)
                # display training status
                counter += 1
                #avgloss += train_loss
                avgcloss += train_closs
                avgAcc += train_accuracy
                avgbpp += train_bpp
                avgRbpp += train_Hreal
                avgMbpp += train_Hmask
                
            # test
#            batchT = self.ip_test.get_batch()
#            batch2 = self.sess.run(batchT)
#            self.test_x = batch2[0]
#            self.test_y = batch2[1]
#            test_feed_dict = {
#                    self.test_inptus : self.test_x,
#                    self.test_labels : self.test_y
#                }
#            summary_str, test_loss, test_accuracy = self.sess.run(
#                    [self.test_summary, self.test_loss, self.test_accuracy], feed_dict=test_feed_dict)
#            self.writer.add_summary(summary_str, counter)
#
#                
            #print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.2f, test_accuracy: %.2f, learning_rate : %.4f" \
            #          % (epoch, idx, self.iteration, time.time() - start_time, train_accuracy, epoch_lr))
            print("Epoch: [%2d], tr_acc: %.2f, trcloss: %.2f, tr_bpp %.4f, tr_Rbpp %.4f, tr_Mbpp %.4f, lr : %.4f"  \
                          % (epoch, avgAcc/self.iteration, avgcloss/self.iteration, avgbpp/self.iteration, avgRbpp/self.iteration,avgMbpp/self.iteration, epoch_lr))
            print('centers:',centers.flatten())
            #print('heat:',train_heat[0,:,0,0].flatten())
            #print('zbar:',zbar.flatten())
            self.__log.write(str(epoch) + '\t'+ str(avgAcc/self.iteration) + '\t'+ str(avgcloss/self.iteration) + '\t' + str(avgMbpp/self.iteration) + '\n')
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0
            #if epoch == 10:
            self.testbt(batch1) ###################### I send the same batch for testing to show that test and train accuracies are different.
            

            # save model
            self.save(self.checkpoint_dir, counter)
            

        # save model for final step
        self.save(self.checkpoint_dir, counter)
        self.__log.close()

    @property
    def model_dir(self):
        return "{}{}_{}_{}_{}".format(self.model_name, self.res_n, self.dataset_name, self.batch_size, self.init_lr)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
# =============================================================================
#         tf.global_variables_initializer().run()
# 
#         self.saver = tf.train.Saver()
#         could_load, checkpoint_counter = self.load(self.checkpoint_dir)
# 
#         if could_load:
#             print(" [*] Load SUCCESS")
#         else:
#             print(" [!] Load failed...")
# =============================================================================
            
        batchT = self.ip_test.get_batch()
        acc = 0
        bpp = 0
        Rbpp =0
        Mbpp =0
        self.tsiteration = self.iteration
        for idx in range(self.tsiteration):
            batch2 = self.sess.run(batchT)
            self.test_x = (batch2[0] - 120.707)/64.15
            self.test_y = batch2[1]
            test_feed_dict = {
                self.test_inptus: self.test_x,
                self.test_labels: self.test_y
            }
            
    
            test_accuracy,test_bpp, test_Rbpp,test_Mbpp = self.sess.run([self.test_accuracy,self.test_bpp,self.test_Hreal,self.test_Hmask], feed_dict=test_feed_dict)
            acc += test_accuracy
            bpp += test_bpp
            Rbpp += test_Rbpp
            Mbpp += test_Mbpp
        print("test_acc: %.2f,test_bpp %.4f, test_Rbpp %.4f, test_Mbpp %.4f"  \
                          % ( acc/self.tsiteration, bpp/self.tsiteration,Rbpp/self.tsiteration,Mbpp/self.tsiteration))
        #print("test_accuracy: {}".format(acc/self.tsiteration)," bpp: {}".format(bpp/self.tsiteration))
        
    def testbt(self, batchTr):
# =============================================================================
#         tf.global_variables_initializer().run()
# 
#         self.saver = tf.train.Saver()
#         could_load, checkpoint_counter = self.load(self.checkpoint_dir)
# 
#         if could_load:
#             print(" [*] Load SUCCESS")
#         else:
#             print(" [!] Load failed...")
# =============================================================================
            
        batchT = batchTr
        acc = 0
        bpp = 0
        Rbpp =0
        Mbpp =0
        self.tsiteration = self.iteration
        for idx in range(self.tsiteration):
            batch2 = batchT
            self.test_x = (batch2[0] - 120.707)/64.15
            self.test_y = batch2[1]
            test_feed_dict = {
                self.test_inptus: self.test_x,
                self.test_labels: self.test_y
            }
            
            #print('label test average:',np.mean(self.test_x))
            test_accuracy,test_bpp, test_Rbpp,test_Mbpp = self.sess.run([self.test_accuracy,self.test_bpp,self.test_Hreal,self.test_Hmask], feed_dict=test_feed_dict)
            acc += test_accuracy
            bpp += test_bpp
            Rbpp += test_Rbpp
            Mbpp += test_Mbpp
        print("test_acc: %.2f,test_bpp %.4f, test_Rbpp %.4f, test_Mbpp %.4f"  \
                          % ( acc/self.tsiteration, bpp/self.tsiteration,Rbpp/self.tsiteration,Mbpp/self.tsiteration))
        #print("test_accuracy: {}".format(acc/self.tsiteration)," bpp: {}".format(bpp/self.tsiteration))
            
            
def _get_heatmap3D(bottleneck,reuse = False):
        """
        create heatmap3D, where
            heatmap3D[x, y, c] = heatmap[x, y] - c \intersect [0, 1]
        """
        assert bottleneck.shape.ndims == 4, bottleneck.shape

        with tf.variable_scope("heatmap", reuse=reuse):
            C = int(bottleneck.shape[1]) - 1  # -1 because first channel is heatmap

            heatmap_channel = bottleneck[:, 0, :, :]  # NHW
            heatmap2D = tf.nn.sigmoid(heatmap_channel) * C  # NHW
            c = tf.range(C, dtype=tf.float32)  # C

            # reshape heatmap2D for broadcasting
            heatmap = tf.expand_dims(heatmap2D, 1)  # N1HW
            # reshape c for broadcasting
            c = tf.reshape(c, (C, 1, 1))  # C11

            # construct heatmap3D
            # if heatmap[x, y] == C, then heatmap[x, y, c] == 1 \forall c \in {0, ..., C-1}
            heatmap3D = tf.maximum(tf.minimum(heatmap - c, 1), 0, name='heatmap3D')  # NCHW
            return heatmap3D


def _mask_with_heatmap(bottleneck, heatmap3D,reuse=False):
        with tf.variable_scope("heatmap_mask", reuse=reuse):
            bottleneck_without_heatmap = bottleneck[:, 1:, ...]
            return heatmap3D * bottleneck_without_heatmap

    
