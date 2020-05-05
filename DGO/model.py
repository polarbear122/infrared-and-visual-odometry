# -*- coding: utf-8 -*-
from __future__ import print_function
from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  gradient,
  lrelu,
  weights_spectral_norm,
  l2_norm
)

import time
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import time
from des_densefuse_net import DenseFuseNet
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class CGAN(object):

  def __init__(self, 
               sess, 
               image_size=132,
               label_size=120,
               batch_size=32,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None,
               des_model_pre_path=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size
    self.des_model_pre_path = des_model_pre_path
    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    with tf.name_scope('IR_input'):
        # 红外图像patch
        self.images_ir = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_ir')
        self.labels_ir = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_ir')
    with tf.name_scope('VI_input'):
        # 可见光图像patch
        self.images_vi = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_vi')
        self.labels_vi = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_vi')
        # self.labels_vi_CLAHE = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_vi_CLAHE')
        # self.labels_vi_gradient=gradient(self.labels_vi)
        # print('tf.placeholder:{}'.format(tf.placeholder))
        # print('self.labels_vi:\n{}'.format(self.labels_vi))
    with tf.name_scope('input'):
        # 将红外和可见光图像在通道方向连起来，第一通道是红外图像，第二通道是可见光图像
        # self.resize_ir=tf.image.resize_images(self.images_ir, (self.image_size, self.image_size), method=2)
        self.input_image=tf.concat([self.images_ir,self.images_vi],axis=-1)
        # print('self.input_image.shape:{}'.format(self.input_image.shape))
    # self.pred=tf.clip_by_value(tf.sign(self.pred_ir-self.pred_vi),0,1)
    # 融合图像
    with tf.name_scope('fusion'): 
        self.fusion_image=self.fusion_model(self.input_image)
        print('end of tf.name_scope(fusion) 融合阶段结束')
    with tf.name_scope('d_loss'):
        # 设置区分器的损失discriminator
        # 判决器对可见光图像和融合图像的预测
        #pos=self.discriminator(self.labels_vi,reuse=False)

        pos = self.discriminator(self.labels_vi, reuse=False)
        # pos_CLAHE=self.discriminator(self.labels_vi_CLAHE,reuse=False)

        # print('self.labels_vi:{}'.format(self.labels_vi))
        # CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # self.labels_vi= clahe.apply(self.labels_vi)
        # print('pos:{}'.format(pos))
        neg=self.discriminator(self.fusion_image,reuse=True,update_collection='NO_OPS')
        # 把真实样本尽量判成1否则有损失（判决器的损失）
        #pos_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pos, labels=tf.ones_like(pos)))
        #pos_loss=tf.reduce_mean(tf.square(pos-tf.ones_like(pos)))
        # pos_loss_CLAHE=tf.reduce_mean(tf.square(pos-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2)))
        pos_loss=tf.reduce_mean(tf.square(pos-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2)))
        # print('tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2):'.format(tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2)))
        # pos_loss 为区分器中可见光的损失
        # 把生成样本尽量判断成0否则有损失（判决器的损失）
        #neg_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg, labels=tf.zeros_like(neg)))
        #neg_loss=tf.reduce_mean(tf.square(neg-tf.zeros_like(neg)))
        neg_loss=tf.reduce_mean(tf.square(neg-tf.random_uniform(shape=[self.batch_size,1],minval=0,maxval=0.3,dtype=tf.float32)))
        # neg_loss 为区分器中融合图像的损失
        # self.d_loss=pos_loss+neg_loss
        self.d_loss=neg_loss+pos_loss
        # self.d_loss_CLAHE = neg_loss + pos_loss_CLAHE
        tf.summary.scalar('loss_d',self.d_loss)
    with tf.name_scope('g_loss'):
        # 设置生成器损失函数generator
        #self.g_loss_1=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg, labels=tf.ones_like(neg)))
        #self.g_loss_1=tf.reduce_mean(tf.square(neg-tf.ones_like(pos)))
        self.g_loss_1=tf.reduce_mean(tf.square(neg-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2,dtype=tf.float32)))
        tf.summary.scalar('g_loss_1',self.g_loss_1)
        # self.g_loss_1为对抗损失。V（FusionGAN）
        #self.g_loss_2=tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir))
        self.g_loss_2=tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir))+5*tf.reduce_mean(tf.square(gradient(self.fusion_image) -gradient (self.labels_vi)))
        tf.summary.scalar('g_loss_2',self.g_loss_2)
        # self.g_loss_2为content损失

        self.g_loss_total=self.g_loss_1+100*self.g_loss_2
        tf.summary.scalar('loss_g',self.g_loss_total)
    self.saver = tf.train.Saver(max_to_keep=50)
    
  def train(self, config):
    # print('train\n\n')
    if config.is_train:
      # print('config:{}'.format(config.is_train))
      input_setup(self.sess, config,"Train_ir")
      input_setup(self.sess,config,"Train_vi")
      # print('input end')
    else:
      nx_ir, ny_ir = input_setup(self.sess, config,"Test_ir")
      nx_vi,ny_vi=input_setup(self.sess, config,"Test_vi")
      # input_setup(self.sess, config, "Test_ir")
      # input_setup(self.sess, config, "Test_vi")
    # print('train1\n')
    if config.is_train:     
      data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir), "Train_ir", "train.h5")
      data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir), "Train_vi", "train.h5")
    else:
      data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir), "Test_ir", "test.h5")
      data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir), "Test_vi", "test.h5")
    # print('train2\n\n')
    train_data_ir, train_label_ir = read_data(data_dir_ir)
    train_data_vi, train_label_vi = read_data(data_dir_vi)
    #找训练时更新的变量组（判决器和生成器是分开训练的，所以要找到对应的变量）
    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
    # print(self.d_vars)
    self.g_vars = [var for var in t_vars if 'fusion_model' in var.name]
    # print(self.g_vars)
    # clip_ops = []
    # for var in self.d_vars:
        # clip_bounds = [-.01, .01]
        # clip_ops.append(
            # tf.assign(
                # var, 
                # tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            # )
        # )
    # self.clip_disc_weights = tf.group(*clip_ops)
    # Stochastic gradient descent with the standard backpropagation
    with tf.name_scope('train_step'):
        self.train_fusion_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total,var_list=self.g_vars)
        self.train_discriminator_op=tf.train.AdamOptimizer(config.learning_rate).minimize(self.d_loss,var_list=self.d_vars)
    #将所有统计的量合起来
    self.summary_op = tf.summary.merge_all()
    #生成日志文件
    self.train_writer = tf.summary.FileWriter(config.summary_dir + '/train',self.sess.graph,flush_secs=60)
    tf.initialize_all_variables().run()
    counter = 0
    start_time = time.time()

    # if self.load(self.checkpoint_dir):
    #   print(" [*] Load SUCCESS")
    # else:
    #   print(" [!] Load failed...")

    if config.is_train:
      print("Training...")

      for ep in range(config.epoch):
        #print('ep:{}'.format(ep))
        # Run by batch images
        batch_idxs = len(train_data_ir) // config.batch_size
        for idx in range(0, batch_idxs):
          #print('idx:{}'.format(idx))
          batch_images_ir = train_data_ir[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_ir = train_label_ir[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_vi = train_data_vi[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_vi = train_label_vi[idx*config.batch_size : (idx+1)*config.batch_size]

          counter += 1
          for i in range(2):
            _, err_d= self.sess.run([self.train_discriminator_op, self.d_loss], feed_dict={self.images_ir: batch_images_ir, self.images_vi: batch_images_vi, self.labels_vi: batch_labels_vi,self.labels_ir:batch_labels_ir})
            # self.sess.run(self.clip_disc_weights)
          # print(' self.images_vi:\n{}\nshape：{}\n'.format(self.images_vi, batch_images_vi.shape))
          _, err_g,summary_str= self.sess.run([self.train_fusion_op, self.g_loss_total,self.summary_op], feed_dict={self.images_ir: batch_images_ir, self.images_vi: batch_images_vi, self.labels_ir: batch_labels_ir,self.labels_vi:batch_labels_vi})
          #将统计的量写到日志文件里
          self.train_writer.add_summary(summary_str,counter)
          # print("every epoch: [%2d], step: [%2d], time: [%4.4f], loss_d: [%.8f],loss_g:[%.8f]" \
          #       % ((ep + 1), counter, time.time() - start_time, err_d, err_g))
          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_d: [%.8f],loss_g:[%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err_d,err_g))
            #print(a)

        self.save(config.checkpoint_dir, ep)

    else:
      print("Testing...")

      result = self.fusion_image.eval(feed_dict={self.images_ir: train_data_ir, self.labels_ir: train_label_ir,self.images_vi: train_data_vi, self.labels_vi: train_label_vi})
      result=result*127.5+127.5
      result = merge(result, [nx_ir, ny_ir])
      result = result.squeeze()
      image_path = os.path.join(os.getcwd(), config.sample_dir)
      image_path = os.path.join(image_path, "test_image.png")
      imsave(result, image_path)

  # 在这里改变生成器的模型
  def fusion_model(self, img):
    with tf.variable_scope('fusion_model', reuse=False):
      # print('img:{}'.format(img))
      gener = DenseFuseNet(self.des_model_pre_path)
      # print('gener:{}'.format(gener))
      img1 = img[:,:,:,0:1]
      # print('img1:{}'.format(img1))
      img2 = img[:,:,:,1:2]
      # print('img2:{}'.format(img2))s
      fused_img = gener.transform_addition(img1, img2)
      # print('fused_img:{}'.format(fused_img))
      fused_img = tf.image.resize_images(fused_img, [120, 120], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
      return fused_img
  
  def discriminator(self,img,reuse,update_collection=None):
    with tf.variable_scope('discriminator',reuse=reuse):
        with tf.variable_scope('layer_1'):
            weights=tf.get_variable("w_1",[3,3,1,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_1",[32],initializer=tf.constant_initializer(0.0))
            conv1_vi=tf.nn.conv2d(img, weights, strides=[1,2,2,1], padding='VALID') + bias
            conv1_vi = lrelu(conv1_vi)
            #print(conv1_vi.shape)
        with tf.variable_scope('layer_2'):
            weights=tf.get_variable("w_2",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_2",[64],initializer=tf.constant_initializer(0.0))
            conv2_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_vi, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_vi = lrelu(conv2_vi)
            #print(conv2_vi.shape)
        with tf.variable_scope('layer_3'):
            weights=tf.get_variable("w_3",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_3",[128],initializer=tf.constant_initializer(0.0))
            conv3_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_vi, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_vi=lrelu(conv3_vi)
            #print(conv3_vi.shape)
        with tf.variable_scope('layer_4'):
            weights=tf.get_variable("w_4",[3,3,128,256],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_4",[256],initializer=tf.constant_initializer(0.0))
            conv4_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_vi, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_vi=lrelu(conv4_vi)
            conv4_vi = tf.reshape(conv4_vi,[self.batch_size,6*6*256])
        with tf.variable_scope('line_5'):
            weights=tf.get_variable("w_5",[6*6*256,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_5",[1],initializer=tf.constant_initializer(0.0))
            line_5=tf.matmul(conv4_vi, weights) + bias
            #conv3_vi= tf.contrib.layers.batch_norm(conv3_vi, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
    return line_5

  def save(self, checkpoint_dir, step):
    model_name = "CGAN.model"
    model_dir = "%s_%s" % ("CGAN", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("CGAN", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
        return True
    else:
        return False
