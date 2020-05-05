# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
import glob
import cv2

# reader = tf.train.NewCheckpointReader("./checkpoint/CGAN_120/CGAN.model-9")


def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    #flatten=True 以灰度图的形式读取 
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def imsave(image, path):
  return scipy.misc.imsave(path, image)
  
  
def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "*.png")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

# def fusion_model(img):
#     with tf.variable_scope('fusion_model'):
#         with tf.variable_scope('layer1'):
#             weights=tf.get_variable("w1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/w1')))
#             bias=tf.get_variable("b1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/b1')))
#             conv1_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
#             conv1_ir = lrelu(conv1_ir)
#         with tf.variable_scope('layer2'):
#             weights=tf.get_variable("w2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/w2')))
#             bias=tf.get_variable("b2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/b2')))
#             conv2_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_ir, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
#             conv2_ir = lrelu(conv2_ir)
#         with tf.variable_scope('layer3'):
#             weights=tf.get_variable("w3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/w3')))
#             bias=tf.get_variable("b3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/b3')))
#             conv3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_ir, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
#             conv3_ir = lrelu(conv3_ir)
#         with tf.variable_scope('layer4'):
#             weights=tf.get_variable("w4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/w4')))
#             bias=tf.get_variable("b4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/b4')))
#             conv4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_ir, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
#             conv4_ir = lrelu(conv4_ir)
#         with tf.variable_scope('layer5'):
#             weights=tf.get_variable("w5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5')))
#             bias=tf.get_variable("b5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/b5')))
#             conv5_ir= tf.nn.conv2d(conv4_ir, weights, strides=[1,1,1,1], padding='VALID') + bias
#             conv5_ir=tf.nn.tanh(conv5_ir)
#     return conv5_ir

def fusion_model(x_input):
  """
  Implementation of the popular ResNet50 the following architecture:
  CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
  -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
  """
  with tf.variable_scope('fusion_model') :
      # tf.get_variable_scope().reuse_variables()
      x = tf.pad(x_input, tf.constant([[0, 0], [3, 3, ], [3, 3], [0, 0]]), "CONSTANT")
      x = tf.image.resize_images(x, [132 * 2, 132 * 2], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
      # training = tf.placeholder(tf.bool, name='training')
      training = False
      #stage 1
      with tf.variable_scope('res_layer1'):
        # tf.get_variable_scope().reuse_variables()
        # w_conv1 = weight_variable([7, 7, 2, 64])
        w_conv1 = tf.get_variable('w1_1',initializer=tf.constant(reader.get_tensor('fusion/fusion_model/res_layer1/Variable')))
        x = tf.nn.conv2d(x, w_conv1, strides=[1, 2, 2, 1], padding='VALID')
        x = tf.layers.batch_normalization(x, axis=3, training=training)
        x = tf.nn.relu(x)
        # x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
        #                strides=[1, 2, 2, 1], padding='VALID')
        # assert (x.get_shape() == (x.get_shape()[0], 15, 15, 64))
        x = tf.image.resize_images(x, [132 * 2, 132 * 2], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
        print('x1:{}'.format(x))
      with tf.variable_scope('res_layer2'):
        # tf.get_variable_scope().reuse_variables()
        #stage 2
        x = convolutional_block(x, 3, 64, [64, 64, 256], 2, 'a', training, stride=1)
        x = identity_block(x, 3, 256, [64, 64, 256], stage=2, block='b', training=training)
        #  X_input, kernel_size, in_filter, out_filters, stage, block, trainin
        x = identity_block(x, 3, 256, [64, 64, 256], stage=2, block='c', training=training)
        print('x2:{}'.format(x))
      with tf.variable_scope('res_layer3'):
        # tf.get_variable_scope().reuse_variables()
        #stage 3
        # size = x.shape()*2
        # x = tf.resize_images(x, size, method=tf.ResizeMethod.BILINEAR, align_corners=False)
        x = convolutional_block(x, 3, 256, [128,128,512], 3, 'a', training)
        x = identity_block(x, 3, 512, [128,128,512], 3, 'b', training=training)
        x = identity_block(x, 3, 512, [128,128,512], 3, 'c', training=training)
        x = identity_block(x, 3, 512, [128,128,512], 3, 'd', training=training)
        print('x3:{}'.format(x))
      with tf.variable_scope('res_layer4'):
        # tf.get_variable_scope().reuse_variables()
        #stage 4
        x = convolutional_block(x, 3, 512, [256, 256, 1024], 4, 'a', training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'b', training=training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'c', training=training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'd', training=training)
        x = identity_block (x, 3, 1024, [256, 256, 1024], 4, 'e', training=training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'f', training=training)
        print('x4:{}'.format(x))
      with tf.variable_scope('res_layer5'):
        # tf.get_variable_scope().reuse_variables()
        #stage 5
        x = tf.image.resize_images(x, [132 * 2, 132 * 2], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
        x = convolutional_block(x, 3, 1024, [512, 512, 2048], 5, 'a', training)
        x = identity_block(x, 3, 2048, [512, 512, 2048], 5, 'b', training=training)
        x = identity_block(x, 3, 2048, [512, 512, 2048], 5, 'c', training=training)
        # x = tf.nn.avg_pool(x, [1, 2, 2, 1], strides=[1,1,1,1], padding='VALID')
        print('x5:{}'.format(x))
      with tf.variable_scope('layer1'):
        weights = tf.get_variable("w1", [5, 5, 2048, 256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable("b1", [256], initializer=tf.constant_initializer(0.0))
        conv1_ir = tf.contrib.layers.batch_norm(
          tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='VALID') + bias, decay=0.9, updates_collections=None,
          epsilon=1e-5, scale=True)
        conv1_ir = lrelu(conv1_ir)
        print('conv1_ir:{}'.format(conv1_ir))
      with tf.variable_scope('layer2'):
        weights = tf.get_variable("w2", [5, 5, 256, 128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable("b2", [128], initializer=tf.constant_initializer(0.0))
        conv2_ir = tf.contrib.layers.batch_norm(
          tf.nn.conv2d(conv1_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias, decay=0.9,
          updates_collections=None, epsilon=1e-5, scale=True)
        conv2_ir = lrelu(conv2_ir)
        print('conv2_ir:{}'.format(conv2_ir))
      with tf.variable_scope('layer3'):
        weights = tf.get_variable("w3", [3, 3, 128, 64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        
        bias = tf.get_variable("b3", [64], initializer=tf.constant_initializer(0.0))
        conv3_ir = tf.contrib.layers.batch_norm(
          tf.nn.conv2d(conv2_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias, decay=0.9,
          updates_collections=None, epsilon=1e-5, scale=True)
        conv3_ir = lrelu(conv3_ir)
        print('conv3_ir:{}'.format(conv3_ir))
      with tf.variable_scope('layer4'):
        weights = tf.get_variable("w4", [3, 3, 64, 32], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        
        bias = tf.get_variable("b4", [32], initializer=tf.constant_initializer(0.0))
        conv4_ir = tf.contrib.layers.batch_norm(
          tf.nn.conv2d(conv3_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias, decay=0.9,
          updates_collections=None, epsilon=1e-5, scale=True)
        conv4_ir = lrelu(conv4_ir)
        print('conv4_ir:{}'.format(conv4_ir))
      with tf.variable_scope('layer5'):
        weights = tf.get_variable("w5", [1, 1, 32, 1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable("b5", [1], initializer=tf.constant_initializer(0.0))
        conv5_ir = tf.nn.conv2d(conv4_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
        conv5_ir = tf.nn.tanh(conv5_ir)
        print('conv5_ir:{}'.format(conv5_ir))
  conv5_ir = tf.image.resize_images(conv5_ir, [132 * 2, 132 * 2], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
  return conv5_ir

# def weight_variable(shape):
#   initial = tf.truncated_normal(shape, stddev=0.1)
#   return tf.Variable(initial)


def identity_block(X_input, kernel_size, in_filter, out_filters, stage, block, training):
  block_name = 'res' + str(stage) + block
  f1, f2, f3 = out_filters
  with tf.variable_scope(block_name):
    X_shortcut = X_input
    tensor_name1 = 'fusion/fusion_model/res_layer' + str(stage) + '/' + block_name + '/Variable'
    tensor_name2 = 'fusion/fusion_model/res_layer' + str(stage) + '/' + block_name + '/Variable_1'
    tensor_name3 = 'fusion/fusion_model/res_layer' + str(stage) + '/' + block_name + '/Variable_2'
    weight_name1 = 'w'+str(stage)+'_'+block+'1'
    weight_name2 = 'w' + str(stage) + '_' + block + '2'
    weight_name3 = 'w' + str(stage) + '_' + block + '3'
    # first
    # W_conv1 = weight_variable([1, 1, in_filter, f1])
    # W_conv1 = tf.get_variable("w5", initializer=tf.constant(reader.get_tensor('fusion/fusion_model/res_layer1/Variable')))
    W_conv1 = tf.get_variable(weight_name1,initializer=tf.constant(reader.get_tensor(tensor_name1)))
    X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
    X = tf.layers.batch_normalization(X, axis=3, training=training)
    X = tf.nn.relu(X)

    # second
    # W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
    W_conv2 = tf.get_variable(weight_name2,initializer=tf.constant(reader.get_tensor(tensor_name2)))
    X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    X = tf.layers.batch_normalization(X, axis=3, training=training)
    X = tf.nn.relu(X)

    # third
    # W_conv3 = weight_variable([1, 1, f2, f3])
    W_conv3 = tf.get_variable(weight_name3,initializer=tf.constant(reader.get_tensor(tensor_name3)))
    X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
    X = tf.layers.batch_normalization(X, axis=3, training=training)

    # final step
    add = tf.add(X, X_shortcut)
    add_result = tf.nn.relu(add)
  return add_result

def convolutional_block(X_input, kernel_size, in_filter,
                        out_filters, stage, block, training, stride=2):
  block_name = 'res' + str(stage) + block
  with tf.variable_scope(block_name):
    f1, f2, f3 = out_filters
  
    x_shortcut = X_input
    # first
    conv_tensor_name1 = 'fusion/fusion_model/res_layer'+str(stage)+'/'+block_name+'/Variable'
    conv_tensor_name2 = 'fusion/fusion_model/res_layer' + str(stage) + '/' + block_name + '/Variable_1'
    conv_tensor_name3 = 'fusion/fusion_model/res_layer' + str(stage) + '/' + block_name + '/Variable_2'
    conv_tensor_name_short_cut = 'fusion/fusion_model/res_layer' + str(stage) + '/' + block_name + '/Variable_3'
    conv_weight_name1 = 'conv_w'+str(stage)+'_'+block+'1'
    conv_weight_name2 = 'conv_w' + str(stage) + '_' + block + '2'
    conv_weight_name3 = 'conv_w' + str(stage) + '_' + block + '3'
    conv_weight_nameshort_cut = 'conv_w' + str(stage) + '_' + block + '4'
    # W_conv1 = weight_variable([1, 1, in_filter, f1])
    W_conv1 = tf.get_variable(conv_weight_name1, initializer=tf.constant(reader.get_tensor(conv_tensor_name1)))
    
    X = tf.nn.conv2d(X_input, W_conv1, strides=[1, stride, stride, 1], padding='VALID')
    X = tf.layers.batch_normalization(X, axis=3, training=training)
    X = tf.nn.relu(X)

    # second
    # W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
    W_conv2 = tf.get_variable(conv_weight_name2, initializer=tf.constant(reader.get_tensor(conv_tensor_name2)))
    X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    X = tf.layers.batch_normalization(X, axis=3, training=training)
    X = tf.nn.relu(X)

    # third
    # W_conv3 = weight_variable([1, 1, f2, f3])
    W_conv3 = tf.get_variable(conv_weight_name3, initializer=tf.constant(reader.get_tensor(conv_tensor_name3)))
    X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
    X = tf.layers.batch_normalization(X, axis=3, training=training)

    # shortcut path
    # W_shortcut = weight_variable([1, 1, in_filter, f3])
    W_shortcut = tf.get_variable( conv_weight_nameshort_cut,
                                  initializer=tf.constant(reader.get_tensor( conv_tensor_name_short_cut)))
    x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

    # final
    add = tf.add(x_shortcut, X)
    add_result = tf.nn.relu(add)

  return add_result
# 新设置的层 结束


def input_setup(index):
    padding=6
    sub_ir_sequence = []
    sub_vi_sequence = []
    input_ir=(imread(data_ir[index])-127.5)/127.5
    input_ir=np.lib.pad(input_ir,((padding,padding),(padding,padding)),'edge')
    w,h=input_ir.shape
    input_ir=input_ir.reshape([w,h,1])
    input_vi=(imread(data_vi[index])-127.5)/127.5
    input_vi=np.lib.pad(input_vi,((padding,padding),(padding,padding)),'edge')
    w,h=input_vi.shape
    input_vi=input_vi.reshape([w,h,1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir= np.asarray(sub_ir_sequence)
    train_data_vi= np.asarray(sub_vi_sequence)
    return train_data_ir,train_data_vi


num_epoch=49
# while num_epoch <= 9:
if num_epoch == 49:
    reader = tf.train.NewCheckpointReader('./checkpoint_20/CGAN_120/CGAN.model-'+ str(num_epoch))
    # reader = tf.train.NewCheckpointReader("./checkpoint/CGAN_120/CGAN.model-9")
    with tf.name_scope('IR_input'):
        #红外图像patch
        images_ir = tf.placeholder(tf.float32, [1,None,None,None], name='images_ir')
    with tf.name_scope('VI_input'):
        #可见光图像patch
        images_vi = tf.placeholder(tf.float32, [1,None,None,None], name='images_vi')
        #labels_vi_gradient=gradient(labels_vi)
    #将红外和可见光图像在通道方向连起来，第一通道是红外图像，第二通道是可见光图像
    with tf.name_scope('input'):
        #resize_ir=tf.image.resize_images(images_ir, (512, 512), method=2)
        input_image=tf.concat([images_ir,images_vi],axis=-1)
    with tf.name_scope('fusion'):
        fusion_image=fusion_model(input_image)

    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        data_ir=prepare_data('Test_ir')
        data_vi=prepare_data('Test_vi')
        for i in range(len(data_ir)):
            start=time.time()
            train_data_ir,train_data_vi=input_setup(i)
            result =sess.run(fusion_image,feed_dict={images_ir: train_data_ir,images_vi: train_data_vi})
            result=result*127.5+127.5
            result = result.squeeze()
            image_path = os.path.join(os.getcwd(), 'result','epoch'+str(num_epoch))
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            if i<=9:
                image_path = os.path.join(image_path,'F9_0'+str(i)+".png")
            else:
                image_path = os.path.join(image_path,'F9_' +str(i)+".png")
            end=time.time()
            # print(out.shape)
            imsave(result, image_path)
            print("Testing [%d] success,Testing time is [%f]"%(i,end-start))
    tf.reset_default_graph()
    num_epoch=num_epoch+1
