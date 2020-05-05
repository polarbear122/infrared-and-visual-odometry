import os
# import cv2
import tensorflow as tf

# dire = 'result'
# data_dir_ir = os.path.join('./{}'.format(dire), "epoch3", "F9_00.bmp")
# # print(data_dir_ir)
# i = cv2.imread(data_dir_ir)
from tensorflow.python import pywrap_tensorflow
# import os

# for line in open('./result/new_net_train.txt', "r"):
# 	n = 0
# 	for i in line:
# 		n+=1
# 		if i == '_':
# 			print(n)


model_dir = './checkpoint/CGAN_120/'
checkpoint_path = os.path.join(model_dir, "model.ckpt")
reader = tf.train.NewCheckpointReader('./checkpoint/CGAN_120/CGAN.model-'+ str(9))
# checkpoint_path = './checkpoint_20/CGAN_120/CGAN.model-9'
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
	print("tensor_name: ", key)

# a = tf.constant([[[[1,2]]]])
# print(a)
# print(?, 120, 120, 1)
# print(i)
