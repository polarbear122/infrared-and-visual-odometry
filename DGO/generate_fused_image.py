import tensorflow as tf
import numpy as np
from generate_model.des_gen_densefuse_net import DenseFuseNet
from scipy.misc import imread, imsave, imresize
from os.path import join, exists, splitext
from os import listdir, mkdir, sep
import matplotlib.pyplot as plt
import imageio
import cv2
from PIL import Image
import os
import scipy
number_of_images = 1
import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
def _handler(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path='./result'):
	for frame_id in range(number_of_images):
		ir_path = './Test_ir/'+str(frame_id)+'.png'
		vis_path = './Test_vi/'+str(frame_id)+'.png'
		ir_img = get_train_images(ir_path, flag=False)  # 获得原始图像
		vis_img = get_train_images(vis_path, flag=False)
		
		dimension = ir_img.shape
		# print('ir_img:{}'.format(ir_img.shape))
		ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])  # 更改图像shape
		vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])
		# print('ir_img:{}'.format(ir_img.shape))
		# ir_img = np.transpose(ir_img, (0, 2, 1, 3))  # 再次更改图像 shape
		# vis_img = np.transpose(vis_img, (0, 2, 1, 3))
		# print('ir_img:{}'.format(ir_img.shape))
		# print('img shape final:{}'.format(ir_img.shape))
	
		with tf.Graph().as_default(), tf.Session() as sess:
			# 将图像变成张量形式
			infrared_field = tf.placeholder(
				tf.float32, shape=ir_img.shape, name='content')
			visible_field = tf.placeholder(
				tf.float32, shape=ir_img.shape, name='style')
	
			dfn = DenseFuseNet(model_pre_path)
			# print('dfn:{}'.format(dfn))
			output_image = dfn.transform_addition(infrared_field, visible_field)
			# print('infrared_field:{}'.format(infrared_field))
			# print('visible_field:{}'.format(visible_field))
			
			# restore the trained model and run the style transferring
			saver = tf.train.Saver()
			saver.restore(sess, model_path)
			# output_image为张量，output为numpy数组  shape均为(1, 1200, 900, 1)
			output = sess.run(output_image, feed_dict={infrared_field: ir_img, visible_field: vis_img})
			# print(output.shape)
			out_shape = output.shape
			save_path = './result/'+str(frame_id)+'.png'
			output_reshape = output[0,:,:,0]
			# im = Image.fromarray(output)
			# plt.imsave('name.png', output_reshape)
			# im.save("your_file.jpeg")
			# print('output_reshape:{}'.format(output_reshape.shape))
			scipy_imsave(output_reshape,save_path)
			print('successfully! the {} frame'.format(frame_id))
			# imageio.imwrite(save_path, output_reshape)
			#
			# save_images(ir_path, output, output_path,
			#                   prefix='fused' + str(frame_id), suffix=None)

		
def scipy_imsave(image, path):
		return scipy.misc.imsave(path, image)

def get_train_images(paths, resize_len=512, crop_height=900, crop_width=1200, flag=True):
		if isinstance(paths, str):
				paths = [paths]
		images = []
		# print('paths:{}'.format(paths))
		for path in paths:
				image = get_image(path, height=crop_height, width=crop_width, set_mode='L')
				# plt.imshow(image)
				# plt.show()
				# print('path:{}'.format(path))
				# print('image:{}'.format(image.shape))
				if flag:
						image = np.stack(image, axis=0)
						image = np.stack((image, image, image), axis=-1)
				else:
						image = np.stack(image, axis=0)
						image = image.reshape([crop_height, crop_width, 1])
						# print('image:{}'.format(image.shape))
				images.append(image)
		images = np.stack(images, axis=-1)
		
		# print('line65:images:{}'.format(images.shape))
		return images

def get_image(path, height=900, width=1200, set_mode='L'):
		image = imread(path, mode=set_mode)
		if height is not None and width is not None:
				image = imresize(image, [height, width], interp='nearest')
		return image
	
def save_images(paths, datas, save_path, prefix=None, suffix=None):
	if isinstance(paths, str):
		paths = [paths]
	# print('datas.shape:{}'.format(datas.shape))
	t1 = len(paths)
	t2 = len(datas)
	assert (len(paths) == len(datas))
	
	if prefix is None:
		prefix = ''
	if suffix is None:
		suffix = ''
	
	for i, path in enumerate(paths):
		data = datas[i]
		# print('data.shape:{}'.format(data.shape))
		# print('data.shape[0]:{}'.format(data.shape[0]))
		# print('data.shape[1]:{}'.format(data.shape[1]))
		# print('data.shape[2]:{}'.format(data.shape[2]))
		# print('data ==>>\n', data)
		if data.shape[2] == 1:
			data = data.reshape([data.shape[0], data.shape[1]])
			# 图像向左旋转了90度
			# print('data reshape==>>\n', data)
		# print('line 113 data:{}'.format(data.shape))
		plt.imshow(data)
		plt.show()
		name, ext = splitext(path)
		name = name.split(sep)[-1]
		# print('save_path:{}'.format(save_path))
		save_path = './result/'
		path = join(save_path, prefix + suffix + ext)
		# print('data path==>>', path)
		# new_im = Image.fromarray(data)
		# new_im.show()
		imsave(path, data)
		
if __name__ == '__main__':
	ir_path = None
	vis_path = None
	model_path = './checkpoint/CGAN_120/CGAN.model-9'
	model_pre_path = None
	ssim_weight = 1000
	index = 1
	_handler(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index)