import tensorflow as tf
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from dropblock import DropBlock3D


def condition(X):

	return X + tf.eye(tf.shape(X)[0]) * 1e-3


def propagate_last(l, num_layer_same_scale, input_prev_layer, num_stride, dim_filter, num_filters, 
	padding, unet_type, mode, keep_prob, convolution_type, deconvolution_shape=None):

	with tf.compat.v1.variable_scope( str(convolution_type)+'_layer_'+str(l)+'_dim_'+str(num_layer_same_scale),reuse=tf.compat.v1.AUTO_REUSE):

		if unet_type=='3D':

			input_prev_layer = tf.compat.v1.layers.conv3d(inputs=input_prev_layer,
				filters=num_filters,
				kernel_size=(dim_filter, dim_filter, dim_filter),
				strides=(num_stride, num_stride, num_stride),
				padding=padding,
				data_format='channels_last',
				dilation_rate=(1, 1, 1),
				activation=None,
				use_bias=False,
				kernel_initializer=tf.initializers.variance_scaling(),
				bias_initializer=tf.zeros_initializer(),
				kernel_regularizer=tf.keras.regularizers.l2(1e-4),
				bias_regularizer=tf.keras.regularizers.l2(1e-4),
				activity_regularizer=None,
				kernel_constraint=None,
				bias_constraint=None,
				trainable=True,
				name='conv3d_layer',
				reuse=tf.compat.v1.AUTO_REUSE
				)

		else:

			input_prev_layer = tf.layers.conv2d(inputs=input_prev_layer,
				filters=num_filters,
				kernel_size=(dim_filter, dim_filter),
				strides=(num_stride, num_stride),
				padding=padding,
				data_format='channels_last',
				dilation_rate=(1, 1),
				activation=None,
				use_bias=False,
				kernel_initializer=tf.initializers.variance_scaling(),
				bias_initializer=tf.zeros_initializer(),
				kernel_regularizer=tf.keras.layers.l2_regularizer(1e-4),
				bias_regularizer=None,
				activity_regularizer=None,
				kernel_constraint=None,
				bias_constraint=None,
				trainable=True,
				name='conv2d_layer',
				reuse=tf.AUTO_REUSE
				)

	return input_prev_layer


class SwinTransformerBlock3D(layers.Layer):
    def __init__(self, dim, heads, window_size, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(num_heads=self.heads, key_dim=self.dim)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = keras.Sequential([
            layers.Dense(int(self.dim * self.mlp_ratio)),
            layers.Activation('gelu'),
            layers.Dense(self.dim),
        ])

    def split_windows(self, x):
        B, D, H, W, C = x.shape
        x = tf.reshape(x, [B, D // self.window_size, self.window_size,
                           H // self.window_size, self.window_size,
                           W // self.window_size, self.window_size, C])
        x = tf.transpose(x, [0, 1, 3, 5, 2, 4, 6, 7])
        return tf.reshape(x, [-1, self.window_size ** 3, C])

    def merge_windows(self, x, B, D, H, W):
        x = tf.reshape(x, [B, D // self.window_size, H // self.window_size, W // self.window_size,
                           self.window_size, self.window_size, self.window_size, -1])
        x = tf.transpose(x, [0, 1, 4, 2, 5, 3, 6, 7])
        return tf.reshape(x, [B, D, H, W, -1])

    def forward(self, x):
        B, D, H, W, C = x.shape
        x = self.norm1(x)
        x = self.split_windows(x)
        x = self.attn(x, x, x)
        x = self.merge_windows(x, B, D, H, W)
        x = x + x
        x = self.norm2(x)
        x = self.mlp(x)
        return x

def propagate_dropout(l, num_layer_same_scale, input_prev_layer, num_stride, dim_filter, num_filters, 
	padding, unet_type, mode, rate, convolution_type, deconvolution_shape=None,Swin=True):

	swin_transformer_block = SwinTransformerBlock3D(dim=num_filters, heads=8, window_size=dim_filter)

	with tf.compat.v1.variable_scope( str(convolution_type)+'_layer_'+str(l)+'_dim_'+str(num_layer_same_scale), reuse=tf.compat.v1.AUTO_REUSE):


		if unet_type=='3D':

			if convolution_type=='upsampling':

				input_prev_layer = tf.compat.v1.layers.conv3d_transpose(
					inputs=input_prev_layer,
					filters=num_filters,
					kernel_size=(dim_filter, dim_filter, dim_filter),
					strides=(num_stride, num_stride, num_stride),
					padding=padding,
					data_format='channels_last',
					activation=None,
					use_bias=False,
					kernel_initializer=tf.initializers.variance_scaling(),
					bias_initializer=tf.zeros_initializer(),
					kernel_regularizer=tf.keras.regularizers.l2(1e-4),
					bias_regularizer=None,
					activity_regularizer=None,
					kernel_constraint=None,
					bias_constraint=None,
					trainable=True,
					name='deconv3d_layer',
					reuse=tf.compat.v1.AUTO_REUSE)


			else:
				input_prev_layer = tf.compat.v1.layers.conv3d(inputs=input_prev_layer,
															  filters=num_filters,
															  kernel_size=(dim_filter, dim_filter, dim_filter),
															  strides=(num_stride, num_stride, num_stride),
															  padding=padding,
															  data_format='channels_last',
															  dilation_rate=(1, 1, 1),
															  activation=None,
															  use_bias=False,
															  kernel_initializer=tf.initializers.variance_scaling(),
															  bias_initializer=tf.zeros_initializer(),
															  kernel_regularizer=tf.keras.regularizers.l2(1e-4),
															  bias_regularizer=None,
															  activity_regularizer=None,
															  kernel_constraint=None,
															  bias_constraint=None,
															  trainable=True,
															  name='conv3d_layer',
															  reuse=tf.compat.v1.AUTO_REUSE)

		else:
			raise ValueError("Only 3D MRI images are supported at the moment.")

		if Swin:
			input_prev_layer = swin_transformer_block(input_prev_layer)


		with tf.compat.v1.variable_scope('squeeze_excite_block', reuse=tf.compat.v1.AUTO_REUSE):

			se = tf.reduce_mean(input_prev_layer, axis=[1, 2, 3], keepdims=False)
			se = tf.compat.v1.layers.dense(inputs=se, units=num_filters / 8,
										   activation=None,
										   use_bias=False,
										   kernel_initializer=tf.initializers.variance_scaling(),
										   bias_initializer=tf.zeros_initializer(),
										   kernel_regularizer=tf.keras.regularizers.l2(1e-4),
										   trainable=True,
										   name='se_dense_layer_1',
										   reuse=tf.compat.v1.AUTO_REUSE)
			se = tf.nn.leaky_relu(features=se, alpha=0.1, name='leaky_relu')
			se = tf.compat.v1.layers.dense(inputs=se, units=num_filters,
										   activation=None, use_bias=False,
										   kernel_initializer=tf.initializers.variance_scaling(),
										   bias_initializer=tf.zeros_initializer(),
										   kernel_regularizer=tf.keras.regularizers.l2(1e-4),
										   trainable=True, name='se_dense_layer_2',
										   reuse=tf.compat.v1.AUTO_REUSE)
			se = tf.math.sigmoid(se)
			se = tf.reshape(se, [tf.shape(se)[0], 1, 1, 1, tf.shape(se)[1]])
		input_prev_layer = input_prev_layer * se

		batch_size = tf.shape(input_prev_layer)[0]
		input_prev_layer = tf.nn.dropout(
			x=input_prev_layer,
			rate=1 - rate,
			noise_shape=[batch_size, 1, 1, 1, input_prev_layer.get_shape().as_list()[-1]])

		input_prev_layer = tf.nn.leaky_relu(features=input_prev_layer, alpha=0.1, name='leaky_relu')

	return input_prev_layer
