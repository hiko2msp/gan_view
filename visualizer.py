import pickle
from PIL import Image
import os
from io import BytesIO
import math
import base64
import tensorflow as tf

import numpy as np

first_layer = 14
w1 = tf.Variable(tf.random_normal([10, first_layer * first_layer * 32], mean=0.0, stddev=0.05), dtype=tf.float32)
scale2 = tf.Variable(tf.ones([first_layer * first_layer * 32]))
beta2 = tf.Variable(tf.zeros([first_layer * first_layer * 32]))
w3 = tf.Variable(tf.random_normal([4, 4, 32, 32], mean=0.0, stddev=0.05), dtype=tf.float32)
w4 = tf.Variable(tf.random_normal([4, 4, 32, 32], mean=0.0, stddev=0.05), dtype=tf.float32)
w5 = tf.Variable(tf.random_normal([4, 4, 3, 32], mean=0.0, stddev=0.05), dtype=tf.float32)

def generator(z):
    #fully-connected
    fc1 = tf.matmul(z, w1)

    #batch normalization
    batch_mean2, batch_var2 = tf.nn.moments(fc1, [0])
    bn2 = tf.nn.batch_normalization(fc1, batch_mean2, batch_var2, beta2, scale2 , 0.001)

    #reshape
    fc1_reshape = tf.reshape(bn2, [-1, first_layer, first_layer, 32])

    #deconvolotion
    deconv3 = tf.nn.conv2d_transpose(fc1_reshape, w3, [tf.shape(fc1_reshape)[0], 14, 14, 32], strides=[1, 1, 1, 1], padding='SAME')
    deconv4 = tf.nn.conv2d_transpose(deconv3, w4, [tf.shape(fc1_reshape)[0], 14, 14, 32], strides=[1, 1, 1, 1], padding='SAME')
    deconv5 = tf.nn.conv2d_transpose(deconv4, w5, [tf.shape(fc1_reshape)[0], 28, 28, 3], strides=[1, 2, 2, 1], padding='SAME')

    return deconv5


def clip_img(x):
	return np.float32(-1 if x<-1 else (1 if x>1 else x))

saver = tf.train.Saver()
z_ = tf.placeholder(tf.float32, [None, 10])
gen = generator(z_)

def gen_image(noise_list):
    z = np.array(noise_list).astype(np.float32).reshape((1, 10))
    z = (z - 50) / float(50)
    with tf.Session() as sess:
        saver.restore(sess, 'models/model.ckpt')
        x = sess.run(gen, feed_dict={z_: z})
    tmp = (np.vectorize(clip_img)(x[0,:,:,:])+1) / 2
    img_array = (tmp * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def gen_image_b64(noise_list):
    return to_base_64(gen_image(noise_list))

def to_base_64(img):
    output = BytesIO()
    img.save(output, format='PNG')
    return base64.b64encode(output.getvalue()).decode()

if __name__ == '__main__':
    noise = (np.random.uniform(-1, 1, (10)).astype(np.float32) + 1) / 2 * 100
    image = gen_image(noise)
    image.save('images/result.png')
