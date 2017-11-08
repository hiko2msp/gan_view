import pickle
from PIL import Image
import os
from io import BytesIO
import math
import base64

import tensorflow as tf

import numpy as np
from PIL import Image


w1 = tf.Variable(tf.random_normal([4, 4, 12, 32], mean=0.0, stddev=0.05), dtype=tf.float32)
scale2 = tf.Variable(tf.ones([32]))
beta2 = tf.Variable(tf.zeros([32]))
w3 = tf.Variable(tf.random_normal([4, 4, 32, 64], mean=0.0, stddev=0.05), dtype=tf.float32)
scale4 = tf.Variable(tf.ones([64]))
beta4 = tf.Variable(tf.zeros([64]))
w5 = tf.Variable(tf.random_normal([4, 4, 32, 64], mean=0.0, stddev=0.05), dtype=tf.float32)
scale6 = tf.Variable(tf.ones([64]))
beta6 = tf.Variable(tf.zeros([64]))
w7 = tf.Variable(tf.random_normal([4, 4, 3, 64], mean=0.0, stddev=0.05), dtype=tf.float32)

def leaky_relu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def generator(z, keep_prob):
    #layer1 convolution
    conv1 = tf.nn.conv2d(z, w1, strides=[1, 2, 2, 1], padding="SAME")
    conv1_drop = tf.nn.dropout(conv1, keep_prob)

    #relu function
    conv1_lrelu = leaky_relu(conv1_drop, alpha=0.2)

    #layer2 batch normalization
    batch_mean2, batch_var2 = tf.nn.moments(conv1_lrelu, [0, 1, 2])
    bn2 = tf.nn.batch_normalization(conv1_lrelu, batch_mean2, batch_var2, beta2, scale2 , 0.001)

    #layer3 convolution
    conv3 = tf.nn.conv2d(bn2, w3, strides=[1, 2, 2, 1], padding="SAME")

    #Leaky_relu function
    conv3_lrelu = leaky_relu(conv3, alpha=0.2)

    #layer4 batch normalization
    batch_mean4, batch_var4 = tf.nn.moments(conv3_lrelu, [0, 1, 2])
    bn4 = tf.nn.batch_normalization(conv3_lrelu, batch_mean4, batch_var4, beta4, scale4 , 0.001)

    #layer5 deconvolution
    batch_size5 = tf.shape(bn4)[0]
    deconv5_shape = tf.stack([batch_size5, 14, 14, 32])
    deconv5 = tf.nn.conv2d_transpose(bn4, w5, deconv5_shape, strides=[1, 2, 2, 1], padding='SAME')

    # concat [bn2, deconv5]
    concat25 = tf.concat([bn2, deconv5], axis=3)

    #layer6 batch normalization
    batch_mean6, batch_var6 = tf.nn.moments(concat25, [0, 1, 2])
    bn6 = tf.nn.batch_normalization(concat25, batch_mean6, batch_var6, beta6, scale6 , 0.001)

    #Leaky_relu function
    bn6_lrelu = leaky_relu(bn6, alpha=0.2)

    #layer7 decovnolution
    batch_size7 = tf.shape(bn6_lrelu)[0]
    deconv7_shape = tf.stack([batch_size7, 28, 28, 3])
    deconv7 = tf.nn.conv2d_transpose(bn6_lrelu, w7, deconv7_shape, strides=[1, 2, 2, 1], padding='SAME')

    return deconv7


z_gen = tf.placeholder(tf.float32, [None, 28, 28, 12])

gen = generator(z_gen, 1)

saver = tf.train.Saver()

def clip_img(x):
	return np.float32(-1 if x<-1 else (1 if x>1 else x))

def load_image(filepath):
    img = Image.open(filepath)
    return img

from collections import Counter
def segment(img):
    img = img.resize((28, 28), Image.BILINEAR)
    img_array = np.asarray(img) - 1
    label = np.zeros((12, 28, 28)).astype("i")
    for j in range(12):
        label[j,:] = img_array == j
    zc, zh, zw = label.shape
    label = label.transpose(1, 2, 0)
    z = label.reshape(-1, zh, zw, zc).astype(np.float32)

    with tf.Session() as sess:
        # Load Model
        saver.restore(sess, 'models/model.ckpt-420')
        # Generate Image from Input Image
        x = sess.run(gen, feed_dict={z_gen: z})

    x = (x + 1) * 128 - 1
    x = x.astype(np.uint8)
    xb, xh, xw, xc = x.shape
    x = x.reshape(xh, xw, xc).astype(np.uint8)
    return Image.fromarray(x)


def to_base_64(image):
    output = BytesIO()
    image.save(output, format='PNG')
    return base64.b64encode(output.getvalue()).decode()

if __name__ == '__main__':
    img = load_image('images/facade.png')
    segmented_img = segment(img)
    segmented_img.save('images/result.png')
