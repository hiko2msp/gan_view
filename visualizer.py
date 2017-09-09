import pickle
from PIL import Image
import os
from io import BytesIO
import math
import base64

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer.utils import type_check
from chainer import function

import chainer.functions as F
import chainer.links as L

import numpy as np
from PIL import Image

class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            c1s=L.Convolution2D(3, 32, 4, stride=2, pad=1),#28x28 to 14x14
            c2s=L.Convolution2D(32, 64, 4, stride=2, pad=1),#14x14 to 7x7
            dc1s=L.Deconvolution2D(64, 32, 4, stride=2, pad=1),
            dc2s=L.Deconvolution2D(64, 2, 4, stride=2, pad=1),
            bn2s=L.BatchNormalization(32),
            bn3s=L.BatchNormalization(64),
            bn1s=L.BatchNormalization(64),
        )

    def __call__(self, z, test=False):
        h1 = self.c1s(z)
        h1 = F.leaky_relu(h1)
        h1 = self.bn2s(h1, test=test)
        h2 = self.c2s(h1)
        h2 = F.leaky_relu(h2)
        h2 = self.bn3s(h2, test=test)
        h2 = self.dc1s(h2)
        h = F.concat((h1, h2), axis=1)#32chan to 64chan
        h = self.bn1s(h, test=test)
        h = F.leaky_relu(h)
        x = F.sigmoid(self.dc2s(h))
        return x

def clip_img(x):
	return np.float32(-1 if x<-1 else (1 if x>1 else x))

gen = Generator()
gen.to_cpu()

def load_image(filepath):
    img = Image.open(filepath)
    return img

def segment(img):
    model_file = 'model.h5'
    serializers.load_hdf5(model_file, gen)
    img = img.resize((28, 28), Image.BILINEAR)
    image_array = (np.asarray(img).astype('f').transpose(2, 0, 1) + 1) / 128.0 - 1.0
    zc, zh, zw = image_array.shape
    z = image_array.reshape(-1, zc, zh, zw).astype(np.float32)
    x = gen(z, test=True)
    x = x.data
    img_array = ((np.vectorize(clip_img)(x[0,:,:,:])+1) * 128 - 1).transpose(1,2,0)
    return Image.fromarray(np.uint8(img_array))


def to_base_64(image):
    output = BytesIO()
    image.save(output, format='PNG')
    return base64.b64encode(output.getvalue()).decode()
