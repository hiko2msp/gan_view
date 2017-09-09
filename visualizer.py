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

class Unet128Generator(chainer.Chain):
    def __init__(self):
        ngf = 32
        input_num_channel = 3
        label_num_channel = 2
        super(Unet128Generator, self).__init__(
            # 128 x 128
            c1=L.Convolution2D(input_num_channel, ngf, 4, stride=2, pad=1),
            # 64 x 64
            c2=L.Convolution2D(ngf, ngf * 2, 4, stride=2, pad=1),
            # 32 x 32
            c3=L.Convolution2D(ngf * 2, ngf * 4, 4, stride=2, pad=1),
            # 16 x 16
            c4=L.Convolution2D(ngf * 4, ngf * 8, 4, stride=2, pad=1),
            # 8 x 8
            c5=L.Convolution2D(ngf * 8, ngf * 8, 4, stride=2, pad=1),
            # 4 x 4
            c6=L.Convolution2D(ngf * 8, ngf * 8, 4, stride=2, pad=1),
            # 2 x 2
            c7=L.Convolution2D(ngf * 8, ngf * 8, 4, stride=2, pad=1),
            # 1 x 1
            d1=L.Deconvolution2D(ngf * 8, ngf * 8, 4, stride=2, pad=1),
            d2=L.Deconvolution2D(ngf * 8 * 2, ngf * 8, 4, stride=2, pad=1),
            d3=L.Deconvolution2D(ngf * 8 * 2, ngf * 8, 4, stride=2, pad=1),
            d4=L.Deconvolution2D(ngf * 8 * 2, ngf * 4, 4, stride=2, pad=1),
            d5=L.Deconvolution2D(ngf * 4 * 2, ngf * 2, 4, stride=2, pad=1),
            d6=L.Deconvolution2D(ngf * 2 * 2, ngf, 4, stride=2, pad=1),
            d7=L.Deconvolution2D(ngf * 1 * 2, label_num_channel, 4, stride=2, pad=1),
            bn1=L.BatchNormalization(ngf * 2),
            bn2=L.BatchNormalization(ngf * 4),
            bn3=L.BatchNormalization(ngf * 8),
            bn4=L.BatchNormalization(ngf * 8),
            bn5=L.BatchNormalization(ngf * 8),
            bn6=L.BatchNormalization(ngf * 8),
            bn7=L.BatchNormalization(ngf * 8),
            bn8=L.BatchNormalization(ngf * 8),
            bn9=L.BatchNormalization(ngf * 8),
            bn10=L.BatchNormalization(ngf * 4),
            bn11=L.BatchNormalization(ngf * 2),
            bn12=L.BatchNormalization(ngf * 1),
        )

    def __call__(self, z, test=False):
        h1 =                     self.c1(z)
        h2 =            self.bn1(self.c2(F.leaky_relu(h1, 0.2)), test=test)
        h3 =            self.bn2(self.c3(F.leaky_relu(h2, 0.2)), test=test)
        h4 =            self.bn3(self.c4(F.leaky_relu(h3, 0.2)), test=test)
        h5 =            self.bn4(self.c5(F.leaky_relu(h4, 0.2)), test=test)
        h6 =            self.bn5(self.c6(F.leaky_relu(h5, 0.2)), test=test)
        h7 =                     self.c7(F.leaky_relu(h6, 0.2))
        h8 =  F.dropout(self.bn7(self.d1(F.relu(h7)), test=test), 0.5)
        h8 =  F.concat((h8, h6))
        h9 =  F.dropout(self.bn8(self.d2(F.relu(h8)), test=test), 0.5)
        h9 =  F.concat((h9, h5))
        h10 = F.dropout(self.bn9(self.d3(F.relu(h9)), test=test), 0.5)
        h10 =  F.concat((h10, h4))
        h11 =           self.bn10(self.d4(F.relu(h10)), test=test)
        h11 =  F.concat((h11, h3))
        h12 =           self.bn11(self.d5(F.relu(h11)), test=test)
        h12 =  F.concat((h12, h2))
        h13 =           self.bn12(self.d6(F.relu(h12)), test=test)
        h13 =  F.concat((h13, h1))
        h14 =              F.tanh(self.d7(F.relu(h13)))
        return h14

def clip_img(x):
	return np.float32(-1 if x<-1 else (1 if x>1 else x))

gen = Unet128Generator()
gen.to_cpu()



def load_image(filepath):
    img = Image.open(filepath)
    return img

def segment(img):
    model_file = 'model.h5'
    serializers.load_hdf5(model_file, gen)
    img = img.resize((128, 128), Image.BILINEAR)
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
