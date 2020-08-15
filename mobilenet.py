# -*- coding: utf-8 -*-
# @Time    : 8/14/20 1:32 PM
# @Author  : Zeqi@@
# @FileName: mobilenet.py.py
# @Software: PyCharm
"""
    MobileNetV2 is a general architecture and can be used for multiple use cases.
    Depending on the use case, it can use different input layer size and
    different width factors. This allows different width models to reduce
    the number of multiply-adds and thereby
    reduce inference cost on mobile devices.
    MobileNetV2 is very similar to the original MobileNet,
    except that it uses inverted residual blocks with
    bottlenecking features. It has a drastically lower
    parameter count than the original MobileNet.
    MobileNets support any input size greater
    than 32 x 32, with larger image sizes
    offering better performance.
    The number of parameters and number of multiply-adds
    can be modified by using the `alpha` parameter,
    which increases/decreases the number of filters in each layer.
    By altering the image size and `alpha` parameter,
    all 22 models from the paper can be built, with ImageNet weights provided.
    The paper demonstrates the performance of MobileNets using `alpha` values of
    1.0 (also called 100 % MobileNet), 0.35, 0.5, 0.75, 1.0, 1.3, and 1.4
    For each of these `alpha` values, weights for 5 different input image sizes
    are provided (224, 192, 160, 128, and 96).
    The following table describes the performance of
    MobileNet on various input sizes:
    ------------------------------------------------------------------------
    MACs stands for Multiply Adds
     Classification Checkpoint| MACs (M) | Parameters (M)| Top 1 Accuracy| Top 5 Accuracy
    --------------------------|------------|---------------|---------|----|-------------
    | [mobilenet_v2_1.4_224]  | 582 | 6.06 |          75.0 | 92.5 |
    | [mobilenet_v2_1.3_224]  | 509 | 5.34 |          74.4 | 92.1 |
    | [mobilenet_v2_1.0_224]  | 300 | 3.47 |          71.8 | 91.0 |
    | [mobilenet_v2_1.0_192]  | 221 | 3.47 |          70.7 | 90.1 |
    | [mobilenet_v2_1.0_160]  | 154 | 3.47 |          68.8 | 89.0 |
    | [mobilenet_v2_1.0_128]  | 99  | 3.47 |          65.3 | 86.9 |
    | [mobilenet_v2_1.0_96]   | 56  | 3.47 |          60.3 | 83.2 |
    | [mobilenet_v2_0.75_224] | 209 | 2.61 |          69.8 | 89.6 |
    | [mobilenet_v2_0.75_192] | 153 | 2.61 |          68.7 | 88.9 |
    | [mobilenet_v2_0.75_160] | 107 | 2.61 |          66.4 | 87.3 |
    | [mobilenet_v2_0.75_128] | 69  | 2.61 |          63.2 | 85.3 |
    | [mobilenet_v2_0.75_96]  | 39  | 2.61 |          58.8 | 81.6 |
    | [mobilenet_v2_0.5_224]  | 97  | 1.95 |          65.4 | 86.4 |
    | [mobilenet_v2_0.5_192]  | 71  | 1.95 |          63.9 | 85.4 |
    | [mobilenet_v2_0.5_160]  | 50  | 1.95 |          61.0 | 83.2 |
    | [mobilenet_v2_0.5_128]  | 32  | 1.95 |          57.7 | 80.8 |
    | [mobilenet_v2_0.5_96]   | 18  | 1.95 |          51.2 | 75.8 |
    | [mobilenet_v2_0.35_224] | 59  | 1.66 |          60.3 | 82.9 |
    | [mobilenet_v2_0.35_192] | 43  | 1.66 |          58.2 | 81.2 |
    | [mobilenet_v2_0.35_160] | 30  | 1.66 |          55.7 | 79.1 |
    | [mobilenet_v2_0.35_128] | 20  | 1.66 |          50.8 | 75.0 |
    | [mobilenet_v2_0.35_96]  | 11  | 1.66 |          45.5 | 70.4 |
"""
import cv2
import time
import numpy as np
import tensorflow
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras import backend as K


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 2 if K.image_data_format() == 'channels_first' else 1
    input_size = K.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

def _make_divisible(v, divisor, layer_name, min_value=None):
    '''It ensures that all layers have a channel number that is divisible by 8'''
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    # print('{3}, Original v: {0}, divisor: {1}, new_v: {2}'.format(v, divisor, new_v, layer_name))
    return new_v



def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    """
     separable_conv2d
        3*3 depyhwise conv + bn + relu6  +  1*1 pointwise conv + bn + relu6

     expanded_conv
        1*1 expansion + depthwise + 1*1 projection
    """
    prefix = 'block_{}_'.format(block_id)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    in_channels = K.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8, layer_name=prefix)
    x = inputs


    if block_id:
        # block_id is 0, keep dimension, expansion = 1
        # block_id is not 0, raise dimension, expansion = 6
        # Expand
        x = Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand')(x)
        x = BatchNormalization(axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand_BN')(x)
        x = ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(x, 3),
                                 name=prefix + 'pad')(x)
    x = DepthwiseConv2D(kernel_size=3,
                       strides=stride,
                       activation=None,
                       use_bias=False,
                       padding='same' if stride == 1 else 'valid',
                       name=prefix + 'depthwise')(x)
    x = BatchNormalization(axis=channel_axis,
                          epsilon=1e-3,
                          momentum=0.999,
                          name=prefix + 'depthwise_BN')(x)

    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
              kernel_size=1,
              padding='same',
              use_bias=False,
              activation=None,
              name=prefix + 'project')(x)
    x = BatchNormalization(axis=channel_axis,
                          epsilon=1e-3,
                          momentum=0.999,
                          name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x

def MobileNetV2(input_shape,
                alpha=1.0,
                include_top=True,
                weights='imagenet',
                classes=1000,
                **kwargs):
    """Instantiates the MobileNetV2 architecture.
    alpha: Expand the basic model and generate a banch of models, but here we use alpha = 1 only
    # Returns
        A Keras model instance.
    """
    rows = input_shape[0]
    img_input = Input(shape=input_shape)

    # 
    # 224, 224, 3 -> 112, 112, 32
    first_block_filters = _make_divisible(32 * alpha, 8, layer_name='First_block_filters')
    # After 3*3 convolution, height/width is half of the previous
    x = ZeroPadding2D(padding=correct_pad(img_input, 3),
                             name='Conv1_pad')(img_input)
    x = Conv2D(first_block_filters,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,
                      name='Conv1')(x)
    x = BatchNormalization(epsilon=1e-3,
                           momentum=0.999,
                           name='bn_Conv1')(x)
    x = ReLU(6., name='Conv1_relu')(x)

    # 112, 112, 32 -> 112, 112, 16
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0)
    # 112, 112, 16 -> 56, 56, 24
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2)

    #  56, 56, 24 -> 28, 28, 32
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5)

    #  28, 28, 32 -> 14, 14, 64
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                            expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=9)

    #  14, 14, 64 -> 14, 14, 96
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=12)

    #  14, 14, 96 -> 7, 7, 160
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                            expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=15)
    #  7, 7, 160 -> 7, 7, 320
    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                            expansion=6, block_id=16)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8, layer_name='alpha>1')
    else:
        last_block_filters = 1280

    #  7, 7, 320 -> 7, 7, 1280
    x = Conv2D(last_block_filters,
              kernel_size=1,
              use_bias=False,
              name='Conv_1')(x)
    x = BatchNormalization(epsilon=1e-3,
                            momentum=0.999,
                            name='Conv_1_bn')(x)
    x = ReLU(6., name='out_relu')(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes, activation='softmax',
                    use_bias=True, name='Logits')(x)
    else:
        x = GlobalAveragePooling2D()(x)
        x = Conv2D(classes,
                   kernel_size=1,
                   use_bias=False,
                   name='Final_conv')(x)

    # Create model.
    model = Model(img_input, x,
                name='mobilenetv2_%0.2f_%s' % (alpha, rows))

    return model

if __name__=='__main__':
    # Read imagenet labals
    f = open("../imagenet_names.txt")
    lines = f.readlines()
    names = []
    for line in lines:
        print(line.split(':')[-1])
        names.append(line.split(':')[-1])
    f.close()

    img = cv2.imread('../images/panda.jpg')
    img = cv2.resize(img, (224, 224))
    cv2.imshow('fox', img)
    cv2.waitKey(100)
    input_image_shape = img.shape
    print(input_image_shape)
  
    model = MobileNetV2(input_image_shape, alpha=1, nclude_top=True, weights='imagenet',classes=1000)

    tf_model = tensorflow.keras.applications.MobileNetV2(
    input_shape=input_image_shape, alpha=1.0, include_top=True, weights='imagenet',
    input_tensor=None, pooling=None, classes=1000, classifier_activation='softmax')
    mobilenetv2_1_224 = tf_model.get_weights()

    model.set_weights(mobilenetv2_1_224)
    start = time.time()
    test_img = np.expand_dims(img, axis=0)
    pred = tf_model.predict(test_img)
    end = time.time()
    pred_argmax = pred[0].argmax()
    print('Prediction: {0} Probability: {1}'.format(names[pred_argmax - 1], pred[0][pred_argmax]))
    print('Execution time: {0} ms'.format(np.round((end-start)*1000),2))




