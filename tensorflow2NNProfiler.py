# -*- coding: utf-8 -*-
# @Time    : 2020/7/6 22:51
# @FileName: tensorflow2NNProfiler.py
# @Software: PyCharm

""" Copyright (c) GrAI Matter Labs SAS 2020. All rights reserved. """

import os
import argparse
import prettytable
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Output the number of model parameters, the number of consumed neurons and the operation numbers of the target model')
    parser.add_argument('--model_name', default='VGG16',
                        help="Set the name of the target model, e.g. ResNet50, VGG16, MobilenetV2")
    parser.add_argument('--model_path', default='models/example_model.h5',
                        help="Import the target model, e.g. example_model.h5")
    args = parser.parse_args()
    return args

def get_params(layer):
    weights = layer.get_weights()
    params = 0
    for w in weights:
        params += np.prod(np.shape(w))
        print(layer.__class__.__name__, np.shape(w))
    return params

def profile_conv2d(layer, _table):
    Bias_add = 1 if layer.use_bias else 0
    input_shape, output_shape = layer.input_shape, layer.output_shape
    nb_neurons_per_layer = np.prod(output_shape[1:])
    k = layer.kernel_size

    # MACs and FLOPs
    conv_flops = np.float64(input_shape[3]*k[0]*k[1] + Bias_add)
    output_shape_prod= np.float64(np.prod(output_shape[1:]))
    FLOPs = 2*conv_flops*output_shape_prod
    MACs = conv_flops* output_shape_prod

    # Parameters
    params = get_params(layer)

    first = 'block_{}'.format(layer.name.split('_')[1]) if layer.name.split('_')[0] == 'block' else layer.name
    _table.add_row([first,
                    layer.__class__.__name__,
                    nb_neurons_per_layer,
                    np.round(MACs/10**6, 2),
                    params,
                    str(input_shape[1:]),
                    str(output_shape[1:]),
                    str(k)])

    return nb_neurons_per_layer, MACs, params


def profile_depthwise_conv2d(layer, _table):
    Bias_add = 1 if layer.use_bias else 0
    input_shape, output_shape = layer.input_shape, layer.output_shape
    nb_neurons_per_layer = np.prod(output_shape[1:])
    k = layer.kernel_size

    #  MACs and FLOPs
    MACs = input_shape[-1]*k[0]*k[1]*output_shape[1]*output_shape[2]

    #  Parameters
    params = get_params(layer)

    _table.add_row(['block_{}'.format(layer.name.split('_')[1]),
                    layer.__class__.__name__,
                    nb_neurons_per_layer,
                    np.round(MACs / 10 ** 6, 2),
                    params,
                    str(input_shape[1:]),
                    str(output_shape[1:]),
                    str(k)])

    return nb_neurons_per_layer, MACs, params

def profile_dense(layer, _table):
    Bias_add = 1 if layer.use_bias else 0
    input_shape, output_shape = layer.input_shape, layer.output_shape
    nb_neurons_per_layer = np.prod(input_shape[1:])

    #  MACs and FLOPs
    # (2*I-1)*O # I is the input dimensionality and O is the output dimensitonality
    MACs = (input_shape[1] - 1 + Bias_add) * output_shape[1]

    #  Parameters
    params = get_params(layer)

    _table.add_row([layer.name,
                    layer.__class__.__name__,
                    nb_neurons_per_layer,
                    np.round(MACs / 10 ** 6, 2),
                    params,
                    str(input_shape[1:]),
                    str(output_shape[1:]),
                    '-'])
    return nb_neurons_per_layer, MACs, params

def profile_BatchNorm2d(layer, _table):
    # BN can be regarded as linear
    # Batch size is variate, ops is not fixed
    # For one input smaple, gamma*in + beta == 2*input_shape ops
    input_shape, output_shape = layer.input_shape, layer.output_shape
    nb_neurons_per_layer = np.prod(output_shape[1:])
    #  MACs and FLOPs
    MACs = 2 * np.prod(input_shape[1:])
    # Parameters
    params = get_params(layer)

    _table.add_row([layer.name,
                    layer.__class__.__name__,
                    nb_neurons_per_layer,
                    np.round(MACs / 10 ** 6, 2),
                    params,
                    str(input_shape[1:]),
                    str(output_shape[1:]),
                    '-'])
    return nb_neurons_per_layer, MACs, params

def profile_maxpool(layer, _table):
    input_shape, output_shape = layer.input_shape, layer.output_shape
    k = np.sqrt(np.prod(input_shape[1:])/ np.prod(output_shape[1:]))
    MACs = (k*k-1)*np.prod(layer.output_shape[1:])
    nb_neurons_per_layer = np.prod(output_shape[1:])

    _table.add_row([layer.name,
                    layer.__class__.__name__,
                    nb_neurons_per_layer,
                    np.round(MACs / 10 ** 6, 2),
                    0,
                    str(input_shape[1:]),
                    str(output_shape[1:]),
                    str(k)])

    return nb_neurons_per_layer, MACs

def profile_avgpool(layer, _table):
    input_shape, output_shape = layer.input_shape, layer.output_shape
    k = np.sqrt(np.prod(input_shape[1:]) / np.prod(output_shape[1:]))
    nb_neurons_per_layer = np.prod(output_shape[1:])

    #  MACs and FLOPs
    MACs = ((k * k - 1) + 1)* np.prod(layer.output_shape[1:])

    _table.add_row([layer.name,
                    layer.__class__.__name__,
                    nb_neurons_per_layer,
                    np.round(MACs / 10 ** 6, 2),
                    0,
                    str(input_shape[1:]),
                    str(output_shape[1:]),
                    str(k)])
    return nb_neurons_per_layer, MACs

def profile_Input(layer, _table):
    input_shape, output_shape = layer.input_shape, layer.output_shape
    nb_neurons_per_layer = np.prod(output_shape[0][1:])
    return nb_neurons_per_layer


def NN_profiler(model, print_file_model, _table):
    #  _table.field_names = [model_name, "layer name", "Neuron numbers", "MACs (M)", "Parameters (M)", "Input shape", "Output shape", "Kernel"] # 8
    Total_neurons, Total_MACs, Total_params  = 0, 0, 0
    for layer in model.layers:
        # print('=====>>>[ {0}] '.format(layer.__class__.__name__), file=print_file_model)
        if layer.__class__.__name__ == 'InputLayer':
            nb_neurons_per_layer = profile_Input(layer, print_file_model)
            Total_neurons += nb_neurons_per_layer
            _table.add_row([layer.name, layer.__class__.__name__, nb_neurons_per_layer, "-", "-", "-", "-", "-"])

        elif layer.__class__.__name__ == 'Conv2D':
            nb_neurons_per_layer, MACs, params = profile_conv2d(layer, _table)
            Total_neurons += nb_neurons_per_layer
            Total_MACs  += MACs
            Total_params += params

        elif layer.__class__.__name__ == 'DepthwiseConv2D':
            nb_neurons_per_layer, MACs, params = profile_depthwise_conv2d(layer, _table)
            Total_neurons += nb_neurons_per_layer
            Total_MACs += MACs
            Total_params += params

        elif layer.__class__.__name__ == 'Dense':
            nb_neurons_per_layer, MACs, params = profile_dense(layer, _table)
            Total_neurons += nb_neurons_per_layer
            Total_MACs += MACs
            Total_params += params

        elif layer.__class__.__name__ == 'MaxPooling2D':
            nb_neurons_per_layer, MACs = profile_maxpool(layer, _table)
            # Total_neurons += nb_neurons_per_layer
            # Total_MACs  += MACs

        elif layer.__class__.__name__ == 'AveragePooling2D':
            nb_neurons_per_layer, MACs = profile_avgpool(layer, _table)
            # Total_neurons += nb_neurons_per_layer
            # Total_MACs  += MACs

        elif layer.__class__.__name__ == 'GlobalAveragePooling2D':
            nb_neurons_per_layer, MACs = profile_avgpool(layer, _table)
            Total_neurons += nb_neurons_per_layer
            Total_MACs += MACs

        elif layer.__class__.__name__ == 'BatchNormalization':
            nb_neurons_per_layer, MACs, params = profile_BatchNorm2d(layer, _table)
            # Total_neurons += nb_neurons_per_layer
            Total_MACs += MACs
            Total_params += params

        else:
            print("Error: Layer == {} is not supported".format(layer.__class__.__name__))
    return Total_neurons, Total_MACs, Total_params

def output_nb_param(model):
    # np.prod == 乘法
    return sum([np.prod(K.get_value(w).shape) for w in model.weights])

def test(model_name, path):
    if model_name == 'VGG16':
        from tensorflow.keras.applications.vgg16 import VGG16
        model = VGG16(weights='imagenet', include_top=True)

    elif model_name == 'ResNet50':
        from tensorflow.keras.applications.resnet50 import ResNet50
        model = ResNet50(weights='imagenet', include_top=True)

    elif model_name == 'MobilenetV2':
        from tensorflow.keras.applications import MobileNetV2
        model = MobileNetV2(alpha=1.0, include_top=True, weights='imagenet')

    else:
        model = load_model(path)

    model.summary()

    from prettytable import PrettyTable
    _table = PrettyTable()
    _table.field_names = [model_name, "layer name", "Neuron numbers", "MACs (M)", "Parameters", "Input shape", "Output shape", "Kernel"] # 8

    Total_neurons, Total_MACs, Total_params = NN_profiler(model, print_file_model, _table)
    total_param = output_nb_param(model)

    _table.add_row(['Total',
                    '-',
                    str(np.round(Total_neurons/10**6, 2)) + ' M',
                    str(np.round(Total_MACs/10**6, 2)) + ' M',
                    str(np.round(total_param/10**6, 3)) + '/' + str(np.round(Total_params/10**6, 3))+ ' M',
                    '-',
                    '-',
                    '-'])
    print(_table)
    print(_table, file=print_file_model)
    print_file_model.close()

if __name__ == '__main__':
    args = parse_arguments()

    model_name = args.model_name
    path = args.model_path

    os.makedirs('statistics_files', exist_ok=True)
    print_file_model = open('statistics_files/{0}_report.txt'.format(model_name), 'w+')
    test(model_name, path)


# Reference
# Ref: Molchanov P, Tyree S, Karras T, et al. Pruning Convolutional Neural Networks for Resource Efficient Inference[J]. 2016

# python tensorflow2NNProfiler.py --model_name=VGG16 --model_path=models/example_model.h5