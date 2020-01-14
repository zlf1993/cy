from __future__ import division, print_function

import tensorflow as tf


def h_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6


def h_swish(x):
    return x * h_sigmoid(x)


def yolo_conv2d(inputs, filters, kernel_size=3, strides=1):  # stride>1时padding，valid卷积实现same
    cond = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    bn = tf.keras.layers.BatchNormalization()(cond)
    outputs = h_swish(bn)
    return outputs


def darknet53_body(inputs, training=True):
    def res_block(inputs, filters):  # same卷积 先1x1降channel再3x3升回channel，再残差连接
        shortcut = inputs
        net = conv2d(inputs, filters * 1, 1, training=training)
        net = conv2d(net, filters * 2, 3, training=training)

        net = net + shortcut

        return net

    def cond(a, n, filters, net):
        return a < n

    def body(a, n, filters, net):
        a = a + 1
        net = res_block(net, filters)  # 104*104*128
        return a, n, filters, net

    # first two conv2d layers
    net = conv2d(inputs, 32, 3, strides=1, training=training)  # same:416*416*32
    net = conv2d(net, 64, 3, strides=2, training=training)  # padding_valid:208*208*64

    # res_block * 1
    net = res_block(net, 32)  # 208*208*64->same:208*208*32->same:208*208*64

    net = conv2d(net, 128, 3, strides=2, training=training)  # padding_valid:104*104*128

    # res_block * 2
    for i in range(2):
        net = res_block(net, 64)  # 104*104*128

    route_0 = net
    net = conv2d(net, 256, 3, strides=2, training=training)  # padding_valid:52*52*256

    # res_block * 8
    for i in range(8):
        net = res_block(net, 128)  # 52*52*256

    route_1 = net
    net = conv2d(net, 512, 3, strides=2)  # padding_valid:26*26*512

    # res_block * 8
    for i in range(8):
        net = res_block(net, 256)  # 26*26*512

    route_2 = net
    net = conv2d(net, 1024, 3, strides=2, training=training)  # padding_valid:13*13*1024

    # res_block * 4
    for i in range(4):
        net = res_block(net, 512)  # 13*13*1024
    route_3 = net

    return route_0, route_1, route_2, route_3


def yolo_block(inputs, filters, reduce=True):  # 1x1->3x3->1x1->3x3->1x1(route)------->3x3(net)
    net = yolo_conv2d(inputs, filters * 1, 1)
    net = yolo_conv2d(net, filters * 2, 3)
    net = yolo_conv2d(net, filters * 1, 1)
    net = yolo_conv2d(net, filters * 2, 3)
    net = yolo_conv2d(net, filters * 1, 1)
    route = net
    if reduce is True:
        return route
    else:
        net = yolo_conv2d(net, filters * 2, 3)
        return net


def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[0], out_shape[1]
    # NOTE: here height is the first
    inputs = tf.compat.v1.image.resize_nearest_neighbor(inputs, (new_height, new_width), align_corners=True, name='upsampled')
    return inputs


def h_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6


def h_swish(x):
    return x * h_sigmoid(x)