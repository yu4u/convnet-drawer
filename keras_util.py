#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras_models import AlexNet
from convnet_drawer import Model, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D


def get_dense_obj(class_object, config):
    units = config.get("units", False)
    return class_object(units)


def get_maxpooling2d_obj(class_object, config):
    pool_size = config.get("pool_size", False)
    strides = config.get("strides", False)
    padding = config.get("padding", False)
    return class_object(pool_size, strides, padding)


def get_conv2d_obj(class_object, config):
    filters = config.get("filters", False)
    kernel_size = config.get("kernel_size", False)
    strides = config.get("strides", False)
    padding = config.get("padding", False)
    return class_object(filters, kernel_size, strides, padding)


def is_class_object(class_name):
    return eval(class_name)


def convert_drawer_model(model):
    _input_shape = model.input_shape
    figure = Model(input_shape=_input_shape[1:])
    for config in model.get_config()["layers"]:
        class_name = config.get("class_name", False)
        class_config = config.get("config", False)
        if class_name and class_config:
            class_obj = is_class_object(class_name)
            if class_name == "Conv2D":
                conv_2d = get_conv2d_obj(class_obj, class_config)
                figure.add(conv_2d)
            elif class_name == "MaxPooling2D":
                max_pooling_2d = get_maxpooling2d_obj(class_obj, class_config)
                figure.add(max_pooling_2d)
            elif class_name == "Dense":
                dense = get_dense_obj(class_obj, class_config)
                figure.add(dense)
            else:
                figure.add(class_obj())
        else:
            raise ValueError

    return figure


def main():
    alex_net = AlexNet.get_model()
    f = convert_drawer_model(alex_net)
    f.save_fig("alex_net.svg")


if __name__ == '__main__':
    main()
