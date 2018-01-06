#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D


def get_model():
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), input_shape=(227, 227, 3)))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(Conv2D(256, (5, 5), padding="same"))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), padding="same"))
    model.add(Conv2D(384, (3, 3), padding="same"))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Dense(4096))
    model.add(Dense(1000))
    return model
