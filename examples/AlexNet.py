import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from convnet_drawer import Model, Conv2D, MaxPooling2D, Flatten, Dense
from pptx_util import save_model_to_pptx


def main():
    model = Model(input_shape=(227, 227, 3))
    model.add(Conv2D(96, (11, 11), (4, 4)))
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
    model.save_fig(os.path.splitext(os.path.basename(__file__))[0] + ".svg")
    save_model_to_pptx(model, os.path.splitext(os.path.basename(__file__))[0] + ".pptx")


if __name__ == '__main__':
    main()
