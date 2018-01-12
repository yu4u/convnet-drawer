import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from convnet_drawer import Model, Conv2D, MaxPooling2D, Flatten, Dense, config
from matplotlib_util import save_model_to_file
from matplotlib import pyplot as plt
plt.xkcd()


def main():
    config.text_size = 16
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
    save_model_to_file(model, os.path.splitext(os.path.basename(__file__))[0] + ".pdf")


if __name__ == '__main__':
    main()
