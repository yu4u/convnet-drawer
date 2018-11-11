import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from convnet_drawer import Model, Conv2D, Deconv2D
from pptx_util import save_model_to_pptx
import config


def main():
    config.channel_scale = 2 / 3

    model = Model(input_shape=(256, 256, 3))
    model.add(Conv2D(16, (3, 3), (2, 2), padding="same"))
    model.add(Conv2D(32, (3, 3), (2, 2), padding="same"))
    model.add(Conv2D(64, (3, 3), (2, 2), padding="same"))
    model.add(Conv2D(128, (3, 3), (2, 2), padding="same"))
    model.add(Conv2D(256, (3, 3), (2, 2), padding="same"))
    model.add(Deconv2D(128, (3, 3), (2, 2), padding="same"))
    model.add(Deconv2D(64, (3, 3), (2, 2), padding="same"))
    model.add(Deconv2D(32, (3, 3), (2, 2), padding="same"))
    model.add(Deconv2D(16, (3, 3), (2, 2), padding="same"))
    model.add(Deconv2D(3, (3, 3), (2, 2), padding="same"))
    model.save_fig(os.path.splitext(os.path.basename(__file__))[0] + ".svg")
    save_model_to_pptx(model, os.path.splitext(os.path.basename(__file__))[0] + ".pptx")


if __name__ == '__main__':
    main()
