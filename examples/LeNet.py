import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from convnet_drawer import Model, Conv2D, MaxPooling2D, Flatten, Dense
from pptx_util import save_model_to_pptx
import config


def main():
    config.inter_layer_margin = 65
    config.channel_scale = 4 / 5

    model = Model(input_shape=(32, 32, 1))
    model.add(Conv2D(6, (5, 5), (1, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, (5, 5)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(10))
    model.save_fig(os.path.splitext(os.path.basename(__file__))[0] + ".svg")
    save_model_to_pptx(model, os.path.splitext(os.path.basename(__file__))[0] + ".pptx")


if __name__ == '__main__':
    main()
