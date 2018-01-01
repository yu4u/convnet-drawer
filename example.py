from convnet_drawer import Model, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense


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
    model.save_fig("example.svg")


if __name__ == '__main__':
    main()
