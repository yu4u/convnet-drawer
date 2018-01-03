import config
import math


class Line:
    def __init__(self, x1, y1, x2, y2, color="black", width=1, dasharray="none"):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.color = color
        self.width = width
        self.dasharray = dasharray

    def get_svg_string(self):
        return '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke-width="{}" stroke-dasharray="{}" stroke="{}"/>\n'.format(
            self.x1, self.y1, self.x2, self.y2, self.width, self.dasharray, self.color)


class Text:
    def __init__(self, x, y, body, color="black", size=20):
        self.x = x
        self.y = y
        self.body = body
        self.color = color
        self.size = size

    def get_svg_string(self):
        return '<text x="{}" y="{}" font-family="arial" font-size="{}px" ' \
               'text-anchor="middle" fill="{}">{}</text>'.format(self.x, self.y, self.size, self.color, self.body)


class Rect:
    def __init__(self, x, y, w, h, color="black", stroke_width=1):
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.color = color
        self.stroke_width = stroke_width

    def get_svg_string(self):
        return '<rect x="{}" y="{}" width="{}" height="{}" stroke-width="{}" stroke="{}"' \
               ' fill="transparent"/>\n'.format(self.x, self.y, self.w, self.h, self.stroke_width, self.color)


class Model:
    def __init__(self, input_shape):
        self.layers = []

        if len(input_shape) != 3:
            raise ValueError("input_shape should be rank 3 but received  {}".format(input_shape))

        self.feature_maps = [FeatureMap(*input_shape)]

    def add_feature_map(self, layer):
        if isinstance(self.feature_maps[-1], FeatureMap):
            h, w = self.feature_maps[-1].h, self.feature_maps[-1].w
            filters = layer.filters if layer.filters else self.feature_maps[-1].c

            if isinstance(layer, GlobalAveragePooling2D):
                self.feature_maps.append(FeatureMap1D(filters))
            elif isinstance(layer, Flatten):
                self.feature_maps.append(FeatureMap1D(h * w * filters))
            else:
                if layer.padding == "same":
                    new_h = math.ceil(h / layer.strides[0])
                    new_w = math.ceil(w / layer.strides[1])
                else:
                    new_h = math.ceil((h - layer.kernel_size[0] + 1) / layer.strides[0])
                    new_w = math.ceil((w - layer.kernel_size[1] + 1) / layer.strides[1])

                self.feature_maps.append(FeatureMap(new_h, new_w, filters))
        else:
            self.feature_maps.append(FeatureMap1D(layer.filters))

    def add(self, layer):
        self.add_feature_map(layer)
        layer.prev_feature_map = self.feature_maps[-2]
        layer.next_feature_map = self.feature_maps[-1]
        self.layers.append(layer)

    def save_fig(self, filename):
        # build
        left = 0

        for feature_map in self.feature_maps:
            right = feature_map.set_objects(left)
            left = right + config.inter_layer_margin

        for i, layer in enumerate(self.layers):
            layer.set_objects()

        # get bounding box
        x = - config.bounding_box_margin - 30
        y = min([f.get_top() for f in self.feature_maps]) - config.text_margin - config.text_size \
            - config.bounding_box_margin
        width = self.feature_maps[-1].right + config.bounding_box_margin * 2 + 30 * 2
        # TODO: automatically calculate the ad-hoc offset "30" from description length
        height = - y * 2 + config.text_size

        # draw
        string = '<svg xmlns="http://www.w3.org/2000/svg" ' \
                 'xmlns:xlink="http://www.w3.org/1999/xlink" width= "{}" height="{}" '.format(width, height) + \
                 'viewBox="{} {} {} {}">\n'.format(x, y, width, height)

        for feature_map in self.feature_maps:
            string += feature_map.get_object_string()

        for layer in self.layers:
            string += layer.get_object_string()

        string += '</svg>'
        f = open(filename, 'w')
        f.write(string)
        f.close()


class FeatureMap:
    def __init__(self, h, w, c):
        self.h = h
        self.w = w
        self.c = c
        self.objects = None
        self.left = None
        self.right = None

    def set_objects(self, left):
        self.left = left
        c_ = math.pow(self.c, config.channel_scale)
        self.right, self.objects = get_rectangular(self.h, self.w, c_, left)
        x = (left + self.right) / 2
        y = self.get_top() - config.text_margin
        self.objects.append(Text(x, y, "{}x{}x{}".format(self.h, self.w, self.c), size=config.text_size))

        return self.right

    def get_object_string(self):
        return get_object_string(self.objects)

    def get_left_for_conv(self):
        return self.left + self.w * config.ratio * math.cos(config.theta) / 2

    def get_top(self):
        return - self.h / 2 + self.w * config.ratio * math.sin(config.theta) / 2

    def get_bottom(self):
        return self.h / 2 - self.w * config.ratio * math.sin(config.theta) / 2

    def get_right_for_conv(self):
        x = self.left + self.w * config.ratio * math.cos(config.theta) / 4
        y = - self.h / 4 + self.w * config.ratio * math.sin(config.theta) / 4

        return x, y


class FeatureMap1D:
    def __init__(self, c):
        self.c = c
        self.objects = None
        self.left = None
        self.right = None

    def set_objects(self, left):
        self.left = left
        c_ = math.pow(self.c, config.channel_scale)
        self.right = left + config.one_dim_width
        # TODO: reflect text length to right
        self.objects = [Rect(left, - c_ / 2, config.one_dim_width, c_)]
        self.objects.append(Text(left + config.one_dim_width / 2, - c_ / 2 - config.text_margin, "{}".format(
            self.c), size=config.text_size))

        return self.right

    def get_object_string(self):
        return get_object_string(self.objects)

    def get_top(self):
        return - math.pow(self.c, config.channel_scale) / 2

    def get_bottom(self):
        return math.pow(self.c, config.channel_scale) / 2


class Layer:
    def __init__(self, filters=None, kernel_size=None, strides=(1, 1), padding="valid"):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.objects = []
        self.prev_feature_map = None
        self.next_feature_map = None
        self.description = None

    def get_description(self):
        return None

    def set_objects(self):
        c = math.pow(self.prev_feature_map.c, config.channel_scale)
        left = self.prev_feature_map.get_left_for_conv()
        start1 = (left + c,
                  -self.kernel_size[0] + self.kernel_size[1] * config.ratio * math.sin(config.theta) / 2
                  + self.kernel_size[0] / 2)
        start2 = (left + c + self.kernel_size[1] * config.ratio * math.cos(config.theta),
                  -self.kernel_size[1] * config.ratio * math.sin(config.theta) / 2 + self.kernel_size[0] / 2)
        end = self.next_feature_map.get_right_for_conv()
        left, self.objects = get_rectangular(self.kernel_size[0], self.kernel_size[1], c, left, color="blue")
        self.objects.append(Line(start1[0], start1[1], end[0], end[1], color="blue", dasharray="none"))
        self.objects.append(Line(start2[0], start2[1], end[0], end[1], color="blue", dasharray="none"))

        x = (self.prev_feature_map.right + self.next_feature_map.left) / 2
        y = max(self.prev_feature_map.get_bottom(), self.next_feature_map.get_bottom()) + config.text_margin \
            + config.text_size

        for i, description in enumerate(self.get_description()):
            self.objects.append(Text(x, y + i * config.text_size, "{}".format(description), size=config.text_size))

    def get_object_string(self):
        return get_object_string(self.objects)


class Conv2D(Layer):
    def get_description(self):
        return ["conv{}x{}, {}".format(self.kernel_size[0], self.kernel_size[1], self.filters),
                "stride {}".format(self.strides)]


class PoolingLayer(Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding="valid"):
        if not strides:
            strides = pool_size
        super().__init__(kernel_size=pool_size, strides=strides, padding=padding)


class AveragePooling2D(PoolingLayer):
    def get_description(self):
        return ["avepool{}x{}".format(self.kernel_size[0], self.kernel_size[1]),
                "stride {}".format(self.strides)]


class MaxPooling2D(PoolingLayer):
    def get_description(self):
        return ["maxpool{}x{}".format(self.kernel_size[0], self.kernel_size[1]),
                "stride {}".format(self.strides)]


class GlobalAveragePooling2D(Layer):
    def __init__(self):
        super().__init__()

    def get_description(self):
        return ["global avepool"]

    def set_objects(self):
        x = (self.prev_feature_map.right + self.next_feature_map.left) / 2
        y = max(self.prev_feature_map.get_bottom(), self.next_feature_map.get_bottom()) + config.text_margin \
            + config.text_size

        for i, description in enumerate(self.get_description()):
            self.objects.append(Text(x, y + i * config.text_size, "{}".format(description), size=config.text_size))


class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def get_description(self):
        return ["flatten"]

    def set_objects(self):
        x = (self.prev_feature_map.right + self.next_feature_map.left) / 2
        y = max(self.prev_feature_map.get_bottom(), self.next_feature_map.get_bottom()) + config.text_margin \
            + config.text_size

        for i, description in enumerate(self.get_description()):
            self.objects.append(Text(x, y + i * config.text_size, "{}".format(description), size=config.text_size))


class Dense(Layer):
    def __init__(self, units):
        super().__init__(filters=units)

    def get_description(self):
        return ["dense"]

    def set_objects(self):
        x1 = self.prev_feature_map.right
        y11 = - math.pow(self.prev_feature_map.c, config.channel_scale) / 2
        y12 = math.pow(self.prev_feature_map.c, config.channel_scale) / 2
        x2 = self.next_feature_map.left
        y2 = - math.pow(self.next_feature_map.c, config.channel_scale) / 4
        self.objects.append(Line(x1, y11, x2, y2, color="blue", dasharray=2))
        self.objects.append(Line(x1, y12, x2, y2, color="blue", dasharray=2))

        x = (self.prev_feature_map.right + self.next_feature_map.left) / 2
        y = max(self.prev_feature_map.get_bottom(), self.next_feature_map.get_bottom()) + config.text_margin \
            + config.text_size

        for i, description in enumerate(self.get_description()):
            self.objects.append(Text(x, y + i * config.text_size, "{}".format(description), size=config.text_size))


def get_rectangular(h, w, c, dx=0, color="black"):
    p = [[0, -h],
         [w * config.ratio * math.cos(config.theta), -w * config.ratio * math.sin(config.theta)],
         [c, 0]]

    dy = w * config.ratio * math.sin(config.theta) / 2 + h / 2
    right = dx + w * config.ratio * math.cos(config.theta) + c
    lines = []

    for i, [x1, y1] in enumerate(p):
        for x2, y2 in [[0, 0], p[(i + 1) % 3]]:
            for x3, y3 in [[0, 0], p[(i + 2) % 3]]:
                lines.append(Line(x2 + x3 + dx, y2 + y3 + dy, x1 + x2 + x3 + dx, y1 + y2 + y3 + dy,
                                  color=color, dasharray="none"))

    for i in [1, 6, 8]:
        lines[i].dasharray = "1"

    return right, lines


def get_object_string(lines):
    return "".join([line.get_svg_string() for line in lines])


def main():
    model = Model(input_shape=(128, 128, 3))
    model.add(Conv2D(32, (11, 11), (2, 2), padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (7, 7), padding="same"))
    model.add(AveragePooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.save_fig("test.svg")


if __name__ == '__main__':
    main()
