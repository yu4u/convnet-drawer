from convnet_drawer import *
import matplotlib.pyplot as plt


def save_model_to_file(model, filename):
    model.build()
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.axis('off')
    plt.xlim(model.x, model.x + model.width)
    plt.ylim(model.y + model.height, model.y)

    for feature_map in model.feature_maps + model.layers:
        for obj in feature_map.objects:
            if isinstance(obj, Line):
                if obj.dasharray == 1:
                    linestyle = ":"
                elif obj.dasharray == 2:
                    linestyle = "--"
                else:
                    linestyle = "-"
                plt.plot([obj.x1, obj.x2], [obj.y1, obj.y2], color=[c / 255 for c in obj.color], lw=obj.width,
                         linestyle=linestyle)
            elif isinstance(obj, Text):
                ax1.text(obj.x, obj.y, obj.body, horizontalalignment="center", verticalalignment="bottom",
                         size=2 * obj.size / 3, color=[c / 255 for c in obj.color])

    plt.savefig(filename)
