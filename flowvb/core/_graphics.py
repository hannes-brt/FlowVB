import matplotlib.pyplot as plt


def plot_clustering(data, labels, colors=None, dim=(0, 1),
                    title='', output='screen',
                    plot_kwargs=dict(), savefig_kwargs=dict()):

    # Set default colors
    if colors == None:
        colors = [[0.5529, 0.8275, 0.7804],
                  [1.0, 1.0, 0.7020],
                  [0.7451, 0.7294, 0.8549],
                  [0.9843, 0.502, 0.4471],
                  [0.502, 0.6941, 0.8275],
                  [0.9922, 0.7059, 0.3843],
                  [0.7020, 0.8706, 0.4118],
                  [0.9882, 0.8039, 0.898],
                  [0.8510, 0.8510, 0.8510],
                  [0.7373, 0.502, 0.7412],
                  [0.8000, 0.9216, 0.7725],
                  [1.0, 0.9294, 0.4353]]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if title is not '':
        plt.title(title)

    lines = []
    for i in range(max(labels) + 1):
        lines.append(ax.plot(data[labels == i, dim[0]],
                             data[labels == i, dim[1]],
                             color=colors[i], linestyle='', marker='o',
                             **plot_kwargs)[0])

    if output == 'screen':
        plt.show()
    else:
        plt.savefig(output, **savefig_kwargs)
