import matplotlib.pyplot as plt


def visualize(y1, y2, epochs, title="Title", title_x="X", title_y="Y"):
    x = [i for i in range(epochs)]
    fig, ax = plt.subplots()
    ax.plot(x, y1, color="royalblue", label="Training Accuracy")
    ax.plot(x, y2, color="orangered", label="Validation")
    ax.set(xlabel=title_x, ylabel=title_y, title=title)
    ax.grid()
    ax.legend()
    fig.savefig(title + ".png")
    plt.show()
