import matplotlib.pyplot as plt


def plot_change_heatmap(diff_map):

    fig, ax = plt.subplots(figsize=(8, 8))

    heatmap = ax.imshow(diff_map, cmap="hot")

    ax.set_title("Urban Growth Heatmap")
    ax.axis("off")

    plt.colorbar(heatmap)

    return fig