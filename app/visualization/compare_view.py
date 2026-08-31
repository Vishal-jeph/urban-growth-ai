import matplotlib.pyplot as plt


def plot_comparison(image1, image2):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(image1)
    axes[0].set_title("Historical Image")

    axes[1].imshow(image2)
    axes[1].set_title("Recent Image")

    for ax in axes:
        ax.axis("off")

    return fig