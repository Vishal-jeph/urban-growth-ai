import matplotlib.pyplot as plt


def plot_ai_prediction(prediction):

    fig, ax = plt.subplots(figsize=(5, 5))

    heatmap = ax.imshow(
        prediction,
        cmap="inferno"
    )

    ax.set_title(
        "AI Confidence Heatmap"
    )

    ax.axis("off")

    plt.colorbar(heatmap)

    return fig