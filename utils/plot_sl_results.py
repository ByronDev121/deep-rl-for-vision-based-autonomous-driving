import json
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_logs_data(x, y, ax):
    modulo = 7
    last = 0

    for r in range(len(x)):
        if r % modulo == 0:
            last = x[r]
        x[r] = last

    max_y = max(y)

    df = pd.DataFrame(
        dict(
            x=x,
            y=y,
            # y=y,
        )
    )

    # Plot the responses for different events and regions
    sns.lineplot(
        x="x",
        y="y",
        ci="sd",
        data=df,
        ax=ax
    )


def build_logs_print_object(x, data):
    # Comment out what you don't want ot print
    return {
        "Epoch": x,
        "Accuracy": data,
    }


def get_logs_data(path):
    with open(
            path,
            encoding="utf8"
    ) as json_file:
        raw = json.load(json_file)
        data = []
        learning_rates = []
        for data_point in raw:
            learning_rates.append(data_point[1])
            data.append(data_point[2])

    return build_logs_print_object(learning_rates, data)


def get_logs_data_path(dir):
    root = str(pathlib.Path('../').resolve())
    path = root + '\\deep_sl\\results\\network_architecture_study\\discrete\\' + dir + '\\val_acc.json'
    return path


def main():
    # sns.set(font_scale=1)
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots()

    for el in [
        'NvidiaCNN',
        'CustomCNN',
        'NatureCNN',
        # 'NvidiaCNN4',
        # 'NvidiaCNN5',
        # 'NvidiaCNN6',
    ]:
        path = get_logs_data_path(el)
        logs_object = get_logs_data(path)

        for key in logs_object:
            x_name = "Epoch"
            y_name = key
            if key != "Epoch":
                plot_logs_data(logs_object["Epoch"], logs_object[key], ax)

    plt.legend(labels=[
        "Nvidia CNN",
        "Custom CNN",
        "Nature CNN",
        # "Random Flip+Random Translate+Random Rotate",
        # "Random Flip+Random Translate+Random Rotate+Random Depth",
        # "Random Flip+Random Translate+Random Rotate+Random Brightness"
    ])

    plt.xlabel(x_name, fontsize=16)
    plt.ylabel(y_name, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


if __name__ == "__main__":
    main()