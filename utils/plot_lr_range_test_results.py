import json
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")


def plot_logs_data(x, y, ax):
    modulo = 7
    last = 0

    for r in range(len(x)):
        val = "%f" % x[r]
        if r % modulo == 0:
            last = val
        x[r] = last

    df = pd.DataFrame(
        dict(
            x=x,
            y=y,
        )
    )

    # df.plot(x='x', y='y', ax=ax)

    # Plot the responses for different events and regions
    sns.lineplot(
        x="x",
        y="y",
        ci="sd",
        data=df,
        ax=ax
    )


def build_logs_print_object(lr, data):
    # Comment out what you don't want ot print
    return {
        "Learning Rate (log Scale)": lr,
        "Loss": data,
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
            learning_rates.append(data_point[0])
            data.append(data_point[1])

    return build_logs_print_object(learning_rates, data)


def get_logs_data_path(file):
    root = str(pathlib.Path('../').resolve())
    path = root + '\\deep-sl\\results\\network_architecture\\CustomCNN\\' + file
    return path


def main():
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set(xscale="symlog")
    # ax.set(xscale="log")

    for el in [
        'data_32.json',
        'data_64.json',
        'data_128.json',
    ]:
        path = get_logs_data_path(el)
        logs_object = get_logs_data(path)

        for key in logs_object:
            x_name = "Learning Rate (log Scale)"
            y_name = key
            if key != "Learning Rate (log Scale)":
                plot_logs_data(logs_object["Learning Rate (log Scale)"], logs_object[key], ax)

    plt.legend(labels=[
        "Batch size: 32",
        "Batch size: 64",
        "Batch size: 128"
    ])

    plt.xticks(rotation="90")
    plt.xticks(logs_object['Learning Rate (log Scale)'][::15])

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


if __name__ == "__main__":
    main()