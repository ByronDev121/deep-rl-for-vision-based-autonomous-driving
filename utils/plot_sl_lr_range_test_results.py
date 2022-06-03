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
        if r % modulo == 0:
            if x[r] != 0:
                last = x[r]
        x[r] = last

    df = pd.DataFrame(
        dict(
            x=x,
            y=y,
        )
    )

    # plt.plot(x,y)

    # Plot the responses for different events and regions
    g = sns.lineplot(
        x="x",
        y="y",
        ci="sd",
        data=df,
        ax=ax
    )

    return g
    # g.set(xscale='log')


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
        for idx, data_point in enumerate(raw):
            if 50 < idx < (len(raw)):
                learning_rates.append(data_point[0])
                data.append(data_point[1])

    return build_logs_print_object(learning_rates, data)


def get_logs_data_path(file):
    root = str(pathlib.Path('../').resolve())
    path = root + '\\deep_sl\\results\\network_architecture_study\\continuous\\LR-Range-Test-NatureCNN\\' + file
    return path


def main():
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 5))



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
                g = plot_logs_data(logs_object["Learning Rate (log Scale)"], logs_object[key], ax)

    g.set(xscale='log')

    plt.legend(labels=[
        "Batch size: 32",
        "Batch size: 64",
        "Batch size: 128"
    ])

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


if __name__ == "__main__":
    main()