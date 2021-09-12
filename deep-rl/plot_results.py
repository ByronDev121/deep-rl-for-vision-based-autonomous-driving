import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme(style="darkgrid")


def plot_logs_data(x, y, x_name, y_name):
    y = np.nan_to_num(y)

    modulo = 7
    last = 0
    for r in range(len(x)):
        r = r + 1
        if r % modulo == 0:
            last = r
        x[r-1] = last

    df = pd.DataFrame(
        dict(
            episode=x,
            value=y,
        )
    )

    # Plot the responses for different events and regions
    sns.relplot(
        x="episode",
        y="value",
        kind="line",
        ci="sd",
        data=df
    )


    # plt.subplots(1, 1)
    plt.plot(x, y)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


def build_logs_print_object(episodes, data):
    # Comment out what you don't want ot print
    return {
        "Episode": episodes,
        # "Average Q Value Per Step": data,
        # "Average Reward Per Step": data,
        "Average Q Value": data,
        # "Average Reward": data
    }


def get_logs_data():
    root = str(pathlib.Path().resolve())
    path = root + '/results/ddqn/average-q-per-episode.json'
    with open(
            path,
            encoding="utf8"
    ) as json_file:
        raw = json.load(json_file)
        data = []
        episodes = []
        for data_point in raw:
            episodes.append(data_point[1])
            data.append(data_point[2])

    return build_logs_print_object(episodes, data)


def main():
    logs_object = get_logs_data()

    for key in logs_object:
        # print(key, '->', logs_object[key])
        if key != "Episode":
            plot_logs_data(logs_object["Episode"], logs_object[key], "Episode", key)


if __name__ == "__main__":
    main()