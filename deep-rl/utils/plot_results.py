import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")


def plot_logs_data(x, y, ax):
    y = np.nan_to_num(y)

    max = y.max()
    y = y / max

    modulo = 7
    last = 0
    for r in range(len(x)):
        r = r + 1
        if r % modulo == 0:
            last = r
        x[r - 1] = last

    df = pd.DataFrame(
        dict(
            episode=x,
            value=y,
        )
    )

    # Plot the responses for different events and regions
    sns.lineplot(
        x="episode",
        y="value",
        ci="sd",
        data=df,
        ax=ax
    )


def build_logs_print_object(episodes, data):
    # Comment out what you don't want ot print
    return {
        "Episode": episodes,
        # "Average Q Value Per Step": data,
        # "Average Reward Per Step": data,
        "Average Q Value": data,
        # "Average Reward": data
    }


def get_logs_data(path):
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


def get_logs_data_path(dir):
    root = str(pathlib.Path().resolve())
    path = root + '/results/reward-function-ablation-study/' + dir + '/average-q-per-episode.json'
    return path


def main():
    fig, ax = plt.subplots()

    for el in [
        'ddqn',
        'ddqn4',
        'ddqn2',
        'ddqn3',
        'ddqn6',
        'ddqn5',
    ]:
        path = get_logs_data_path(el)
        logs_object = get_logs_data(path)

        for key in logs_object:
            x_name = "Episode"
            y_name = key
            if key != "Episode":
                plot_logs_data(logs_object["Episode"], logs_object[key], ax)

    plt.legend(labels=[
        "Terminal state",
        "Terminal state + Car angle",
        "Terminal state + Nearest waypoint",
        "Terminal state + Center of track",
        "Terminal state +  Car angle + Nearest waypoint",
        "Terminal state +  Car angle + Center of track",
    ])
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


if __name__ == "__main__":
    main()
