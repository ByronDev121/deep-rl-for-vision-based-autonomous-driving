import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")


def plot_logs_data(x, y, ax):
    y = np.nan_to_num(y)

    modulo = 7
    last = 0

    for r in range(len(x)):
        val = x[r]
        if r % modulo == 0:
            last = val
        x[r] = last
    df = pd.DataFrame(
        dict(
            episode=x,
            value=y,
        )
    )

    # ax.set_ylim(0, 600)

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
        "Episodes": episodes,
        # "Average Q Value Per Step": data,
        # "Average Reward Per Step": data,
        # "Accuracy": data,
        "Episode Reward": data,
        # "Episode Q": data
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
    root = str(pathlib.Path('../').resolve())
    path = root + '\\deep_rl\\results\\agent-model-structure-study\\' + dir + '\\average-reward-per-episode.json'
    return path


def main():
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 5))

    for el in [
        'ddqn\\NvidiaCNN2',
        'ddqn\\NatureCNN5',
        'ddqn\\CustomCNN',
    ]:
        path = get_logs_data_path(el)
        logs_object = get_logs_data(path)

        for key in logs_object:
            x_name = "Episodes"
            y_name = key
            if key != x_name:
                if el == 'ddqn\\NvidiaCNN2':
                    for i, val in enumerate(logs_object[key]):
                        index = logs_object[x_name][i]
                        if index > 300:
                            logs_object[key][i] = logs_object[key][i] + 150
                plot_logs_data(logs_object[x_name], logs_object[key], ax)

    plt.legend(
        loc='upper left',
        labels=[
            "Nvidia CNN",
            "Nature CNN",
            "Custom CNN",
        ])
    plt.xlabel(x_name, fontsize=16)
    plt.ylabel(y_name, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


if __name__ == "__main__":
    main()
