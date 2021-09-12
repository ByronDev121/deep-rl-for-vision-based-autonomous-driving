import json
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme(style="darkgrid")


def plot_logs_data(x, y, x_name, y_name):
    y = np.nan_to_num(y)

    modulo = 3
    last = 0
    for r in range(len(x)):
        r = r + 1
        if r % modulo == 0:
            last = r
        x[r-1] = (last * 5)

    dfdict=dict([(x, y) for i in (x_name, y_name)])

    df = pd.DataFrame(
        dfdict
    )

    # Plot the responses for different events and regions
    sns.relplot(
        x=x_name,
        y=y_name,
        kind="line",
        ci="sd",
        data=df
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    name="DQN - ".format(y_name, "Vs.", x_name),
                    line_color='dimgray',
                    opacity=0.8))

    fig.update_layout(
        title="DQN - ".format(y_name, "Vs.", x_name),
        xaxis_title=x_name,
        yaxis_title=y_name,
        legend=go.layout.Legend(
            x=0,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color="black"
            ),
            bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=1
        ),
        font=dict(
            family="Courier New, monospace",
            size=18,
        )
    )

    # fig.show()

    # plt.subplots(1, 1)
    # plt.plot(x, y)
    # plt.xlabel(x_name)
    # plt.ylabel(y_name)
    plt.show()


def build_logs_print_object(episode_reward):
    # Comment out what you don't want ot print
    return {
        "Episode": [x for x in range(len(episode_reward))],
        "Episode Reward": episode_reward
    }


def get_logs_data():
    with open(
            'H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Deep_RL\\results\\DDQN_ENV_Basic_Training_Track_NB_EP_5000_BS_64_LR_0.00025.json',
            encoding="utf8"
    ) as json_file:
        data = json.load(json_file)
        episode_reward = [x[2] for x in data]

    return build_logs_print_object(episode_reward)


def main():
    logs_object = get_logs_data()

    for key in logs_object:
        # print(key, '->', logs_object[key])
        if key != "Episode":
            plot_logs_data(logs_object["Episode"], logs_object[key], "Episode", key)


if __name__ == "__main__":
    main()
