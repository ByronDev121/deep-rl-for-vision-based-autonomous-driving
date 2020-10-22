import json
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt


def plot_logs_data(x, y, x_name, y_name):
    y = np.nan_to_num(y)

    # x = x[2000:]
    # y = y[2000:]
    #
    x = x[:-150]
    y = y[:-150]

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

    fig.show()

    plt.subplots(1, 1)
    plt.plot(x, y)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


def build_logs_print_object(episode, loss, mean_eps, mean_q, episode_reward):
    # Comment out what you don't want ot print
    return {
        "Episode": episode,
        "Loss": loss,
        "Mean Epsilon": mean_eps,
        "Mean Q-Value": mean_q,
        "Episode Reward": episode_reward
    }


def get_logs_data():
    with open(
            'H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\AirSim-RL\\logs\\dqn_AirSimCarRL_log.json',
            encoding="utf8"
    ) as json_file:
        data = json.load(json_file)
        episode = data['episode']
        mean_eps = data['mean_eps']
        loss = data['loss']
        mean_q = data['mean_q']
        episode_reward = data['episode_reward']

    return build_logs_print_object(episode, loss, mean_eps, mean_q, episode_reward)


def main():
    logs_object = get_logs_data()

    for key in logs_object:
        # print(key, '->', logs_object[key])
        if key != "Episode":
            plot_logs_data(logs_object["Episode"], logs_object[key], "Episode", key)


if __name__ == "__main__":
    main()
