""" Load and display pre-trained model in OpenAI Gym Environment
"""
import math
import time
import keyboard
from configparser import ConfigParser
from os.path import dirname, abspath, join
from airsim_gym.gym import AirSimCarEnv


def instantiate_environment():
    """ Instantiate AirSim Gym Environment using parameters set in arguments from command line input
    """
    env = AirSimCarEnv()
    env.reset()
    return env


def get_controls():
    config = ConfigParser()
    config.read(join(dirname(dirname(abspath(__file__))), 'airsim_gym', 'config.ini'))
    act_dim = int(config['car_agent']['act_dim'])
    max_steering_angle = float(config['car_agent']['max_steering_angle'])
    discrete_delta = max_steering_angle / ((act_dim - 1) / 2)
    steering_values = [0] * act_dim
    for i in range(act_dim):
        steering_values[i] = -max_steering_angle + (i * discrete_delta)
    return steering_values, math.floor(len(steering_values)/2)


def get_action(steering_values, action_index):
    if keyboard.is_pressed('left'):  # if key 'q' is pressed
        print('You Pressed left Key!')
        if action_index > 0:
            print("Index: ", action_index-1)
            time.sleep(0.1)
            return action_index - 1
    if keyboard.is_pressed('right'):  # if key 'q' is pressed
        print('You Pressed right Key!')
        if action_index < len(steering_values)-1:
            print("Index: ", action_index + 1)
            time.sleep(0.1)
            return action_index + 1

    return action_index


def main():
    # Instantiate Environment
    env = instantiate_environment()

    steering_values, action_index = get_controls()

    # Display agent
    state, time = env.reset(), 0
    while True:
        action_index = get_action(steering_values, action_index)

        state, r, done, _ = env.step(action_index)
        time += 1
        if done:
            steering_values, action_index = get_controls()
            env.reset()


if __name__ == "__main__":
    main()
