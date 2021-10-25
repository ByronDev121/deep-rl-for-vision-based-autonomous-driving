""" Load and display pre-trained model in OpenAI Gym Environment
"""

import os
import sys
import time
import argparse
from cnn import CNN
import numpy as np
import cv2
import pandas as pd
from configparser import ConfigParser
from os.path import dirname, abspath, join
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from Masters.airsim_gym.gym import AirSimCarEnv


def parse_args(args):
    """ Parse arguments from command line input
    """
    """
        Parse arguments from command line input
        """
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--model_type', type=str, default='NatureCNN',
                        help="CNN model trained")
    #
    parser.add_argument('--augment', type=bool, default=False,
                        help="Augment data with noise before adding to replay bugger")
    #
    parser.add_argument('--track', type=str, default='basic_training_track',
                        help="AirSim Track")
    #
    parser.add_argument('--model_path', type=str,
                        default='./results/NatureCNN5/best_model/best_model.h5',
                        help="Path to model")

    return parser.parse_args(args)


def keras_session_init():
    """
    Set up tf/keras session - use gpu device
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))


def instantiate_environment():
    """ Instantiate AirSim Gym Environment using parameters set in arguments from command line input
    """
    env = AirSimCarEnv()
    env.reset()
    return env


def instantiate_model(args, save_dir):
    """
    Instantiate the selected DRL algorithm based on arguments from command line input
    """
    cnn = CNN(args, save_dir)
    cnn.create_model()
    return cnn


def load_image(data_path, image_name):
    """
    Load RGB images from a file
    """
    image = cv2.imread(
        os.path.join(
            '{}\\images'.format(data_path),
            image_name
        ))
    converted_img = cv2.convertScaleAbs(image, alpha=(255.0))
    return converted_img


def load_data(args):
    """ Load training data where x is a list of image paths and y is a list of the corresponding steering angles
    """
    # read CSV file into a single data frame variable
    data_df = pd.read_csv(
        os.path.join(args.data_path, 'data.csv'),
        delimiter=';',
        names=['throttle', 'steering', 'break', 'speed', 'img']
    )

    X = data_df[['img']].values
    y = data_df['steering'].values

    return X, y


def drive(args=None):
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Instantiate Environment
    env = instantiate_environment()

    # Instantiate Algorithm
    model = instantiate_model(args, None)

    model.load_weights(args.model_path)

    # Display agent
    state, i = env.reset(), 0

    start_time = time.time()

    while i < 5:
        cv2.imshow("Observation", state)
        cv2.waitKey(5)
        state = cv2.convertScaleAbs(state, alpha=(255.0))
        state = state / 256
        a = model.predict((np.array(state).reshape(-1, *state.shape)))
        action = np.argmax(a)
        state, r, done, _ = env.step(action)
        if done:
            env.reset()
            i += 1

    elapsed_time = time.time() - start_time
    autonomy = (1 - (5*6)/elapsed_time) * 100
    print("Autonomy percentage: {}%".format(autonomy))


def test_model(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    config = ConfigParser()
    config.read(join(dirname(dirname(abspath(__file__))), 'airsim_gym', 'config.ini'))

    act_dim = int(config['car_agent']['act_dim'])
    max_steering_angle = float(config['car_agent']['max_steering_angle'])
    steering_values = [0] * act_dim
    discrete_delta = max_steering_angle / ((act_dim - 1) / 2)
    for i in range(act_dim):
        steering_values[i] = -max_steering_angle + (i * discrete_delta)

    # Instantiate Environment
    X, y = load_data(args)

    # Instantiate Algorithm
    model = instantiate_model(args, None)

    model.load_weights(args.model_path)

    for x in range(100):
        rand = np.random.random()
        element_index = int(rand * len(X))

        image_name = X[element_index][0]

        ground_truth_a = y[element_index]

        image = load_image(args.data_path, image_name)

        predicted_a = model.predict((np.array(image).reshape(-1, *image.shape)))

        print("\n \nGround truth:", ground_truth_a, "\nPredicted: ", steering_values[np.argmax(predicted_a)])


if __name__ == "__main__":
    keras_session_init()
    drive()
    # test_model()
