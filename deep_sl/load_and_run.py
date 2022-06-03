""" Load and display pre-trained model in OpenAI Gym Environment
"""

import os
import sys
import time
import argparse
from deep_sl.cnn import CNN
import numpy as np
import cv2
import pandas as pd
from configparser import ConfigParser
from os.path import dirname, abspath, join
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from utils.image_processing import ImageProcessing

from airsim_gym.gym import AirSimCarEnv


def parse_args(args):
    """ Parse arguments from command line input
    """
    """
        Parse arguments from command line input
        """
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--model_type', type=str, default='NvidiaCNN',
                        help="CNN model trained")
    #
    parser.add_argument('--augment', type=bool, default=False,
                        help="Augment data with noise before adding to replay bugger")
    #
    parser.add_argument('--model_path', type=str,
                        default='./results/augmentation_ablation_study/continuous/NvidiaCNN4/best_model/best_model.h5',
                        help="Path to model")
    #
    parser.add_argument(
        '--output_activation',
        type=str,
        default='linear',
        # default='softmax',
        help="Output activation function for the CNN")
    #

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
    cnn = CNN(args, save_dir, False)
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


def getConfig():
    config = ConfigParser()
    config.read(join(dirname(abspath(__file__)), '../airsim_gym/config.ini'))

    state_height = int(config['car_agent']['state_height'])
    state_width = int(config['car_agent']['state_width'])
    consecutive_frames = int(config['car_agent']['consecutive_frames'])

    return state_height, state_width, consecutive_frames


def y2indicator(steering_angle, maximum=0.3):
    Y_ind = steering_angle * maximum
    return Y_ind


def drive(args=None):
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    state_height, state_width, consecutive_frames = getConfig()

    process = ImageProcessing(state_height, state_width, consecutive_frames)

    # Instantiate Environment
    env = instantiate_environment()

    # Instantiate Algorithm
    model = instantiate_model(args, None)

    model.load_weights(args.model_path)

    # Display agent
    state, i = env.reset(), 0

    state = process.preprocess(state, True)

    start_time = time.time()

    while i < 5:
        cv2.imshow("Observation", state)
        cv2.waitKey(5)
        a = model.predict((np.array(state).reshape(-1, *state.shape)))
        # action = np.argmax(a)
        action = y2indicator(a)
        state, r, done, _ = env.step(action)
        state = process.preprocess(state, False)
        if done:
            env.reset()
            i += 1

    elapsed_time = time.time() - start_time
    autonomy = (1 - (5*6)/elapsed_time) * 100
    print("Autonomy percentage: {}%".format(autonomy))


# def test_model(args=None):
#     if args is None:
#         args = sys.argv[1:]
#     args = parse_args(args)
#
#     config = ConfigParser()
#     config.read(join(dirname(dirname(abspath(__file__))), 'airsim_gym', 'config.ini'))
#
#     act_dim = int(config['car_agent']['act_dim'])
#     max_steering_angle = float(config['car_agent']['max_steering_angle'])
#     steering_values = [0] * act_dim
#     discrete_delta = max_steering_angle / ((act_dim - 1) / 2)
#     for i in range(act_dim):
#         steering_values[i] = -max_steering_angle + (i * discrete_delta)
#
#     # Instantiate Environment
#     X, y = load_data(args)
#
#     # Instantiate Algorithm
#     model = instantiate_model(args, None)
#
#     model.load_weights(args.model_path)
#
#     for x in range(100):
#         rand = np.random.random()
#         element_index = int(rand * len(X))
#
#         image_name = X[element_index][0]
#
#         ground_truth_a = y[element_index]
#
#         image = load_image(args.data_path, image_name)
#
#         predicted_a = model.predict((np.array(image).reshape(-1, *image.shape)))
#
#         print("\n \nGround truth:", ground_truth_a, "\nPredicted: ", steering_values[np.argmax(predicted_a)])


if __name__ == "__main__":
    keras_session_init()
    drive()
    # test_model()
