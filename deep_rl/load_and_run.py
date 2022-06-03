""" Load and display pre-trained model in OpenAI Gym Environment
"""

import argparse
import sys
import time

import tensorflow as tf
from deep_rl.a2c.a2c import A2C_
from deep_rl.ddqn.ddqn import DDQN
from keras.backend.tensorflow_backend import set_session
from deep_rl.ppo.ppo import PPO_2

from airsim_gym.gym import AirSimCarEnv


def parse_args(args):
    """ Parse arguments from command line input
    """
    """
        Parse arguments from command line input
        """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--algorithm', type=str, default='ddqn',
                        help="Algorithm to train {ddqn, a2c, ppo}")
    #
    parser.add_argument('--output_activation', type=str, default='linear',
                        help="Output activation function for the CNN")
    #
    parser.add_argument('--model_type', type=str, default='CustomCNN',
                        help="Policy model to train {ddqn, a2c, CustomCNN}")
    #
    parser.add_argument('--epsilon', type=int, default=0,
                        help="Epsilon")
    #
    parser.add_argument('--model_path', type=str,
                        default='./results/agent-model-structure-study/ddqn/CustomCNN/best_model/best_model.h5',
                        help="AirSim Track")
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


def instantiate_algorithm(args):
    """
    Instantiate the selected DRL algorithm based on arguments from command line input
    """
    if args.algorithm == "ddqn":
        return DDQN(args, '')
    elif args.algorithm == "a2c":
        return A2C_('')
    elif args.algorithm == "ppo":
        return PPO_2('')


def drive(algorithm, env):
    # Display agent
    state = env.reset()
    while True:
        a, q = algorithm.policy_action(state)
        state, r, done, _ = env.step(a)
        if done:
            env.reset()


def test_model(algorithm, env):
    state, i = env.reset(), 0
    start_time = time.time()
    while i < 5:
        # Display agent
        a, q = algorithm.policy_action(state)
        state, r, done, _ = env.step(a)
        if done:
            env.reset()
            i += 1

    elapsed_time = time.time() - start_time
    autonomy = (1 - (5 * 6) / elapsed_time) * 100
    print("Autonomy percentage: {}%".format(autonomy))


def main(args=None):
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Set tf/keras session
    keras_session_init()

    # Instantiate Environment
    env = instantiate_environment()

    # Instantiate Algorithm
    algorithm = instantiate_algorithm(args)
    algorithm.load_weights(args.model_path)

    # drive(algorithm, env)
    test_model(algorithm, env)


if __name__ == "__main__":
    main()
