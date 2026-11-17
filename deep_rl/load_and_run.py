"""Load and display pre-trained model in OpenAI Gym Environment"""

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Allow running this script directly from any directory by putting the repo root on sys.path
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import tensorflow as tf
from deep_rl.a2c.a2c import A2C_
from deep_rl.ddqn.ddqn import DDQN
from keras.backend.tensorflow_backend import set_session
from deep_rl.ppo.ppo import PPO_2

from airsim_gym.gym import AirSimCarEnv


def parse_args(args):
    """
    Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description="Training parameters")
    #
    parser.add_argument(
        "--algorithm",
        type=str,
        default="a2c",
        help="Algorithm to train {ddqn, a2c, ppo}",
    )
    #
    parser.add_argument(
        "--output_activation",
        type=str,
        default="softmax",
        help="Output activation function for the CNN",
    )
    #
    parser.add_argument(
        "--model_type",
        type=str,
        default="NatureCNN",
        help="Policy model architecture {NatureCNN, NvidiaCNN, CustomCNN}. ",
    )
    #
    parser.add_argument("--epsilon", type=int, default=0, help="Epsilon")
    #
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(
            SCRIPT_DIR,
            "results",
            "state-of-the-art-study",
            "a2c",
            "discrete",
            "best_model",
            "best_model.zip",
        ),
        help="Path to the trained model weights to load",
    )
    return parser.parse_args(args)


def keras_session_init():
    """
    Set up tf/keras session - use gpu device
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))


def instantiate_environment():
    """
    Instantiate AirSim Gym Environment using parameters set in arguments from command line input
    """
    env = AirSimCarEnv()
    env.reset()
    return env


def instantiate_algorithm(args):
    """
    Instantiate the selected DRL algorithm based on arguments from command line input
    """
    if args.algorithm == "ddqn":
        return DDQN(args, "")
    elif args.algorithm == "a2c":
        return A2C_("")
    elif args.algorithm == "ppo":
        return PPO_2("")


def drive(algorithm, env):
    # Display agent
    state = env.reset()
    while True:
        a, q = algorithm.policy_action(state)
        state, r, done, _ = env.step(a)
        if done:
            state = env.reset()


def test_model(algorithm, env):
    state, i = env.reset(), 0
    start_time = time.time()
    while i < 5:
        # Display agent
        a, q = algorithm.policy_action(state)
        state, r, done, _ = env.step(a)
        if done:
            state = env.reset()
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
