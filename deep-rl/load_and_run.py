""" Load and display pre-trained model in OpenAI Gym Environment
"""

import sys
import argparse
import tensorflow as tf

from ddqn.ddqn import DDQN
# from ddpg.ddpg import DDPG
# from a2c.a2c import A2C

from airsim_gym.gym import AirSimCarEnv

from keras.backend.tensorflow_backend import set_session

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--algorithm', type=str, default='ddqn',
                        help="Algorithm to train {ddqn, a2c, ppo}")
    #
    parser.add_argument('--model_type', type=str, default='NatureCNN',
                        help="Policy model to train {ddqn}")
    #
    parser.add_argument('--double_deep', type=bool, default=True,
                        help="Use Double Deep DQN {ddqn}")
    #
    parser.add_argument('--with_per', type=bool, default=False,
                        help="Use Prioritized Experience Replay (ddqn + PER)")
    #
    parser.add_argument('--with_hrs', type=bool, default=False,
                        help="Use Hindsight Reward Shaping (ddqn + HRS)")
    #
    parser.add_argument('--dueling', type=bool, default=False,
                        help="Use a Dueling Architecture (ddqn)")
    #
    parser.add_argument('--train_in_loop', type=bool, default=True,
                        help="train q-network in a loop on separate thread")
    #
    parser.add_argument('--nb_episodes', type=int, default=10000,
                        help="Number of training episodes")
    #
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size (experience replay)")
    #
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help="Number of training episodes")
    #
    parser.add_argument('--gamma', type=float, default=0.95,
                        help="Number of training episodes")
    #
    parser.add_argument('--epsilon', type=float, default=0,
                        help="Number of training episodes")
    #
    parser.add_argument('--epsilon_decay', type=float, default=0,
                        help="Number of training episodes")
    #
    parser.add_argument('--buffer_size', type=int, default=5000,
                        help="Number of training episodes")
    #
    parser.add_argument('--augment', type=bool, default=False,
                        help="Augment data with noise before adding to replay bugger")
    #
    parser.add_argument('--track', type=str, default='basic_training_track',
                        help="AirSim Track")
    #
    parser.add_argument('--model_path', type=str, default='./results/ddqn6/best_model/best_model.h5',
                        help="AirSim Track")
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


def instantiate_algorithm(args):
    """
    Instantiate the selected DRL algorithm based on arguments from command line input
    """
    if args.algorithm == "ddqn":
        return DDQN(args, '')
    # elif args.algorithm == "ddpg":
    #     return DDPG(args)
    # elif args.algorithm == "a2c":
    #     return A2C(args)


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

    # Display agent
    state, time = env.reset(), 0
    while True:
       a, q = algorithm.policy_action(state)
       state, r, done, _ = env.step(a)
       time += 1
       if done:
           env.reset()


if __name__ == "__main__":
    main()
