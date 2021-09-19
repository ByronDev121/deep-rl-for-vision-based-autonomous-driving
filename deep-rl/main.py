""" Deep RL Algorithms for autonomous driving in AirSim
"""

import sys
import argparse
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from ddqn.ddqn import DDQN
# from a2c.a2c import A2C
# from ppo.ppo import PPO

from airsim_gym.gym import AirSimCarEnv
from utils.path import get_export_path


def parse_args(args):
    """
    Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--algorithm', type=str, default='ddqn',
                        help="APolicy algorithm used to train the autonomous driving agent [ddqn, a2c, ppo]")
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
    parser.add_argument('--train_in_loop', type=bool, default=False,
                        help="Train q-network in a loop on separate thread")
    #
    parser.add_argument('--nb_steps_per_train_iter', type=int, default=4,
                        help="Number of actions selected by the agent between successive stochastic gradient decent ("
                             "SGD) updates. Using a value of 4 results in the agent selecting 4 actions between each "
                             "pair of successive updates")
    #
    parser.add_argument('--nb_episodes', type=int, default=1000,
                        help="Number of training episodes")
    #
    parser.add_argument('--target_network_update', type=int, default=250,
                        help="The frequency (measured in the number of parameter updates) in which the target network "
                             "is updated")
    #
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Number of training cases over which each SGD update is computed")
    #
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help="The learning rate used by the Adam optimizer for each SGD step")
    #
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Discount factor used in the Q-learning update")
    #
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help="Initial value of epsilon in epsilon-greedy exploration")
    #
    parser.add_argument('--epsilon_decay', type=float, default=0.99,
                        help="epsilon decay per episode in epsilon-greedy exploration")
    #
    parser.add_argument('--epsilon_final', type=float, default=0.1,
                        help="Final value of epsilon in epsilon-greedy exploration")
    #
    parser.add_argument('--replay_buffer_size', type=int, default=10000,
                        help="Reply buffer size")
    #
    parser.add_argument('--replay_start_size', type=int, default=2500,
                        help="Use uniform random policy for this may steps before sampling replay buffer to train agent")
    #
    parser.add_argument('--augment', type=bool, default=False,
                        help="Augment data with noise before adding to replay bugger")
    #
    parser.add_argument('--track', type=str, default='basic_training_track',
                        help="AirSim Track")
    return parser.parse_args(args)


def keras_session_init(save_dir):
    """
    Set up tf/keras session - use gpu device
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    return tf.summary.FileWriter("{}/tensorboard/".format(save_dir))


def instantiate_environment():
    """ Instantiate AirSim Gym Environment using parameters set in arguments from command line input
    """
    env = AirSimCarEnv()
    env.reset()
    return env


def instantiate_algorithm(args, save_dir):
    """
    Instantiate the selected DRL algorithm based on arguments from command line input
    """
    if args.algorithm == "ddqn":
        return DDQN(args, save_dir)
    # elif args.algorithm == "a2c":
    #     return A2C(args)
    # elif args.algorithm == "ppo":
    #     return PPO(args)


def main(args=None):
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    save_dir = get_export_path(args)

    # Set tf/keras session
    summary_writer = keras_session_init(save_dir)

    # Instantiate Environment
    env = instantiate_environment()

    # Instantiate Algorithm
    algorithm = instantiate_algorithm(args, save_dir)

    # Train Algorithm
    algorithm.train(env, summary_writer)

    # Close Environment
    env.close()


if __name__ == "__main__":
    main()
