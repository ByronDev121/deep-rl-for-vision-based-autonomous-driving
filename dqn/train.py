from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from keras.optimizers import Adam
import numpy as np
from PIL import Image
from configparser import ConfigParser
import os
from os.path import join, exists

from gym_airsim.airsim_car_env import AirSimCarEnv

from CNN import CNN

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


class AirSimCarProcessor(Processor):

    def __init__(self, window_length, input_shape):
        self.window_length = window_length
        self.input_shape = input_shape

    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(self.input_shape).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == self.input_shape
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


def build_callbacks(env_name):
    log_dir = 'logs'
    if not exists(log_dir):
        os.makedirs(log_dir)
    
    checkpoint_weights_filename = join(log_dir, 'dqn_' + env_name + '_weights_{step}.h5f')
    log_filename = join(log_dir, 'dqn_{}_log.json'.format(env_name))
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=25000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    return callbacks


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    set_session(tf.Session(config=config))

    config = ConfigParser()
    config.read('config.ini')
    num_actions = int(config['car_agent']['actions'])

    window_length = 4
    input_shape = (84, 84)
    np.random.seed(123)

    cnn = CNN()
    cnn.create_model(window_length, input_shape, num_actions)
    # cnn.model.load_weights("logs/dqn_AirSimCarRL_weights_1000000.h5f")
    env = AirSimCarEnv()

    memory = SequentialMemory(limit=50000, window_length=window_length)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                                  value_min=.1, value_test=.05, nb_steps=1000000)
    processor = AirSimCarProcessor(window_length, input_shape)

    dqn = DQNAgent(model=cnn.model, nb_actions=num_actions, policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.0001), metrics=['mae'])

    callbacks = build_callbacks('AirSimCarRL')

    dqn.fit(env, nb_steps=2000000,
            visualize=False,
            verbose=2,
            callbacks=callbacks)


if __name__ == '__main__':
    main()
