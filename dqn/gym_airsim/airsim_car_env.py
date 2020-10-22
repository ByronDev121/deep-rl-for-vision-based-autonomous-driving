from configparser import ConfigParser
import gym
from gym import spaces
import math
import numpy as np
from numpy.linalg import norm
from .car_agent import CarAgent
from os.path import dirname, abspath, join
import sys
sys.path.append('..')
import utils

class AirSimCarEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        config = ConfigParser()
        config.read(join(dirname(dirname(abspath(__file__))), 'config.ini'))

        # Using discrete actions
        #TODO Compute number of actions from other settings
        self.action_space = spaces.Discrete(int(config['car_agent']['actions']))
    
        self.image_height = int(config['airsim_settings']['image_height'])
        self.image_width = int(config['airsim_settings']['image_width'])
        self.image_channels = int(config['airsim_settings']['image_channels'])
        image_shape = (self.image_height, self.image_width, self.image_channels)

        self.track_width = float(config['airsim_settings']['track_width'])

        # Using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)

        self.car_agent = CarAgent()

    def step(self, action):
        # move the car according to the action
        self.car_agent.move(action)

        # compute reward
        reward = self._compute_reward()

        # log info
        info = {}

        # get observation
        observation = self.car_agent.observe()

        collision_info = self.car_agent.simGetCollisionInfo()

        done = False
        if collision_info.has_collided:
            done = True
            reward = - 1

        return observation, reward, done, info

    def reset(self):
        self.car_agent.restart()
        observation = self.car_agent.observe()

        return observation  # reward, done, info can't be included

    def close (self):
        self.car_agent.reset()
        return

    def _compute_reward(self):
        return self.car_agent.simGetClosestWayPointsDist()
